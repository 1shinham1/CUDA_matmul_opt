// [공용 헤더] 매크로, 랜덤/변환 헬퍼, 메모리 추적, fp64 CPU 레퍼런스(+캐싱),
// 상대오차 검증, 타이밍 유틸, 그리고 tiled(FMA) forward 커널 자체까지 담는다.
// (CUDA_matmul_opt의 include/gemm.h와 같은 역할 -- 다만 gemm.h는 커널을 담지
// 않는 반면, 여기서는 텐서 코어 커널(01_flash_tc_naive.cu)이 이 tiled 커널을
// "프로세스 안에서 비교할 빠른 기준"으로 재사용해야 해서 launcher까지 포함한다.)
//
// [정밀도] Q,K,V,O는 __half(fp16) 저장, softmax/누적은 fp32 -- PyTorch AMP와
// 동일한 "저장은 fp16, 계산은 fp32" 방식. forward만 다루므로 dQ/dK/dV/dO,
// logsumexp(L) 저장은 전부 없다 (backward가 없으면 저장할 이유가 없다).
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

// ─── 에러 체크 매크로 ─────────────────────────────────────────
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err__));                    \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ─── 기본 벤치마크 설정 ────────────────────────────────────────
// GEMM이 M=K=N=4096을 모든 커널에 고정해 공정 비교하는 것과 같은 취지로,
// BH(batch*heads)/HEAD_DIM을 고정한다. seq_len만 sweep.
#define DEFAULT_BH 32
#define DEFAULT_HEAD_DIM 64
constexpr int WARM_UP = 3;

inline std::vector<int> default_seq_lens() {
    return {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
}

// seq_len이 작을수록 커널 1회 실행이 짧아 GPU 스케줄링 지터에 흔들리기 쉬우므로
// 반복 횟수를 늘리고, 크면 이미 안정적이라 반복을 줄여 전체 벤치마크 시간을 아낀다.
inline int iters_for(int seq_len) {
    if (seq_len <= 1024) return 20;
    if (seq_len <= 4096) return 10;
    return 5;
}

// ─── 메모리 사용량 추적 할당자 ──────────────────────────────────
// FlashAttention의 O(seq_len) 메모리 특성(S/P를 HBM에 안 씀)을 정확한
// 바이트 수로 보여주기 위한 할당자.
struct DeviceAllocTracker {
    size_t current = 0;
    size_t peak = 0;

    void* alloc(size_t bytes) {
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        current += bytes;
        if (current > peak) peak = current;
        return ptr;
    }
    void free(void* ptr, size_t bytes) {
        CUDA_CHECK(cudaFree(ptr));
        current -= bytes;
    }
    void reset_peak() { peak = current; }
    double peak_mb() const { return peak / (1024.0 * 1024.0); }
};

// cudaMalloc을 try/catch처럼 쓰는 대신, cudaMemGetInfo()로 미리 여유가 있는지
// 확인한다. 1.2배 마진은 CUDA 컨텍스트 자체와 다른 프로세스의 사용량을 감안.
inline bool fits_in_gpu(size_t bytes_needed) {
    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    return (double)bytes_needed * 1.2 < (double)free_b;
}

// ─── 랜덤 초기화 / fp16 변환 헬퍼 ────────────────────────────────
inline void fill_random(std::vector<float>& v, unsigned seed, float scale = 1.0f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : v) x = dist(gen) * scale;
}

inline std::vector<__half> to_half(const std::vector<float>& v) {
    std::vector<__half> out(v.size());
    for (size_t i = 0; i < v.size(); i++) out[i] = __float2half(v[i]);
    return out;
}

inline std::vector<float> to_float(const std::vector<__half>& v) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); i++) out[i] = __half2float(v[i]);
    return out;
}

// GPU가 fp16으로 저장하는 순간 반올림되는 값과 CPU 레퍼런스 입력을 맞추기 위한 스냅.
inline void snap_to_half(std::vector<float>& v) {
    for (auto& x : v) x = __half2float(__float2half(x));
}

// ─── GPU 타이밍 (warmup + N_ITERS 평균, CUDA_matmul_opt의 인라인 이벤트 패턴을
// 재사용 가능한 함수로 뺀 것) ─────────────────────────────────────
template <typename Fn>
inline float time_ms_avg(Fn fn, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

// forward attention FLOPs 근사치: QK^T와 PV 두 매트멀, 각 2*BH*N^2*d(FMA는 2 FLOP)
// -- causal이면 위쪽 삼각형이 대략 절반 스킵되므로 절반으로 근사.
inline double attn_gflops(int BH, int seq_len, int head_dim, bool causal, double ms) {
    double flops = 4.0 * (double)BH * (double)seq_len * (double)seq_len * (double)head_dim;
    if (causal) flops *= 0.5;
    return flops / (ms / 1000.0) / 1e9;
}

// ─── CPU 레퍼런스 (fp64, OpenMP 병렬화) ─────────────────────────
// GPU 커널 정확성 검증의 "정답" 기준. O = softmax(QK^T/sqrt(d)) V, causal이면
// 미래 위치는 마스킹. row(=한 쿼리의 점수 벡터)는 각 반복 안에서 지역
// 변수로 선언해 OpenMP 병렬 for 아래에서도 스레드마다 독립적으로 동작한다.
inline void ref_forward(const std::vector<float>& Q, const std::vector<float>& K,
                         const std::vector<float>& V, std::vector<float>& O,
                         int BH, int seq_len, int d, bool causal) {
    double scale = 1.0 / std::sqrt((double)d);
    long total_rows = (long)BH * seq_len;

    #pragma omp parallel for schedule(dynamic)
    for (long idx = 0; idx < total_rows; idx++) {
        int bh = (int)(idx / seq_len);
        int i = (int)(idx % seq_len);
        const float* Qb = &Q[(size_t)bh * seq_len * d];
        const float* Kb = &K[(size_t)bh * seq_len * d];
        const float* Vb = &V[(size_t)bh * seq_len * d];
        float* Ob = &O[(size_t)bh * seq_len * d];
        int jmax = causal ? (i + 1) : seq_len;

        std::vector<double> row(jmax);
        double m = -1e300;
        for (int j = 0; j < jmax; j++) {
            double s = 0.0;
            for (int t = 0; t < d; t++) s += (double)Qb[i * d + t] * (double)Kb[j * d + t];
            s *= scale;
            row[j] = s;
            if (s > m) m = s;
        }
        double l = 0.0;
        for (int j = 0; j < jmax; j++) { row[j] = std::exp(row[j] - m); l += row[j]; }
        for (int t = 0; t < d; t++) {
            double acc = 0.0;
            for (int j = 0; j < jmax; j++) acc += row[j] * (double)Vb[j * d + t];
            Ob[i * d + t] = (float)(acc / l);
        }
    }
}

// A/B/V가 fill_random(seed=1,2,3)으로 항상 같은 값으로 채워진다는 전제로,
// (BH,seq_len,d,causal) 조합별 CPU 레퍼런스를 bin/ 아래에 캐싱한다 --
// gemm.h의 gemm_cpu_cached와 동일한 목적/패턴.
inline void ref_forward_cached(const std::vector<float>& Q, const std::vector<float>& K,
                                const std::vector<float>& V, std::vector<float>& O,
                                int BH, int seq_len, int d, bool causal) {
    char path[256];
    snprintf(path, sizeof(path), "bin/.cpu_ref_flash_%dx%dx%d_c%d.bin", BH, seq_len, d, causal ? 1 : 0);

    FILE* f = fopen(path, "rb");
    if (f) {
        size_t got = fread(O.data(), sizeof(float), O.size(), f);
        fclose(f);
        if (got == O.size()) {
            printf("[CPU 참조값 캐시 사용: %s]\n", path);
            return;
        }
    }

    ref_forward(Q, K, V, O, BH, seq_len, d, causal);

    f = fopen(path, "wb");
    if (f) {
        fwrite(O.data(), sizeof(float), O.size(), f);
        fclose(f);
    }
}

// GPU 결과 vs CPU fp64 레퍼런스의 상대(L2) 오차. tol 미만이면 true.
inline bool verify_against_cpu(const std::vector<float>& O, const std::vector<float>& O_ref,
                                float tol, const char* label) {
    double diff_norm = 0.0, ref_norm = 0.0;
    for (size_t i = 0; i < O.size(); i++) {
        double d = (double)O[i] - (double)O_ref[i];
        diff_norm += d * d;
        ref_norm += (double)O_ref[i] * (double)O_ref[i];
    }
    double rel_err = std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-12);
    bool ok = rel_err < tol;
    printf("  [%s] relative error vs CPU fp64 ref: %.6e %s\n", label, rel_err, ok ? "(OK)" : "(WARNING: large error)");
    return ok;
}

// ===========================================================================
// tiled forward (FlashAttention Algorithm 2) -- 타일링 + online softmax,
// 텐서 코어 없는 스칼라 FMA 반복문. S/P를 HBM에 한 번도 쓰지 않는다.
// 01_flash_tc_naive.cu(Tensor Core)가 이 커널을 "프로세스 안에서 비교할 빠른
// 기준"으로도 재사용한다 (GEMM에서 cuBLAS가 하는 역할).
// ===========================================================================
template <int HEAD_DIM, int BLOCK_N>
__global__ void flash_tiled_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                    const __half* __restrict__ V, __half* __restrict__ O,
                                    int seq_len, float scale, bool causal) {
    extern __shared__ float smem[];
    const int BLOCK_M = blockDim.x;
    float* Qs = smem;
    float* Ks = Qs + BLOCK_M * HEAD_DIM;
    float* Vs = Ks + BLOCK_N * HEAD_DIM;

    const int bh = blockIdx.y;
    const int row = threadIdx.x;
    const int q_idx = blockIdx.x * BLOCK_M + row;

    const __half* Qbh = Q + (size_t)bh * seq_len * HEAD_DIM;
    const __half* Kbh = K + (size_t)bh * seq_len * HEAD_DIM;
    const __half* Vbh = V + (size_t)bh * seq_len * HEAD_DIM;

    for (int t = 0; t < HEAD_DIM; t++)
        Qs[row * HEAD_DIM + t] = (q_idx < seq_len) ? __half2float(Qbh[(size_t)q_idx * HEAD_DIM + t]) : 0.0f;

    float m_i = -INFINITY, l_i = 0.0f;
    float acc[HEAD_DIM];
#pragma unroll
    for (int t = 0; t < HEAD_DIM; t++) acc[t] = 0.0f;

    const int hi = causal ? min(seq_len, blockIdx.x * BLOCK_M + BLOCK_M) : seq_len;
    __syncthreads();

    for (int kv0 = 0; kv0 < hi; kv0 += BLOCK_N) {
        for (int idx = threadIdx.x; idx < BLOCK_N * HEAD_DIM; idx += BLOCK_M) {
            int r = idx / HEAD_DIM, c = idx % HEAD_DIM;
            int kv_idx = kv0 + r;
            Ks[idx] = (kv_idx < seq_len) ? __half2float(Kbh[(size_t)kv_idx * HEAD_DIM + c]) : 0.0f;
            Vs[idx] = (kv_idx < seq_len) ? __half2float(Vbh[(size_t)kv_idx * HEAD_DIM + c]) : 0.0f;
        }
        __syncthreads();

        if (q_idx < seq_len) {
            float scores[BLOCK_N];
            float blk_max = -INFINITY;
#pragma unroll
            for (int c = 0; c < BLOCK_N; c++) {
                int kv_idx = kv0 + c;
                float s = -INFINITY;
                if (kv_idx < seq_len && !(causal && kv_idx > q_idx)) {
                    float dot = 0.0f;
#pragma unroll
                    for (int t = 0; t < HEAD_DIM; t++) dot += Qs[row * HEAD_DIM + t] * Ks[c * HEAD_DIM + t];
                    s = dot * scale;
                }
                scores[c] = s;
                blk_max = fmaxf(blk_max, s);
            }
            float m_new = fmaxf(m_i, blk_max);
            float alpha = expf(m_i - m_new);
            l_i *= alpha;
#pragma unroll
            for (int t = 0; t < HEAD_DIM; t++) acc[t] *= alpha;
#pragma unroll
            for (int c = 0; c < BLOCK_N; c++) {
                float p = expf(scores[c] - m_new);
                l_i += p;
#pragma unroll
                for (int t = 0; t < HEAD_DIM; t++) acc[t] += p * Vs[c * HEAD_DIM + t];
            }
            m_i = m_new;
        }
        __syncthreads();
    }

    if (q_idx < seq_len)
        for (int t = 0; t < HEAD_DIM; t++) O[((size_t)bh * seq_len + q_idx) * HEAD_DIM + t] = __float2half(acc[t] / l_i);
}

template <int HEAD_DIM, int BLOCK_N>
void flash_tiled_launch(const __half* Q, const __half* K, const __half* V, __half* O,
                         int BH, int seq_len, bool causal, int BLOCK_M) {
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    size_t smem_bytes = (size_t)(BLOCK_M + 2 * BLOCK_N) * HEAD_DIM * sizeof(float);
    auto kernel = flash_tiled_kernel<HEAD_DIM, BLOCK_N>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    dim3 grid((seq_len + BLOCK_M - 1) / BLOCK_M, BH);
    kernel<<<grid, BLOCK_M, smem_bytes>>>(Q, K, V, O, seq_len, scale, causal);
}
