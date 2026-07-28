// [00] 표준 어텐션(standard attention) -- FlashAttention을 하나도 안 쓴 기준점.
// FlashAttention 논문이 비교 대상으로 삼는 그 구현이다. 01/02가 가진 것들이
// 전부 빠져 있다:
//   - 커널 퓨전 없음  : QK^T / softmax / PV 가 각각 별도 커널 (런치 3회)
//   - 타일링 없음     : shared memory를 안 쓰고 스레드가 global에서 직접 읽음
//   - online softmax 없음 : 행을 세 번 훑어(max -> exp합 -> 정규화) 계산
//
// 셋이 빠진 대가가 곧 FlashAttention의 존재 이유다: N×N 점수 행렬 S를 HBM에
// 통째로 올려야 한다. 01/02는 S를 HBM에 단 한 번도 쓰지 않는(레지스터/shared
// 안에서 소비하는) 반면, 여기서는 S를 쓰고 -> 다시 읽어 softmax하고 -> 다시
// 읽어 PV를 한다. 메모리가 seq_len에 대해 O(N^2)로 늘어, 이 프로젝트 기준
// 설정(BH=32, seq_len=4096)에서 S 하나가 2 GiB를 먹는다 (Q/K/V/O 전부 합쳐
// 64 MiB인 것과 대조).
//
// [causal 주의] 마스킹된 위치도 -INF를 채워 넣을 뿐 계산을 건너뛰지 않는다
// (타일링이 없으니 블록 단위로 스킵할 방법이 없다). 그래서 causal일 때
// attn_gflops()가 FLOP을 절반으로 세는 것과 달리 실제 일은 그대로라, 리포트
// 되는 GFLOPS가 non-causal보다 낮게 나오는 게 정상이다.
#include "flash.h"
#include <fstream>
#include <string>

// ── 1단계: S = (Q @ K^T) * scale, causal이면 위쪽 삼각형에 -INF ──────────
// 스레드 하나가 S의 원소 하나를 맡아 head_dim만큼 내적한다. shared memory를
// 안 쓰므로 같은 Q행/K행을 여러 스레드가 매번 global에서 다시 읽는다.
template <int HEAD_DIM>
__global__ void qk_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                          float* __restrict__ S, int seq_len, float scale, bool causal) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;  // key 위치
    const int i = blockIdx.y * blockDim.y + threadIdx.y;  // query 위치
    const int bh = blockIdx.z;
    if (i >= seq_len || j >= seq_len) return;

    float* Srow = S + ((size_t)bh * seq_len + i) * seq_len;
    if (causal && j > i) { Srow[j] = -INFINITY; return; }

    const half* Qb = Q + (size_t)bh * seq_len * HEAD_DIM;
    const half* Kb = K + (size_t)bh * seq_len * HEAD_DIM;
    float acc = 0.0f;
    for (int t = 0; t < HEAD_DIM; t++)
        acc += __half2float(Qb[(size_t)i * HEAD_DIM + t]) * __half2float(Kb[(size_t)j * HEAD_DIM + t]);
    Srow[j] = acc * scale;
}

// ── 2단계: 행별 softmax, S를 P로 제자리 덮어쓰기 ────────────────────────
// 블록 하나가 행 하나를 맡는다. online softmax(한 번 훑으며 max/합을 갱신)가
// 아니라 교과서적인 3-pass: 최댓값 -> exp 합 -> 정규화. 행이 HBM에 있으니
// 패스마다 HBM을 왕복한다(= 이 커널이 존재하는 것 자체가 FlashAttention이
// 없애려는 트래픽이다).
template <int NTHREADS>
__global__ void softmax_kernel(float* __restrict__ S, int seq_len) {
    static_assert((NTHREADS & (NTHREADS - 1)) == 0, "NTHREADS must be a power of 2 for the tree reduction");
    __shared__ float red[NTHREADS];
    float* Srow = S + (size_t)blockIdx.x * seq_len;  // blockIdx.x = (bh, i)를 flatten
    const int tid = threadIdx.x;

    // (1) 행 최댓값
    float m = -INFINITY;
    for (int j = tid; j < seq_len; j += NTHREADS) m = fmaxf(m, Srow[j]);
    red[tid] = m;
    __syncthreads();
    for (int s = NTHREADS / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float row_max = red[0];
    __syncthreads();

    // (2) exp 합 (행을 다시 읽는다)
    float l = 0.0f;
    for (int j = tid; j < seq_len; j += NTHREADS) l += expf(Srow[j] - row_max);
    red[tid] = l;
    __syncthreads();
    for (int s = NTHREADS / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float row_sum = red[0];
    __syncthreads();

    // (3) 또 한 번 읽어서 정규화 (exp를 저장 안 했으므로 다시 계산)
    for (int j = tid; j < seq_len; j += NTHREADS) Srow[j] = expf(Srow[j] - row_max) / row_sum;
}

// ── 3단계: O = P @ V ────────────────────────────────────────────────────
// 스레드 하나가 O의 원소 하나를 맡아 P의 행 하나(길이 seq_len)를 통째로 다시
// 읽는다 -- 1단계에서 HBM에 써둔 걸 되가져오는 비용이 여기서 또 발생한다.
template <int HEAD_DIM>
__global__ void pv_kernel(const float* __restrict__ P, const half* __restrict__ V,
                          half* __restrict__ O, int seq_len) {
    const int t = threadIdx.x;                             // head_dim 방향 (blockDim.x == HEAD_DIM)
    const int i = blockIdx.y * blockDim.y + threadIdx.y;   // query 위치
    const int bh = blockIdx.z;
    if (i >= seq_len) return;

    const float* Prow = P + ((size_t)bh * seq_len + i) * seq_len;
    const half* Vb = V + (size_t)bh * seq_len * HEAD_DIM;
    float acc = 0.0f;
    for (int j = 0; j < seq_len; j++)
        acc += Prow[j] * __half2float(Vb[(size_t)j * HEAD_DIM + t]);
    O[((size_t)bh * seq_len + i) * HEAD_DIM + t] = __float2half(acc);
}

// S는 호출자가 할당해 넘긴다 -- 이 커널의 핵심 비용(O(N^2) HBM)이 호출부에서
// 눈에 보여야 하고, 벤치마크 반복마다 재할당하지 않기 위해서다.
template <int HEAD_DIM>
void attention_naive_launch(const __half* Q, const __half* K, const __half* V, __half* O,
                            float* S, int BH, int seq_len, bool causal) {
    static_assert(HEAD_DIM == 64, "attention_naive_launch is only implemented for HEAD_DIM=64");
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    constexpr int SM_THREADS = 256;

    dim3 qk_block(16, 16);
    dim3 qk_grid((seq_len + 15) / 16, (seq_len + 15) / 16, BH);
    qk_kernel<HEAD_DIM><<<qk_grid, qk_block>>>(
        reinterpret_cast<const half*>(Q), reinterpret_cast<const half*>(K), S, seq_len, scale, causal);

    softmax_kernel<SM_THREADS><<<BH * seq_len, SM_THREADS>>>(S, seq_len);

    dim3 pv_block(HEAD_DIM, 4);
    dim3 pv_grid(1, (seq_len + 3) / 4, BH);
    pv_kernel<HEAD_DIM><<<pv_grid, pv_block>>>(
        S, reinterpret_cast<const half*>(V), reinterpret_cast<half*>(O), seq_len);
}

// ─────────────────────────────────────────────────────────────
constexpr int HEAD_DIM = DEFAULT_HEAD_DIM;
constexpr int FMA_BLOCK_N = 32;
constexpr int FMA_BLOCK_M = 64;

static size_t s_matrix_bytes(int BH, int seq_len) {
    return (size_t)BH * seq_len * seq_len * sizeof(float);
}

static bool run_correctness_case(int BH, int seq_len, bool causal) {
    size_t qkv_elems = (size_t)BH * seq_len * HEAD_DIM;
    std::vector<float> hQ(qkv_elems), hK(qkv_elems), hV(qkv_elems);
    fill_random(hQ, 1); fill_random(hK, 2); fill_random(hV, 3);
    snap_to_half(hQ); snap_to_half(hK); snap_to_half(hV);

    std::vector<float> refO(qkv_elems);
    ref_forward_cached(hQ, hK, hV, refO, BH, seq_len, HEAD_DIM, causal);

    std::vector<__half> hQh = to_half(hQ), hKh = to_half(hK), hVh = to_half(hV);
    DeviceAllocTracker alloc;
    size_t qkv_bytes = qkv_elems * sizeof(__half);
    size_t s_bytes = s_matrix_bytes(BH, seq_len);
    __half *Q = (__half*)alloc.alloc(qkv_bytes), *K = (__half*)alloc.alloc(qkv_bytes);
    __half *V = (__half*)alloc.alloc(qkv_bytes), *O_naive = (__half*)alloc.alloc(qkv_bytes);
    __half *O_fma = (__half*)alloc.alloc(qkv_bytes);
    float *S = (float*)alloc.alloc(s_bytes);
    CUDA_CHECK(cudaMemcpy(Q, hQh.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(K, hKh.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V, hVh.data(), qkv_bytes, cudaMemcpyHostToDevice));

    attention_naive_launch<HEAD_DIM>(Q, K, V, O_naive, S, BH, seq_len, causal);
    int block_m = min(FMA_BLOCK_M, seq_len);
    flash_tiled_launch<HEAD_DIM, FMA_BLOCK_N>(Q, K, V, O_fma, BH, seq_len, causal, block_m);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> gOh(qkv_elems), gOfmaH(qkv_elems);
    CUDA_CHECK(cudaMemcpy(gOh.data(), O_naive, qkv_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gOfmaH.data(), O_fma, qkv_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> gO = to_float(gOh), gOfma = to_float(gOfmaH);

    char label[128];
    snprintf(label, sizeof(label), "BH=%d seq_len=%d d=%d causal=%d (naive vs CPU ref)", BH, seq_len, HEAD_DIM, causal);
    bool ok = verify_against_cpu(gO, refO, 3e-2f, label);
    ok &= verify_against_cpu(gO, gOfma, 3e-2f, "naive vs FMA-tiled kernel");

    alloc.free(Q, qkv_bytes); alloc.free(K, qkv_bytes); alloc.free(V, qkv_bytes);
    alloc.free(O_naive, qkv_bytes); alloc.free(O_fma, qkv_bytes); alloc.free(S, s_bytes);
    return ok;
}

int main(int argc, char** argv) {
    bool causal = argc > 1 && std::string(argv[1]) == "causal";

    printf("=== [00] standard attention (no fusion / no tiling / no online softmax) ===\n");
    bool all_ok = true;
    all_ok &= run_correctness_case(2, 128, false);
    all_ok &= run_correctness_case(2, 128, true);
    all_ok &= run_correctness_case(2, 130, false);
    printf("%s\n\n", all_ok ? "All correctness cases OK" : "SOME CASES FAILED");

    printf("=== [00] benchmark vs. tiled-FMA baseline (BH=%d, head_dim=%d, causal=%d) ===\n",
           DEFAULT_BH, HEAD_DIM, causal);
    printf("%8s | %10s %10s %8s %10s %10s\n", "seq_len", "naive_ms", "fma_ms", "speedup", "gflops", "S_MiB");

    std::string csv_path = causal ? "results/results_00_attention_naive_causal.csv"
                                  : "results/results_00_attention_naive_noncausal.csv";
    std::ofstream csv(csv_path);
    csv << "seq_len,time_ms,fma_ms,speedup_vs_fma,gflops,s_matrix_mib\n";

    for (int seq_len : default_seq_lens()) {
        size_t qkv_elems = (size_t)DEFAULT_BH * seq_len * HEAD_DIM;
        size_t qkv_bytes = qkv_elems * sizeof(__half);
        size_t s_bytes = s_matrix_bytes(DEFAULT_BH, seq_len);
        double s_mib = s_bytes / (1024.0 * 1024.0);

        // S가 O(N^2)이라 seq_len이 조금만 커져도 GPU에 안 들어간다 -- 바로
        // 이게 FlashAttention이 없앤 제약이라, 못 돌리면 -1로 기록만 남긴다.
        if (!fits_in_gpu(4 * qkv_bytes + s_bytes)) {
            printf("%8d | %10s %10s %8s %10s %10.1f  (S가 GPU 메모리에 안 들어감 -- 스킵)\n",
                   seq_len, "-", "-", "-", "-", s_mib);
            csv << seq_len << ",-1,-1,-1,-1," << s_mib << "\n";
            continue;
        }

        std::vector<float> hQ(qkv_elems), hK(qkv_elems), hV(qkv_elems);
        fill_random(hQ, 1); fill_random(hK, 2); fill_random(hV, 3);
        std::vector<__half> hQh = to_half(hQ), hKh = to_half(hK), hVh = to_half(hV);

        DeviceAllocTracker alloc;
        __half *Q = (__half*)alloc.alloc(qkv_bytes), *K = (__half*)alloc.alloc(qkv_bytes);
        __half *V = (__half*)alloc.alloc(qkv_bytes), *O = (__half*)alloc.alloc(qkv_bytes);
        float *S = (float*)alloc.alloc(s_bytes);
        CUDA_CHECK(cudaMemcpy(Q, hQh.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(K, hKh.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(V, hVh.data(), qkv_bytes, cudaMemcpyHostToDevice));

        auto run_naive = [&]() { attention_naive_launch<HEAD_DIM>(Q, K, V, O, S, DEFAULT_BH, seq_len, causal); };
        float naive_ms = time_ms_avg(run_naive, WARM_UP, iters_for(seq_len));

        auto run_fma = [&]() { flash_tiled_launch<HEAD_DIM, FMA_BLOCK_N>(Q, K, V, O, DEFAULT_BH, seq_len, causal, FMA_BLOCK_M); };
        float fma_ms = time_ms_avg(run_fma, WARM_UP, iters_for(seq_len));

        double gflops = attn_gflops(DEFAULT_BH, seq_len, HEAD_DIM, causal, naive_ms);
        double speedup = fma_ms / naive_ms;

        printf("%8d | %10.4f %10.4f %7.2fx %10.2f %10.1f\n", seq_len, naive_ms, fma_ms, speedup, gflops, s_mib);
        printf("           GPU 메모리 peak %.1f MiB 중 S가 %.1f MiB -- Q/K/V/O는 전부 합쳐 %.1f MiB\n",
               alloc.peak_mb(), s_mib, 4 * qkv_bytes / (1024.0 * 1024.0));
        csv << seq_len << "," << naive_ms << "," << fma_ms << "," << speedup << "," << gflops << "," << s_mib << "\n";

        alloc.free(Q, qkv_bytes); alloc.free(K, qkv_bytes); alloc.free(V, qkv_bytes);
        alloc.free(O, qkv_bytes); alloc.free(S, s_bytes);
    }

    printf("\nSaved %s\n", csv_path.c_str());
    return all_ok ? 0 : 1;
}
