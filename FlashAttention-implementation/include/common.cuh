// [공용 유틸리티] 이 디렉토리의 모든 .cu 파일이 공통으로 쓰는 도구 모음.
// - 에러 체크 매크로
// - GPU 시간 측정용 타이머
// - 몇 바이트 할당했는지 직접 추적하는 할당자 (naive의 O(seq_len^2) vs
//   flash의 O(seq_len) 메모리 사용량을 정확한 숫자로 비교하기 위함)
// - double precision(fp64) CPU attention 구현 -- 이 디렉토리의 모든
//   정확성 테스트(test_naive.cu, test_flash.cu)가 "정답"으로 삼는 기준.
// - fp16(half) <-> fp32(float) 변환 헬퍼 -- 논문/Triton 버전과 정밀도를
//   맞추기 위해 Q,K,V,O,dQ,dK,dV,dO는 이제 __half(fp16)로 저장한다
//   (논문 Appendix E.1/E.2: "We train with FP16 precision using Apex AMP").
//   다만 softmax·누적 계산 자체는 정밀도 손실을 막기 위해 fp32로 계산하고,
//   HBM에 저장/전송하는 시점에만 fp16으로 변환한다 -- PyTorch AMP나 저희
//   Triton 버전(tl.dot의 fp32 accumulator)과 동일한 "저장은 fp16, 계산은
//   fp32" 방식.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

// CUDA API 호출 결과를 매번 손으로 if문 쓰기 귀찮으니 매크로로 감싼 것.
// 실패하면 에러 메시지 찍고 바로 종료(exit)
// 사용 예: CUDA_CHECK(cudaMalloc(&ptr, bytes));
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err__));                    \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// [메모리 사용량 추적 할당자]
// cudaMemGetInfo()로 "지금 GPU에 남은 전체 메모리"를 물어보는 방식은
// CUDA 컨텍스트 자체의 오버헤드나 다른 프로세스의 사용량까지 섞여서
// 나오기 때문에 잡음(noise)이 많다. 대신 우리가 cudaMalloc/cudaFree를
// 부를 때마다 직접 바이트 수를 더하고 빼면서 "이 프로그램이 실제로
// 논리적으로 얼마나 할당했는가"를 정확히 추적한다. current는 "지금
// 들고 있는 총량", peak는 "지금까지 본 적 있는 최댓값".
// ---------------------------------------------------------------------------
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
    // "지금부터 측정 시작"이라는 의미로 peak를 현재값으로 리셋.
    // 예: reset_peak() 직후에 커널을 돌리고 peak_mb()를 부르면,
    // "그 커널을 도는 동안 새로 늘어난 메모리의 최댓값 + 이미 갖고
    // 있던 것"을 알 수 있다 (benchmark.cu에서 이 패턴을 자주 씀).
    void reset_peak() { peak = current; }
    double peak_mb() const { return peak / (1024.0 * 1024.0); }
};

// ---------------------------------------------------------------------------
// [GPU 타이머] CUDA event 기반. CPU 쪽 시계(예: std::chrono)로 재면
// "커널을 GPU에 던지는" 시점만 재게 되어 실제 GPU 연산 시간과 다르다
// (CUDA 커널 launch는 비동기이기 때문). cudaEvent는 GPU 타임라인
// 위에 타임스탬프를 찍는 방식이라 실제 연산 시간을 정확히 잰다.
// ---------------------------------------------------------------------------
struct GpuTimer {
    cudaEvent_t start_ev, stop_ev;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_ev));
        CUDA_CHECK(cudaEventCreate(&stop_ev));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_ev)); }
    // stop_ev를 기록하고, GPU가 거기까지 실제로 도달할 때까지
    // cudaEventSynchronize로 기다린 다음, start~stop 사이 경과 시간(ms)을 반환.
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_ev));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
        return ms;
    }
};

// fn()을 `warmup`번 먼저 (측정 없이) 실행해서 최초 실행 시 발생하는
// 일회성 비용(CUDA 컨텍스트 초기화 등)을 미리 소진시킨 뒤, `iters`번을
// 실제로 측정해서 그 중앙값(median)을 반환한다. 평균 대신 중앙값을
// 쓰는 이유는 가끔 튀는 값(다른 프로세스의 간섭 등) 하나가 평균을
// 심하게 왜곡시키는 걸 막기 위함.
template <typename Fn>
float time_ms_median(Fn fn, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> times(iters);
    GpuTimer timer;
    for (int i = 0; i < iters; i++) {
        timer.start();
        fn();
        times[i] = timer.stop_ms();
    }
    std::sort(times.begin(), times.end());
    return times[iters / 2];
}

// ---------------------------------------------------------------------------
// [랜덤 초기화 / 비교용 헬퍼]
// ---------------------------------------------------------------------------
// seed를 다르게 주면 Q,K,V 등이 서로 다른 값으로 채워진다 (실제 attention
// 내용은 안 중요하고, 그냥 "제대로 계산이 되는지"만 검증하면 되므로
// 표준정규분포 난수로 충분).
inline void fill_random(std::vector<float>& v, unsigned seed, float scale = 1.0f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : v) x = dist(gen) * scale;
}

// GPU 결과와 CPU 정답 사이의 "가장 크게 틀린 지점"의 절댓값 오차.
// 이 값이 작으면(예: 1e-3 미만) "같은 계산을 한다"고 판단.
inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// ---------------------------------------------------------------------------
// [fp16 <-> fp32 변환 헬퍼]
// ---------------------------------------------------------------------------
// fp32 벡터를 fp16으로 변환. GPU에 업로드할 입력(Q,K,V,dO)을 준비할 때 사용.
inline std::vector<__half> to_half(const std::vector<float>& v) {
    std::vector<__half> out(v.size());
    for (size_t i = 0; i < v.size(); i++) out[i] = __float2half(v[i]);
    return out;
}

// fp16 벡터를 fp32로 변환. GPU 결과(O,dQ,dK,dV)를 CPU 정답과 비교하기 전에 사용.
inline std::vector<float> to_float(const std::vector<__half>& v) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); i++) out[i] = __half2float(v[i]);
    return out;
}

// fp32 값을 "fp16으로 반올림했을 때의 값"으로 스냅한다 (fp32 상태 그대로 유지).
// CPU 레퍼런스가 "GPU가 실제로 보는 것과 동일한 입력"을 계산하도록, 랜덤
// 입력을 만든 직후 이 함수로 한 번 스냅해두면, 이후 GPU(fp16 저장)와
// CPU(fp64 계산) 결과를 비교할 때 "입력값 자체가 미묘하게 다른" 잡음 없이
// 순수하게 "계산 방식의 차이"만 비교할 수 있다.
inline void snap_to_half(std::vector<float>& v) {
    for (auto& x : v) x = __half2float(__float2half(x));
}

// ---------------------------------------------------------------------------
// [CPU 레퍼런스 attention, double precision]
// GPU 커널이 정확한지 검증할 "정답"으로 쓰는 구현. GPU는 fp32로 계산해서
// 어쩔 수 없이 반올림 오차가 생기는데, CPU 쪽을 fp64(double)로 짜두면
// "GPU의 fp32 반올림 오차"와 "알고리즘 자체가 틀린 버그"를 구분할 수
// 있다 (버그가 있으면 오차가 1e-3 수준이 아니라 훨씬 크게 튄다).
// 성능은 전혀 신경 안 쓴 3중 for문 구현 -- 오직 정답 확인용.
//
// 레이아웃: Q,K,V,O,dO,dQ,dK,dV 모두 (batch*heads, seq_len, d) 순서의
// row-major 1차원 배열로 표현한다 (GPU 쪽 커널들과 동일한 메모리 레이아웃).
// ---------------------------------------------------------------------------

// Forward: O = softmax(QK^T / sqrt(d)) V  (causal=true면 미래 위치는 마스킹)
//BH: batch × attention heads
//N = seq_len: 토큰 수
//d: head dimension
//Q, K, V: 논리적 shape [BH, seq_len, head_dim]
//S = QKᵀ / √d: attention score
//P = softmax(S): attention 확률
//O = PV: attention 출력
//dO: 출력 O로부터 전달받은 gradient
inline void ref_forward(const std::vector<float>& Q, const std::vector<float>& K,
                         const std::vector<float>& V, std::vector<float>& O,
                         int BH, int seq_len, int d, bool causal) {
    double scale = 1.0 / std::sqrt((double)d);              // 1/(d)^0.5
    std::vector<double> row(seq_len);                       // q의 1행의 score 저장
    for (int bh = 0; bh < BH; bh++) {
        const float* Qb = &Q[(size_t)bh * seq_len * d];
        const float* Kb = &K[(size_t)bh * seq_len * d];
        const float* Vb = &V[(size_t)bh * seq_len * d];
        float* Ob = &O[(size_t)bh * seq_len * d];
        for (int i = 0; i < seq_len; i++) {                 // 쿼리(질의) 위치 i
            int jmax = causal ? (i + 1) : seq_len;          // causal이면 j<=i까지만
            // 1단계: 이 행(row)의 점수 S_ij = scale * Q_i . K_j 를 전부 계산하면서 행 최댓값 m도 같이 구함
            double m = -1e300;                              // score 행의  최댓값을 저장하는 변수
            for (int j = 0; j < jmax; j++) {
                double s = 0.0;
                for (int t = 0; t < d; t++) {
                    s += (double)Qb[i * d + t] * Kb[j * d + t]; //Q_i * K_j 계산
                }
                s *= scale;
                row[j] = s;
                if (s > m) m = s;
            }
            // 2단계: 수치 안정적인 softmax -- exp(s - m)로 오버플로우 방지
            double l = 0.0; // 지수 계산과 분모 합산
            for (int j = 0; j < jmax; j++) {
                row[j] = std::exp(row[j] - m);
                l += row[j];
            }
            // 3단계: O_i = sum_j softmax(S)_ij * V_j
            for (int t = 0; t < d; t++) {
                double acc = 0.0;
                for (int j = 0; j < jmax; j++) acc += row[j] * Vb[j * d + t];
                Ob[i * d + t] = (float)(acc / l);
            }
        }
    }
}

// Backward: forward의 O=softmax(QK^T)V에 대한 dQ, dK, dV를 직접 미분해서 계산.
// (FlashAttention 논문 Appendix B.2의 수식을 그대로 3중 for문으로 옮긴 것)
//   dV_j = sum_i P_ij * dO_i
//   dP_ij = dO_i . V_j
//   D_i = sum_j P_ij * dP_ij            (= dO_i . O_i 와 같은 값)
//   dS_ij = P_ij * (dP_ij - D_i)
//   dQ_i = sum_j dS_ij * scale * K_j
//   dK_j = sum_i dS_ij * scale * Q_i

//dO -> dV,dP -> dS -> dQ,dK
inline void ref_backward(const std::vector<float>& Q, const std::vector<float>& K,
                          const std::vector<float>& V, const std::vector<float>& dO,
                          std::vector<float>& dQ, std::vector<float>& dK, std::vector<float>& dV,
                          int BH, int seq_len, int d, bool causal) {
    double scale = 1.0 / std::sqrt((double)d);
    std::vector<double> P(seq_len);
    std::fill(dQ.begin(), dQ.end(), 0.0f);
    std::fill(dK.begin(), dK.end(), 0.0f);
    std::fill(dV.begin(), dV.end(), 0.0f);
    for (int bh = 0; bh < BH; bh++) {
        const float* Qb = &Q[(size_t)bh * seq_len * d];
        const float* Kb = &K[(size_t)bh * seq_len * d];
        const float* Vb = &V[(size_t)bh * seq_len * d];
        const float* dOb = &dO[(size_t)bh * seq_len * d];
        float* dQb = &dQ[(size_t)bh * seq_len * d];
        float* dKb = &dK[(size_t)bh * seq_len * d];
        float* dVb = &dV[(size_t)bh * seq_len * d];
        for (int i = 0; i < seq_len; i++) {
            int jmax = causal ? (i + 1) : seq_len;
            // forward와 똑같이 이 행의 softmax P_i를 다시 계산 (recompute).
            // GPU flash 버전은 여기서 저장해둔 logsumexp L을 재활용하지만,
            // CPU 레퍼런스는 어차피 속도가 중요하지 않으므로 그냥 다시 계산.
            double m = -1e300;
            for (int j = 0; j < jmax; j++) {
                double s = 0.0;
                for (int t = 0; t < d; t++) s += (double)Qb[i * d + t] * Kb[j * d + t];
                s *= scale;
                P[j] = s;
                if (s > m) m = s;
            }
            double l = 0.0;
            for (int j = 0; j < jmax; j++) { P[j] = std::exp(P[j] - m); l += P[j]; }
            for (int j = 0; j < jmax; j++) P[j] /= l;

            // dV_j += P_ij * dO_i ; 
            // dP_ij = dO_i . V_j ; 
            // D_i = sum_j P_ij*dP_ij
            std::vector<double> dP(jmax);
            double D = 0.0;
            for (int j = 0; j < jmax; j++) {
                double dp = 0.0;
                for (int t = 0; t < d; t++) dp += (double)dOb[i * d + t] * Vb[j * d + t];
                dP[j] = dp;
                D += P[j] * dp;
                for (int t = 0; t < d; t++) dVb[j * d + t] += (float)(P[j] * dOb[i * d + t]);
            }
            for (int j = 0; j < jmax; j++) {
                double dS = P[j] * (dP[j] - D) * scale;
                for (int t = 0; t < d; t++) {
                    dQb[i * d + t] += (float)(dS * Kb[j * d + t]);
                    dKb[j * d + t] += (float)(dS * Qb[i * d + t]);
                }
            }
        }
    }
}
