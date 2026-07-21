// [FlashAttention] forward(Algorithm 2) + backward(Algorithm 4)를 손으로 짠
// fused CUDA 커널로 구현. naive_kernels.cuh와 제일 다른 점: seq_len x seq_len 크기
// 행렬을 단 한 번도 global memory(HBM)에 쓰지 않는다. flash_attention.py의
// Triton 커널과 완전히 같은 알고리즘을 raw CUDA로 옮긴 것 -- Triton의
// tl.load/tl.dot 대신 우리가 직접 __shared__ 타일을 관리하고, 텐서 코어도
// 안 쓰고 그냥 스칼라 FMA(곱셈+누적) 반복문으로 계산한다. 그래서 Triton
// 버전보다는 느리지만, "타일링 + 온라인 소프트맥스 + recomputation"이라는
// 알고리즘 자체는 순수 CUDA C++만으로도 똑같이 구현 가능하다는 걸 보여준다.
//
// [스레드 배치 큰 그림] CUDA 스레드 1개 = Q의 한 행(forward, backward의 dQ
// 계산) 또는 K/V의 한 행(backward의 dK/dV 계산)을 담당한다. HEAD_DIM과
// K/V 타일 폭(BLOCK_N)은 "레지스터에 담을 배열의 크기"로 쓰이기 때문에
// 반드시 컴파일 타임에 정해져 있어야 해서 C++ 템플릿 매개변수로 뺐다.
// Q 타일 폭(BLOCK_M)은 그런 제약이 없어서 그냥 blockDim.x로 실행 시점에
// 정한다.
//
// [정밀도] Q,K,V,O,dQ,dK,dV,dO는 논문과 동일하게 __half(fp16)로 저장한다.
// shared memory 타일(Qs,Ks,Vs), 온라인 소프트맥스 상태(m_i,l_i,acc), 그리고
// logsumexp(L)/D 통계량은 전부 fp32로 유지한다 -- naive_kernels.cuh와 같은
// 이유(softmax·누적 연산의 정밀도 보존)이고, 실제로 Triton 버전의
// tl.dot(fp16 입력, fp32 accumulator)이나 PyTorch AMP와 동일한 "저장은
// fp16, 계산은 fp32" 방식이다. 그래서 global memory(HBM)에서 값을 읽어올
// 때 __half2float(), 다시 HBM에 쓸 때 __float2half()가 붙는다.
#pragma once // 헤더 컴파일 한 번 할 때 딱 1번만 읽어라(중복 include 방지)의 의미
#include "common.cuh"

// ---------------------------------------------------------------------------
// Forward (논문 Algorithm 2)
// ---------------------------------------------------------------------------
// [템플릿 매개변수]
//   HEAD_DIM  : head 차원 (예: 64) -- acc[HEAD_DIM] 레지스터 배열 크기로 쓰임
//   BLOCK_N   : K/V 타일 폭 (예: 32) -- scores[BLOCK_N] 레지스터 배열 크기로 쓰임
// [실행 시점 배치]
//   grid = (Q 타일 개수, batch*heads),  block = BLOCK_M개의 스레드
//   즉 "블록 하나 = Q 타일 하나 담당", "그 블록 안 스레드 하나 = Q의 행 하나 담당"
template <int HEAD_DIM, int BLOCK_N>
__global__ void flash_fwd_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                  const __half* __restrict__ V, __half* __restrict__ O,
                                  float* __restrict__ L, int seq_len, float scale, bool causal) {
    // dynamic shared memory 한 덩어리를 세 부분으로 나눠서 씀 (Q타일, K타일, V타일).
    // 크기가 head_dim=128일 때 48KB 정적 __shared__ 한도를 넘길 수 있어서
    // extern __shared__ + launch 시점에 크기 지정하는 "동적" 방식을 쓴다.
    // Q,K,V는 fp16으로 저장돼있지만 타일은 fp32로 풀어서 담는다 (계산 정밀도 보존).
    extern __shared__ float smem[];
    const int BLOCK_M = blockDim.x;
    float* Qs = smem;                        // [BLOCK_M][HEAD_DIM]  -- 이 블록이 담당하는 Q 타일
    float* Ks = Qs + BLOCK_M * HEAD_DIM;      // [BLOCK_N][HEAD_DIM]  -- 현재 순회 중인 K 타일
    float* Vs = Ks + BLOCK_N * HEAD_DIM;      // [BLOCK_N][HEAD_DIM]  -- 현재 순회 중인 V 타일

    const int bh = blockIdx.y;      // 이 블록이 담당하는 (batch, head) 인덱스
    const int row = threadIdx.x;    // 이 스레드가 담당하는, Q 타일 안에서의 상대 행 번호
    const int q_idx = blockIdx.x * BLOCK_M + row;  // 전체 시퀀스 기준 절대 행 번호 (=쿼리 위치)

    const __half* Qbh = Q + (size_t)bh * seq_len * HEAD_DIM;
    const __half* Kbh = K + (size_t)bh * seq_len * HEAD_DIM;
    const __half* Vbh = V + (size_t)bh * seq_len * HEAD_DIM;

    // 각 스레드가 자기 담당 Q 행을 global memory에서 읽어와(fp16->fp32 변환)
    // shared memory에 저장 (논문 Algorithm 2, 9번째 줄 "Load Q_i ... to
    // on-chip SRAM"에 해당). q_idx가 seq_len을 넘어가면(=seq_len이 BLOCK_M으로
    // 안 나눠떨어지는 경계) 0으로 채워서 안전하게 처리.
    for (int t = 0; t < HEAD_DIM; t++)
        Qs[row * HEAD_DIM + t] = (q_idx < seq_len) ? __half2float(Qbh[(size_t)q_idx * HEAD_DIM + t]) : 0.0f;

    // 온라인 소프트맥스의 상태값들. 전부 "이 스레드 하나"(=이 쿼리 행 하나) 전용이라
    // 레지스터에 그대로 둔다. acc는 아직 정규화 안 된(un-normalized) 출력 누적값.
    float m_i = -INFINITY, l_i = 0.0f;
    float acc[HEAD_DIM];
#pragma unroll
    for (int t = 0; t < HEAD_DIM; t++) acc[t] = 0.0f;

    // causal이면 이 Q 타일의 대각선보다 뒤에 있는 K/V 타일은 계산할 필요가
    // 아예 없으므로(전부 마스킹되어 -inf) 루프 상한(hi)을 줄여서 건너뛴다.
    // -> causal 모드에서 flash가 non-causal보다 훨씬 빠른 이유 중 하나.
    const int hi = causal ? min(seq_len, blockIdx.x * BLOCK_M + BLOCK_M) : seq_len;
    __syncthreads();  // 모든 스레드가 Qs 쓰기를 마칠 때까지 대기 (아래에서 다른 스레드가 쓴 값도 읽으므로 필수)

    // ===== K/V 타일을 순회하며 온라인 소프트맥스 갱신 (Algorithm 2의 핵심 루프) =====
    for (int kv0 = 0; kv0 < hi; kv0 += BLOCK_N) {
        // 이 블록의 모든 스레드가 협력해서 K,V 타일을 shared memory로 복사(+fp16->fp32 변환).
        // (스레드 수 BLOCK_M과 타일 원소 수 BLOCK_N*HEAD_DIM이 다를 수 있어서
        // "각 스레드가 여러 개씩, stride만큼 건너뛰며" 담당하는 방식)
        for (int idx = threadIdx.x; idx < BLOCK_N * HEAD_DIM; idx += BLOCK_M) {
            int r = idx / HEAD_DIM, c = idx % HEAD_DIM;
            int kv_idx = kv0 + r;
            Ks[idx] = (kv_idx < seq_len) ? __half2float(Kbh[(size_t)kv_idx * HEAD_DIM + c]) : 0.0f;
            Vs[idx] = (kv_idx < seq_len) ? __half2float(Vbh[(size_t)kv_idx * HEAD_DIM + c]) : 0.0f;
        }
        __syncthreads();  // K,V 타일 복사가 끝날 때까지 대기 -- 안 하면 다른 스레드가 덜 채워진 값을 읽을 수 있음

        if (q_idx < seq_len) {
            // S_ij = scale * Q_i . K_j 를 이 타일의 BLOCK_N개 열에 대해 전부 계산.
            // scores[]는 레지스터 배열 (BLOCK_N이 컴파일타임 상수라서 가능).
            float scores[BLOCK_N];
            float blk_max = -INFINITY;
#pragma unroll
            for (int c = 0; c < BLOCK_N; c++) {
                int kv_idx = kv0 + c;
                float s = -INFINITY;
                if (kv_idx < seq_len && !(causal && kv_idx > q_idx)) {  // 범위 밖이거나 causal 마스킹 대상이면 -inf 유지
                    float dot = 0.0f;
#pragma unroll
                    for (int t = 0; t < HEAD_DIM; t++) dot += Qs[row * HEAD_DIM + t] * Ks[c * HEAD_DIM + t];
                    s = dot * scale;
                }
                scores[c] = s;
                blk_max = fmaxf(blk_max, s);
            }
            // ----- 온라인 소프트맥스 갱신 (논문 3.1절 수식, Algorithm 2의 12~13번째 줄) -----
            // 새 타일까지 합친 새 최댓값(m_new)을 구하고, 기존 누적값(l_i, acc)을
            // "최댓값이 바뀐 만큼" 보정(alpha)한 뒤 새 타일의 기여분을 더한다.
            float m_new = fmaxf(m_i, blk_max);
            float alpha = expf(m_i - m_new);   // 이전 최댓값 기준 값들을 새 최댓값 기준으로 재조정하는 배율
            l_i *= alpha;
#pragma unroll
            for (int t = 0; t < HEAD_DIM; t++) acc[t] *= alpha;
#pragma unroll
            for (int c = 0; c < BLOCK_N; c++) {
                float p = expf(scores[c] - m_new);  // 이 타일에서의 (정규화 전) softmax 값
                l_i += p;
#pragma unroll
                for (int t = 0; t < HEAD_DIM; t++) acc[t] += p * Vs[c * HEAD_DIM + t];  // O_i += p * V_j 누적
            }
            m_i = m_new;
        }
        __syncthreads();  // 다음 반복에서 Ks/Vs를 새 타일로 덮어쓰기 전에, 이번 타일 계산이 다 끝났는지 확인
    }

    // 루프가 다 끝난 뒤 마지막으로 나눗셈(정규화)을 한 번만 수행 -- l_i로 나누는 시점을
    // 매 타일마다가 아니라 맨 마지막에 딱 한 번만 하는 것이 온라인 소프트맥스의 핵심 아이디어.
    // fp32 acc를 fp16으로 변환해서 저장 (O의 저장 형식이 fp16이므로).
    if (q_idx < seq_len) {
        for (int t = 0; t < HEAD_DIM; t++) O[((size_t)bh * seq_len + q_idx) * HEAD_DIM + t] = __float2half(acc[t] / l_i);
        // L = logsumexp = m + log(l). backward에서 소프트맥스를 다시 계산할 때
        // "온라인 최댓값 추적"을 처음부터 다시 안 해도 되도록 이 값 하나만 저장해둔다.
        // (L 자체는 fp32 그대로 -- 통계량이라 fp16으로 내릴 이유가 없음)
        L[(size_t)bh * seq_len + q_idx] = m_i + logf(l_i);
    }
}

// [호스트 측 launcher] 커널을 실제로 실행시키는 함수.
template <int HEAD_DIM, int BLOCK_N>
void flash_forward_launch(const __half* Q, const __half* K, const __half* V, __half* O, float* L,
                           int BH, int seq_len, bool causal, int BLOCK_M) {
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    // Qs + Ks + Vs 세 타일을 합친 만큼의 shared memory가 필요 (fp32로 담으므로 fp16 원본의 2배 크기).
    size_t smem_bytes = (size_t)(BLOCK_M + 2 * BLOCK_N) * HEAD_DIM * sizeof(float);
    auto kernel = flash_fwd_kernel<HEAD_DIM, BLOCK_N>;
    // 정적 __shared__는 48KB까지만 허용되므로, 그보다 큰 동적 shared memory를 쓰려면
    // 이 함수 호출로 "이 커널은 이만큼까지 써도 된다"고 미리 opt-in 해줘야 한다
    // (head_dim=128 같은 큰 설정에서 필요해짐).
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    dim3 grid((seq_len + BLOCK_M - 1) / BLOCK_M, BH);  // (Q 타일 개수, batch*heads)
    kernel<<<grid, BLOCK_M, smem_bytes>>>(Q, K, V, O, L, seq_len, scale, causal);
}

// ---------------------------------------------------------------------------
// [Algorithm 1/2를 "글자 그대로" 재현한 버전] 위의 flash_fwd_kernel과 딱 한
// 군데만 다르다: O_i를 정규화(l_i로 나눔)해서 HBM에 쓰는 시점이 "루프가 다
// 끝난 뒤 딱 한 번"이 아니라 "매 K/V 타일마다 한 번씩"이다 -- 논문 Algorithm
// 1의 12번째 줄("Write O_i ← diag(ℓ_new)^-1 (...) to HBM"), Algorithm 2의
// 15번째 줄이 문자 그대로 요구하는 방식. 이건 FlashAttention-2 논문이 자기
// 기여로 내세우는 "정규화를 맨 끝까지 미뤄서 non-matmul 연산과 HBM 쓰기를
// 줄인다"는 최적화를 빼버린 버전이라고 보면 된다.
//
// 최종 결과값은 flash_fwd_kernel과 완전히 동일하다 (마지막 반복에서 쓰는
// 값이 결국 진짜 최종값이고, 그 전 반복들의 쓰기는 그저 나중에 덮어써질
// "낭비된 중간 결과"일 뿐이므로). 차이는 오직 "타일 개수(T_c)만큼 나눗셈과
// HBM 쓰기를 더 한다"는 실행 비용에만 있다 -- 그래서 benchmark_normalization.cu에서
// 이 커널과 flash_fwd_kernel의 실행 시간만 비교하면 "정규화를 미루는 것"
// 하나의 효과를 순수하게 분리해서 측정할 수 있다.
template <int HEAD_DIM, int BLOCK_N>
__global__ void flash_fwd_kernel_strict(const __half* __restrict__ Q, const __half* __restrict__ K,
                                         const __half* __restrict__ V, __half* __restrict__ O,
                                         float* __restrict__ L, int seq_len, float scale, bool causal) {
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

            // ===== 여기가 flash_fwd_kernel과의 유일한 차이 =====
            // Algorithm 1/2가 문자 그대로 요구하는 대로, "이번 타일까지 반영된
            // 정규화 완료 O_i"를 매 타일마다 HBM에 쓴다 (+ ℓ, m도 마찬가지로
            // 매번 씀 -- Algorithm 1의 13번째 줄). 다음 반복에서 또 덮어써질
            // 걸 알면서도 매번 쓰는 것이 핵심 -- 이게 FA-2가 없앤 "낭비".
            for (int t = 0; t < HEAD_DIM; t++)
                O[((size_t)bh * seq_len + q_idx) * HEAD_DIM + t] = __float2half(acc[t] / l_i);
            L[(size_t)bh * seq_len + q_idx] = m_i + logf(l_i);
        }
        __syncthreads();
    }
    // 루프가 끝나면 이미 마지막 반복에서 정답이 쓰인 상태 -- flash_fwd_kernel처럼
    // "루프 끝난 뒤 한 번 더 쓰는" 코드가 필요 없다 (있으면 오히려 중복).
}

template <int HEAD_DIM, int BLOCK_N>
void flash_forward_launch_strict(const __half* Q, const __half* K, const __half* V, __half* O, float* L,
                                  int BH, int seq_len, bool causal, int BLOCK_M) {
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    size_t smem_bytes = (size_t)(BLOCK_M + 2 * BLOCK_N) * HEAD_DIM * sizeof(float);
    auto kernel = flash_fwd_kernel_strict<HEAD_DIM, BLOCK_N>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    dim3 grid((seq_len + BLOCK_M - 1) / BLOCK_M, BH);
    kernel<<<grid, BLOCK_M, smem_bytes>>>(Q, K, V, O, L, seq_len, scale, causal);
}

// ---------------------------------------------------------------------------
// Backward (논문 Algorithm 4). Recomputation(재계산) 방식 -- forward 때
// 저장해둔 O와 L(logsumexp)만 가지고 S,P를 다시 계산해서 그래디언트를 구한다.
// Triton 버전(flash_attention.py)과 같은 이유로 커널을 2개로 나눴다: 이렇게
// 나누면 각 출력 텐서(dK,dV 또는 dQ)를 "정확히 스레드 하나가만" 쓰게 되어서
// 여러 스레드가 동시에 같은 곳에 더하는 atomic 연산이 전혀 필요 없어진다.
// ---------------------------------------------------------------------------

// D_i = rowsum(dO_i * O_i). backward 공식에 필요한 작은 통계량 하나
// (자세한 유도는 common.cuh의 ref_backward 주석 참고). 크기가 O(seq_len)이라
// (seq_len x seq_len이 아니라) 아주 가볍다 -- 스레드 하나가 행 하나씩 담당.
// O, DO는 fp16 저장이라 읽을 때 fp32로 변환해서 누적 (D 자체는 fp32).
__global__ void flash_bwd_d_kernel(const __half* __restrict__ O, const __half* __restrict__ DO,
                                    float* __restrict__ D, int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;
    int bh = blockIdx.y;
    const __half* Oi = O + ((size_t)bh * seq_len + idx) * head_dim;
    const __half* DOi = DO + ((size_t)bh * seq_len + idx) * head_dim;
    float acc = 0.0f;
    for (int t = 0; t < head_dim; t++) acc += __half2float(Oi[t]) * __half2float(DOi[t]);
    D[(size_t)bh * seq_len + idx] = acc;
}

// [dK, dV 계산] "블록 하나 = K/V 타일 하나" 배치. 즉 forward와 반대로,
// 이번엔 Q 타일이 아니라 K/V 타일을 기준으로 병렬화하고, 그 안에서
// 모든 Q 타일을 순회하며 dK_j, dV_j를 전부 누적한 다음 딱 한 번만 쓴다
// (flash_attention.py의 _bwd_dkdv_kernel과 동일한 구조).
template <int HEAD_DIM>
__global__ void flash_bwd_dkdv_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                       const __half* __restrict__ V, const __half* __restrict__ DO,
                                       const float* __restrict__ L, const float* __restrict__ D,
                                       __half* __restrict__ DK, __half* __restrict__ DV,
                                       int seq_len, float scale, bool causal) {
    extern __shared__ float smem[];
    const int BLOCK_M = blockDim.x;  // 여기서는 이 값이 "Q 타일 폭"으로 쓰인다 (K/V 타일 폭이 아님에 주의)
    float* Qs = smem;                    // [BLOCK_M][HEAD_DIM]
    float* DOs = Qs + BLOCK_M * HEAD_DIM; // [BLOCK_M][HEAD_DIM]
    float* Ls = DOs + BLOCK_M * HEAD_DIM; // [BLOCK_M]
    float* Ds = Ls + BLOCK_M;             // [BLOCK_M]

    const int bh = blockIdx.y;
    const int kv_idx = blockIdx.x * blockDim.x + threadIdx.x;  // 이 스레드가 담당하는 K/V 행 (forward의 q_idx 자리에 대응)

    const __half* Qbh = Q + (size_t)bh * seq_len * HEAD_DIM;
    const __half* Kbh = K + (size_t)bh * seq_len * HEAD_DIM;
    const __half* Vbh = V + (size_t)bh * seq_len * HEAD_DIM;
    const __half* DObh = DO + (size_t)bh * seq_len * HEAD_DIM;
    const float* Lbh = L + (size_t)bh * seq_len;
    const float* Dbh = D + (size_t)bh * seq_len;

    // 이 스레드가 맡은 K_j, V_j는 전체 루프 동안 안 바뀌므로 한 번만 읽어서(fp32로 변환) 레지스터에 고정.
    float k_row[HEAD_DIM], v_row[HEAD_DIM];
#pragma unroll
    for (int t = 0; t < HEAD_DIM; t++) {
        k_row[t] = (kv_idx < seq_len) ? __half2float(Kbh[(size_t)kv_idx * HEAD_DIM + t]) : 0.0f;
        v_row[t] = (kv_idx < seq_len) ? __half2float(Vbh[(size_t)kv_idx * HEAD_DIM + t]) : 0.0f;
    }
    float dk_acc[HEAD_DIM], dv_acc[HEAD_DIM];
#pragma unroll
    for (int t = 0; t < HEAD_DIM; t++) { dk_acc[t] = 0.0f; dv_acc[t] = 0.0f; }

    // ===== 모든 Q 타일을 순회 =====
    for (int q0 = 0; q0 < seq_len; q0 += BLOCK_M) {
        // 이 Q 타일에 필요한 Q, dO, L(logsumexp), D를 블록 전체가 협력해서
        // shared memory로 복사 (forward의 K/V 타일 로딩과 같은 패턴, 역할만 반대).
        int q_idx_load = q0 + threadIdx.x;
        for (int t = 0; t < HEAD_DIM; t++) {
            Qs[threadIdx.x * HEAD_DIM + t] = (q_idx_load < seq_len) ? __half2float(Qbh[(size_t)q_idx_load * HEAD_DIM + t]) : 0.0f;
            DOs[threadIdx.x * HEAD_DIM + t] = (q_idx_load < seq_len) ? __half2float(DObh[(size_t)q_idx_load * HEAD_DIM + t]) : 0.0f;
        }
        Ls[threadIdx.x] = (q_idx_load < seq_len) ? Lbh[q_idx_load] : 0.0f;
        Ds[threadIdx.x] = (q_idx_load < seq_len) ? Dbh[q_idx_load] : 0.0f;
        __syncthreads();

        // causal 최적화: 이 K/V 행(kv_idx)보다 훨씬 앞쪽에 있는 Q 타일은
        // (causal 마스킹 규칙상 q_idx >= kv_idx여야 유효하므로) 통째로 스킵 가능.
        // m_lo = kv_idx가 속한 Q-타일-정렬 경계값 -- 그보다 앞선 타일은 전부 무효.
        int m_lo = causal ? (kv_idx / BLOCK_M) * BLOCK_M : 0;
        if (kv_idx < seq_len && q0 + BLOCK_M > m_lo) {
#pragma unroll 4
            for (int r = 0; r < BLOCK_M; r++) {  // 이 Q 타일 안의 행들을 하나씩 순회 (shared memory에서 읽음)
                int q_idx = q0 + r;
                bool valid = q_idx < seq_len && (!causal || q_idx >= kv_idx);
                if (!valid) continue;
                // S_ij, P_ij를 저장해뒀던 L(logsumexp)로 "다시 계산"(recompute) --
                // 온라인 최댓값 추적을 처음부터 다시 안 해도 되는 이유가 이것.
                float s = 0.0f;
#pragma unroll
                for (int t = 0; t < HEAD_DIM; t++) s += Qs[r * HEAD_DIM + t] * k_row[t];
                s *= scale;
                float p = expf(s - Ls[r]);
                float dp = 0.0f;
#pragma unroll
                for (int t = 0; t < HEAD_DIM; t++) dp += DOs[r * HEAD_DIM + t] * v_row[t];
                float ds = p * (dp - Ds[r]) * scale;
#pragma unroll
                for (int t = 0; t < HEAD_DIM; t++) {
                    dv_acc[t] += p * DOs[r * HEAD_DIM + t];   // dV_j += P_ij * dO_i
                    dk_acc[t] += ds * Qs[r * HEAD_DIM + t];   // dK_j += dS_ij * Q_i
                }
            }
        }
        __syncthreads();
    }

    // 모든 Q 타일을 다 순회한 뒤 딱 한 번만 씀 -- 이 스레드(=이 K/V 행)를 다른
    // 어떤 스레드도 건드리지 않으므로 atomic이 필요 없다. (fp32 -> fp16 변환해서 저장)
    if (kv_idx < seq_len) {
        for (int t = 0; t < HEAD_DIM; t++) {
            DK[((size_t)bh * seq_len + kv_idx) * HEAD_DIM + t] = __float2half(dk_acc[t]);
            DV[((size_t)bh * seq_len + kv_idx) * HEAD_DIM + t] = __float2half(dv_acc[t]);
        }
    }
}

// [dQ 계산] forward 커널과 구조가 완전히 동일 (블록 하나 = Q 타일 하나,
// 그 안에서 K/V 타일을 순회) -- 다른 점은 온라인 소프트맥스 대신
// 저장해둔 L을 그대로 써서 재계산만 한다는 것, 그리고 누적하는 게
// O_i가 아니라 dQ_i라는 것뿐. (flash_attention.py의 _bwd_dq_kernel과 동일)
template <int HEAD_DIM, int BLOCK_N>
__global__ void flash_bwd_dq_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                     const __half* __restrict__ V, const __half* __restrict__ DO,
                                     const float* __restrict__ L, const float* __restrict__ D,
                                     __half* __restrict__ DQ, int seq_len, float scale, bool causal) {
    extern __shared__ float smem[];
    const int BLOCK_M = blockDim.x;
    float* Ks = smem;                    // [BLOCK_N][HEAD_DIM]
    float* Vs = Ks + BLOCK_N * HEAD_DIM;  // [BLOCK_N][HEAD_DIM]

    const int bh = blockIdx.y;
    const int q_idx = blockIdx.x * BLOCK_M + threadIdx.x;

    const __half* Kbh = K + (size_t)bh * seq_len * HEAD_DIM;
    const __half* Vbh = V + (size_t)bh * seq_len * HEAD_DIM;

    // 이 스레드가 맡은 Q_i, dO_i, L_i, D_i는 전체 루프 동안 안 바뀌므로 한 번만 로드(+fp32 변환).
    float q_row[HEAD_DIM], do_row[HEAD_DIM];
    float l_i = 0.0f, d_i = 0.0f;
#pragma unroll
    for (int t = 0; t < HEAD_DIM; t++) {
        q_row[t] = (q_idx < seq_len) ? __half2float(Q[((size_t)bh * seq_len + q_idx) * HEAD_DIM + t]) : 0.0f;
        do_row[t] = (q_idx < seq_len) ? __half2float(DO[((size_t)bh * seq_len + q_idx) * HEAD_DIM + t]) : 0.0f;
    }
    if (q_idx < seq_len) { l_i = L[(size_t)bh * seq_len + q_idx]; d_i = D[(size_t)bh * seq_len + q_idx]; }

    float dq_acc[HEAD_DIM];
#pragma unroll
    for (int t = 0; t < HEAD_DIM; t++) dq_acc[t] = 0.0f;

    // forward와 똑같은 causal 스킵: 이 Q 타일보다 뒤에 있는 K/V는 애초에 안 봐도 됨.
    const int hi = causal ? min(seq_len, blockIdx.x * BLOCK_M + BLOCK_M) : seq_len;

    for (int kv0 = 0; kv0 < hi; kv0 += BLOCK_N) {
        // K, V 타일을 shared memory로 협력 로드 (forward와 완전히 같은 패턴)
        for (int idx = threadIdx.x; idx < BLOCK_N * HEAD_DIM; idx += BLOCK_M) {
            int r = idx / HEAD_DIM, c = idx % HEAD_DIM;
            int kv_idx = kv0 + r;
            Ks[idx] = (kv_idx < seq_len) ? __half2float(Kbh[(size_t)kv_idx * HEAD_DIM + c]) : 0.0f;
            Vs[idx] = (kv_idx < seq_len) ? __half2float(Vbh[(size_t)kv_idx * HEAD_DIM + c]) : 0.0f;
        }
        __syncthreads();

        if (q_idx < seq_len) {
#pragma unroll
            for (int c = 0; c < BLOCK_N; c++) {
                int kv_idx = kv0 + c;
                bool valid = kv_idx < seq_len && (!causal || q_idx >= kv_idx);
                if (!valid) continue;
                // 저장해둔 l_i로 P_ij를 재계산 (forward에서 했던 것과 동일한 recompute 트릭)
                float s = 0.0f;
#pragma unroll
                for (int t = 0; t < HEAD_DIM; t++) s += q_row[t] * Ks[c * HEAD_DIM + t];
                s *= scale;
                float p = expf(s - l_i);
                float dp = 0.0f;
#pragma unroll
                for (int t = 0; t < HEAD_DIM; t++) dp += do_row[t] * Vs[c * HEAD_DIM + t];
                float ds = p * (dp - d_i) * scale;
#pragma unroll
                for (int t = 0; t < HEAD_DIM; t++) dq_acc[t] += ds * Ks[c * HEAD_DIM + t];  // dQ_i += dS_ij * K_j
            }
        }
        __syncthreads();
    }

    if (q_idx < seq_len)
        for (int t = 0; t < HEAD_DIM; t++) DQ[((size_t)bh * seq_len + q_idx) * HEAD_DIM + t] = __float2half(dq_acc[t]);
}

// [순수 커널 실행 버전] D 버퍼가 이미 할당되어 있다고 가정하고 커널만 실행.
// benchmark.cu가 반복 측정 시 할당 오버헤드 없이 순수 커널 시간만 재기 위해 사용.
template <int HEAD_DIM, int BLOCK_N>
void flash_backward_launch_buf(const __half* Q, const __half* K, const __half* V, const __half* O,
                                const __half* DO, const float* L, float* D, __half* DQ, __half* DK,
                                __half* DV, int BH, int seq_len, bool causal, int BLOCK_M) {
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    // 1단계: D_i = rowsum(dO_i * O_i) 계산 (O(seq_len) 작업, 가벼움)
    dim3 grid_d((seq_len + 255) / 256, BH);
    flash_bwd_d_kernel<<<grid_d, 256>>>(O, DO, D, seq_len, HEAD_DIM);

    // 2단계: dK, dV 계산 (K/V 타일 기준으로 병렬화)
    size_t smem_dkdv = (size_t)(2 * BLOCK_M) * HEAD_DIM * sizeof(float) + 2 * BLOCK_M * sizeof(float);
    auto kernel_dkdv = flash_bwd_dkdv_kernel<HEAD_DIM>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel_dkdv, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dkdv));
    dim3 grid_n((seq_len + BLOCK_M - 1) / BLOCK_M, BH);
    kernel_dkdv<<<grid_n, BLOCK_M, smem_dkdv>>>(Q, K, V, DO, L, D, DK, DV, seq_len, scale, causal);

    // 3단계: dQ 계산 (Q 타일 기준, forward와 같은 grid 구성)
    size_t smem_dq = (size_t)(2 * BLOCK_N) * HEAD_DIM * sizeof(float);
    auto kernel_dq = flash_bwd_dq_kernel<HEAD_DIM, BLOCK_N>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq));
    dim3 grid_m((seq_len + BLOCK_M - 1) / BLOCK_M, BH);
    kernel_dq<<<grid_m, BLOCK_M, smem_dq>>>(Q, K, V, DO, L, D, DQ, seq_len, scale, causal);
}

// [편의용 wrapper] D 스크래치 버퍼를 alloc으로 직접 할당해주는 버전.
// test_flash.cu처럼 "한 번만 돌려서 정답과 비교"하는 용도로 사용.
template <int HEAD_DIM, int BLOCK_N>
void flash_backward_launch(DeviceAllocTracker& alloc, const __half* Q, const __half* K, const __half* V,
                            const __half* O, const __half* DO, const float* L, __half* DQ, __half* DK,
                            __half* DV, int BH, int seq_len, bool causal, int BLOCK_M) {
    size_t d_bytes = (size_t)BH * seq_len * sizeof(float);
    float* D = (float*)alloc.alloc(d_bytes);
    flash_backward_launch_buf<HEAD_DIM, BLOCK_N>(Q, K, V, O, DO, L, D, DQ, DK, DV, BH, seq_len, causal, BLOCK_M);
    alloc.free(D, d_bytes);
}
