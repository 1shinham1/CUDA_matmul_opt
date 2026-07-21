// [naive/standard attention] 논문의 Algorithm 0(forward) / Algorithm 3(backward)를
// 그대로 옮긴, 합쳐지지(fuse) 않은 CUDA 커널들.
//
// 핵심 특징: 중간 결과물인 S, P, dP, dS가 전부 "각각 별도의 seq_len x seq_len 크기
// 버퍼"로 global memory(=HBM, GPU의 큰 메모리)에 저장된다. 즉 커널 하나가
// 끝날 때마다 결과를 HBM에 썼다가, 다음 커널이 그걸 다시 읽어오는 식.
// 이게 바로 논문이 "느리다"고 지적하는 바로 그 패턴 -- HBM 접근이
// O(seq_len^2)만큼 여러 번 일어난다.
//
// 커널 자체도 일부러 제일 단순하게 짰다 (shared memory 활용도 최소,
// 타일링 없음). 이건 "빠르게 최적화된 baseline"이 아니라 "논문이 비교
// 대상으로 삼는 표준적인 방식"을 재현하려는 것이 목적.
//
// [정밀도] Q,K,V,O,dQ,dK,dV,dO는 논문과 동일하게 __half(fp16)로 저장한다.
// 반면 S,P,dP,dS(softmax가 걸리는 seq_len x seq_len 스크래치 버퍼)는 fp32로
// 유지한다 -- softmax의 exp/합산은 정밀도에 민감해서, 실제 mixed-precision
// 구현(PyTorch AMP 등)도 이 부분은 fp32로 계산하는 게 표준 관행이다.
// 그래서 각 커널 안에서 fp16 입력을 읽을 때는 __half2float()로, fp16
// 출력에 쓸 때는 __float2half()로 변환하는 코드가 추가되어 있다.
#pragma once
#include "common.cuh"

// ===== forward: S = scale * Q K^T =====
// [스레드 배치] 이 커널은 "출력 행렬 S의 원소 하나 = 스레드 하나"로 배치한다.
//   blockIdx.z = bh   (배치*헤드 인덱스)
//   blockIdx.y * blockDim.y + threadIdx.y = i   (S의 행, 즉 쿼리 위치)
//   blockIdx.x * blockDim.x + threadIdx.x = j   (S의 열, 즉 키 위치)
// 즉 (i,j) 좌표를 가진 스레드 하나가 Q_i, K_j의 내적을 직접 계산해서
// S[bh,i,j]에 쓴다. causal=true면 j>i(미래 위치)인 칸은 -infinity로
// 채워서 나중 softmax에서 확률이 0이 되게 만든다.
__global__ void naive_scores_kernel(const __half* Q, const __half* K, float* S,
                                     int seq_len, int d, float scale, bool causal) {
    int bh = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len || j >= seq_len) return;  // seq_len이 블록 크기로 안 나누어떨어질 때의 경계 처리
    const __half* Qi = Q + ((size_t)bh * seq_len + i) * d;
    const __half* Kj = K + ((size_t)bh * seq_len + j) * d;
    float* Sij = S + ((size_t)bh * seq_len + i) * seq_len + j;
    if (causal && j > i) { *Sij = -INFINITY; return; }
    float acc = 0.0f;
    for (int t = 0; t < d; t++) acc += __half2float(Qi[t]) * __half2float(Kj[t]);  // fp16 읽어서 fp32로 누적
    *Sij = acc * scale;
}

// ===== forward: P = softmax(S), 행(row) 단위로 =====
// [스레드 배치] 이번엔 스레드 하나가 아니라 "블록 하나 = 행 하나"를 담당.
//   blockIdx.x = i (행 번호), blockIdx.y = bh
// 한 행의 원소 seq_len개를 blockDim.x개의 스레드가 나눠서 처리 (stride 방식:
// threadIdx.x, threadIdx.x+blockDim.x, threadIdx.x+2*blockDim.x, ...).
// softmax는 "행 전체의 최댓값"과 "행 전체의 합"이 필요한 연산이라, 각
// 스레드가 자기 담당 구간의 부분 최댓값/부분합을 구한 뒤 shared memory
// 배열(sh[])을 이용한 병렬 리덕션(reduction)으로 블록 전체의 최댓값/합을
// 구한다 (전형적인 "tree reduction": 절반씩 접어가며 합치기).
// S, P 둘 다 fp32 -- softmax는 정밀도 손실을 피하려고 fp16으로 안 내린다.
__global__ void naive_softmax_kernel(const float* S, float* P, int seq_len) {
    int bh = blockIdx.y;
    int i = blockIdx.x;
    const float* Si = S + ((size_t)bh * seq_len + i) * seq_len;
    float* Pi = P + ((size_t)bh * seq_len + i) * seq_len;

    extern __shared__ float sh[];  // 블록 내 스레드들이 공유하는 온칩 메모리 (launch 시 크기 지정)

    // 1단계: 각 스레드가 자기 담당 구간의 최댓값을 구하고, shared memory에 모은 뒤
    // tree reduction으로 "이 행 전체의 최댓값"을 구한다 (오버플로우 방지용, 논문 3.1절 수식).
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) local_max = fmaxf(local_max, Si[j]);
    sh[threadIdx.x] = local_max;
    __syncthreads();  // 블록 내 모든 스레드가 sh[]에 다 쓸 때까지 대기
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sh[0];  // 리덕션이 끝나면 sh[0]에 전체 최댓값이 남음
    __syncthreads();        // sh[]를 다음 리덕션(합계용)에 재사용하기 전 동기화

    // 2단계: exp(S - row_max)를 계산하며 동시에 합계도 구함 (마찬가지로 리덕션)
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        float e = expf(Si[j] - row_max);
        Pi[j] = e;  // 정규화 전 값을 일단 P에 저장해둠
        local_sum += e;
    }
    sh[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = sh[0];
    __syncthreads();

    // 3단계: 각자 담당 구간을 row_sum으로 나눠서 진짜 softmax 확률을 완성
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) Pi[j] /= row_sum;
}

// ===== forward: O = P V =====
// [스레드 배치] 출력 O[bh,i,t] 원소 하나 = 스레드 하나. t는 head_dim 축.
// 한 스레드가 P의 i번째 행 전체(seq_len개)와 V의 t번째 열 전체(seq_len개)를 내적.
__global__ void naive_output_kernel(const float* P, const __half* V, __half* O, int seq_len, int d) {
    int bh = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len || t >= d) return;
    const float* Pi = P + ((size_t)bh * seq_len + i) * seq_len;
    const __half* Vb = V + (size_t)bh * seq_len * d;
    float acc = 0.0f;
    for (int j = 0; j < seq_len; j++) acc += Pi[j] * __half2float(Vb[j * d + t]);  // seq_len번 순회 (=O(seq_len) per 스레드, 전체는 O(seq_len^2))
    O[((size_t)bh * seq_len + i) * d + t] = __float2half(acc);
}

// ===== backward: dV_j = sum_i P_ij * dO_i =====
// forward의 O=PV를 P에 대해 미분한 것의 역방향: dV = P^T dO.
// [스레드 배치] dV[bh,j,t] 원소 하나 = 스레드 하나. i에 대해 seq_len번 순회하며 누적.
__global__ void naive_dv_kernel(const float* P, const __half* dO, __half* dV, int seq_len, int d) {
    int bh = blockIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= seq_len || t >= d) return;
    const float* Pb = P + (size_t)bh * seq_len * seq_len;
    const __half* dOb = dO + (size_t)bh * seq_len * d;
    float acc = 0.0f;
    for (int i = 0; i < seq_len; i++) acc += Pb[i * seq_len + j] * __half2float(dOb[i * d + t]);  // 열(j) 방향으로 훑어야 해서 Pb[i*seq_len+j] 처럼 stride seq_len 접근
    dV[((size_t)bh * seq_len + j) * d + t] = __float2half(acc);
}

// ===== backward: dP_ij = dO_i . V_j =====
// naive_scores_kernel과 구조가 거의 동일 (내적 하나 = 스레드 하나).
__global__ void naive_dp_kernel(const __half* dO, const __half* V, float* dP, int seq_len, int d) {
    int bh = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len || j >= seq_len) return;
    const __half* dOi = dO + ((size_t)bh * seq_len + i) * d;
    const __half* Vj = V + ((size_t)bh * seq_len + j) * d;
    float acc = 0.0f;
    for (int t = 0; t < d; t++) acc += __half2float(dOi[t]) * __half2float(Vj[t]);
    dP[((size_t)bh * seq_len + i) * seq_len + j] = acc;
}

// ===== backward: D_i = sum_j P_ij * dP_ij, dS_ij = P_ij * (dP_ij - D_i) * scale =====
// softmax의 그래디언트 공식 (야코비안이 diag(P) - P P^T 형태인 것에서 유도됨).
// [스레드 배치] naive_softmax_kernel과 완전히 같은 패턴: 블록 하나 = 행 하나,
// D_i를 구하기 위해 먼저 "P_ij*dP_ij의 행 합계"를 tree reduction으로 구하고,
// 그 다음 각 스레드가 자기 담당 원소들의 dS를 계산. (전부 fp32, naive_softmax_kernel과 동일 이유)
__global__ void naive_ds_kernel(const float* P, const float* dP, float* dS, int seq_len, float scale) {
    int bh = blockIdx.y;
    int i = blockIdx.x;
    const float* Pi = P + ((size_t)bh * seq_len + i) * seq_len;
    const float* dPi = dP + ((size_t)bh * seq_len + i) * seq_len;
    float* dSi = dS + ((size_t)bh * seq_len + i) * seq_len;

    extern __shared__ float sh[];
    float local = 0.0f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) local += Pi[j] * dPi[j];
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float D = sh[0];  // D_i = 이 행의 sum(P*dP)
    __syncthreads();

    for (int j = threadIdx.x; j < seq_len; j += blockDim.x)
        dSi[j] = Pi[j] * (dPi[j] - D) * scale;
}

// ===== backward: dQ_i = sum_j dS_ij * K_j =====
__global__ void naive_dq_kernel(const float* dS, const __half* K, __half* dQ, int seq_len, int d) {
    int bh = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len || t >= d) return;
    const float* dSi = dS + ((size_t)bh * seq_len + i) * seq_len;
    const __half* Kb = K + (size_t)bh * seq_len * d;
    float acc = 0.0f;
    for (int j = 0; j < seq_len; j++) acc += dSi[j] * __half2float(Kb[j * d + t]);
    dQ[((size_t)bh * seq_len + i) * d + t] = __float2half(acc);
}

// ===== backward: dK_j = sum_i dS_ij * Q_i =====
// naive_dv_kernel과 마찬가지로 열(j) 방향 접근이라 dSb[i*seq_len+j] stride 패턴.
__global__ void naive_dk_kernel(const float* dS, const __half* Q, __half* dK, int seq_len, int d) {
    int bh = blockIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= seq_len || t >= d) return;
    const float* dSb = dS + (size_t)bh * seq_len * seq_len;
    const __half* Qb = Q + (size_t)bh * seq_len * d;
    float acc = 0.0f;
    for (int i = 0; i < seq_len; i++) acc += dSb[i * seq_len + j] * __half2float(Qb[i * d + t]);
    dK[((size_t)bh * seq_len + j) * d + t] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// [호스트 측 launcher 함수들] 커널을 실제로 <<<grid, block>>> 문법으로
// 실행시켜주는 C++ 함수들. 이 아래부터는 GPU가 아니라 CPU에서 실행되는
// "커널을 던지는" 코드.
// ---------------------------------------------------------------------------

// naive_forward()가 반환하는, S/P 버퍼 포인터를 담는 작은 구조체.
// (backward 단계에서 forward가 만든 P를 다시 읽어야 하므로 넘겨줘야 함)
struct NaiveForwardBuffers {
    float *S, *P;
    size_t sp_bytes;
};

// [순수 커널 실행 버전] 메모리 할당은 전혀 하지 않고, 이미 크기가 맞게
// 준비된 S, P 버퍼를 받아서 커널 3개(scores -> softmax -> output)를
// 순서대로 실행만 한다. benchmark.cu가 이 버전을 쓰는 이유: 반복 측정할
// 때마다 cudaMalloc/cudaFree를 하면 "커널 자체의 속도"가 아니라 "할당
// 오버헤드"까지 같이 재게 되어버리기 때문.
inline void naive_forward_launch(const __half* Q, const __half* K, const __half* V, __half* O,
                                  float* S, float* P, int BH, int seq_len, int d, bool causal) {
    float scale = 1.0f / sqrtf((float)d);
    dim3 block2(16, 16);  // 16x16=256 스레드/블록 (2D 그리드 커널들의 표준 블록 크기)
    dim3 grid_s((seq_len + 15) / 16, (seq_len + 15) / 16, BH);  // (i,j,bh) 축을 각각 커버하도록 그리드 크기 계산 (올림 나눗셈)
    naive_scores_kernel<<<grid_s, block2>>>(Q, K, S, seq_len, d, scale, causal);

    dim3 grid_sm(seq_len, BH);  // 행(i) 하나당 블록 하나, 배치*헤드(bh)당 또 하나
    naive_softmax_kernel<<<grid_sm, 256, 256 * sizeof(float)>>>(S, P, seq_len);  // 세 번째 인자: shared memory 크기(바이트)

    dim3 grid_o((d + 15) / 16, (seq_len + 15) / 16, BH);
    naive_output_kernel<<<grid_o, block2>>>(P, V, O, seq_len, d);
}

// forward와 동일한 이유로, dP/dS도 미리 할당된 버퍼를 받아서 커널
// 5개(dv, dp, ds, dq, dk)를 순서대로 실행만 한다.
inline void naive_backward_launch(const __half* Q, const __half* K, const __half* V, const __half* dO,
                                   const float* P, float* dP, float* dS, __half* dQ, __half* dK,
                                   __half* dV, int BH, int seq_len, int d, bool causal) {
    float scale = 1.0f / sqrtf((float)d);
    dim3 block2(16, 16);
    dim3 grid_dv((d + 15) / 16, (seq_len + 15) / 16, BH);
    naive_dv_kernel<<<grid_dv, block2>>>(P, dO, dV, seq_len, d);

    dim3 grid_dp((seq_len + 15) / 16, (seq_len + 15) / 16, BH);
    naive_dp_kernel<<<grid_dp, block2>>>(dO, V, dP, seq_len, d);

    dim3 grid_ds(seq_len, BH);
    naive_ds_kernel<<<grid_ds, 256, 256 * sizeof(float)>>>(P, dP, dS, seq_len, scale);

    naive_dq_kernel<<<grid_dv, block2>>>(dS, K, dQ, seq_len, d);  // dQ, dK는 (seq_len,d) 크기라 grid_dv와 모양이 같아서 재사용
    naive_dk_kernel<<<grid_dv, block2>>>(dS, Q, dK, seq_len, d);
}

// [편의용 wrapper 버전] S/P/dP/dS 버퍼를 tracker를 통해 직접 할당까지
// 해주는 버전. test_naive.cu처럼 "한 번만 실행하고 결과 확인"하는
// 용도에서는 매번 할당하는 오버헤드가 문제되지 않으므로, 코드를
// 간결하게 쓰기 위해 이 편의용 함수들을 쓴다.
inline NaiveForwardBuffers naive_forward(DeviceAllocTracker& alloc, const __half* Q, const __half* K,
                                          const __half* V, __half* O, int BH, int seq_len, int d,
                                          bool causal) {
    NaiveForwardBuffers buf;
    buf.sp_bytes = (size_t)BH * seq_len * seq_len * sizeof(float);
    buf.S = (float*)alloc.alloc(buf.sp_bytes);
    buf.P = (float*)alloc.alloc(buf.sp_bytes);
    naive_forward_launch(Q, K, V, O, buf.S, buf.P, BH, seq_len, d, causal);
    return buf;
}

inline void naive_backward(DeviceAllocTracker& alloc, const __half* Q, const __half* K,
                            const __half* V, const __half* dO, const NaiveForwardBuffers& fwd,
                            __half* dQ, __half* dK, __half* dV, int BH, int seq_len, int d, bool causal) {
    float* dP = (float*)alloc.alloc(fwd.sp_bytes);
    float* dS = (float*)alloc.alloc(fwd.sp_bytes);
    naive_backward_launch(Q, K, V, dO, fwd.P, dP, dS, dQ, dK, dV, BH, seq_len, d, causal);
    alloc.free(dP, fwd.sp_bytes);
    alloc.free(dS, fwd.sp_bytes);
}
