// [텐서 코어 버전 FlashAttention forward] flash_kernels.cuh의 flash_fwd_kernel과
// 알고리즘(타일링 + 온라인 소프트맥스)은 완전히 동일하지만, Q@K^T와 P@V 두
// 행렬곱을 손으로 짠 FMA 반복문 대신 CUDA 표준 텐서 코어 API인
// nvcuda::wmma(<mma.h>, 워프 단위 16x16x16 MMA)로 계산한다.
//
// CUDA 툴킷에 내장된 nvcuda::wmma API를 사용한다. 핵심 실행 단위는
// "16x16 타일 하나를 한 워프(32 스레드)가 협력해서 계산"하는 구조다.
//
// [스레드/워프 배치] flash_fwd_kernel(비-텐서코어 버전)은 "스레드 1개 = 쿼리
// 행 1개"였지만, WMMA는 반드시 "워프 32개 스레드가 협력해서 16x16 타일
// 하나"를 계산해야 하는 하드웨어 제약이 있어서 구조가 달라진다:
//   - 블록 하나 = Q 타일 64행 (WARPS_PER_BLOCK=4개 워프 x 16행)
//   - 워프 하나 = 그 중 16개 행을 전담 (WMMA_M=16)
//   - K/V 타일 폭은 WMMA_N=16 (한 번에 16개 키 위치씩 처리)
//   - Q@K^T의 축소 차원(head_dim)과 P@V의 축소 차원(key 위치 16개)은
//     WMMA_K=16 단위로 잘라서 여러 번 mma_sync를 누적한다.
// WMMA 연산(load_matrix_sync/mma_sync/store_matrix_sync) 자체는 32 스레드가
// 전부 참여해야 하지만, 그 사이사이의 온라인 소프트맥스 통계 갱신(행 최댓값/
// 합 추적)은 행렬곱이 아니라서 텐서 코어 대상이 아니다 -- 이 부분만 워프 32개
// 중 16개 레인(lane_id<16, 딱 이 워프가 담당하는 16개 행 개수만큼)이 맡아서
// flash_fwd_kernel과 똑같은 스칼라 방식으로 계산한다.
//
// [forward 커널 warp-tiling + 더블 버퍼링, HEAD_DIM=64 전용] flash_fwd_kernel_tc는
// 위 설명(K/V 타일 폭 WMMA_N=16)에서 한 단계 더 나아가, 한 번의 outer-loop
// 반복에서 KV 64개(WMMA_N의 4배, N_SUB=4개 서브타일)를 한꺼번에 처리한다 --
// sync 지점당 처리하는 mma_sync 양을 늘려서(4배) sync/디스패치 오버헤드 비중을
// 줄이기 위함. K/V는 stage 2개짜리 shared memory 버퍼로 더블 버퍼링하되(다음
// 타일 로드를 이번 타일 연산보다 먼저 "발행"해서 서로 다른 stage를 건드리는
// 두 작업이 겹쳐 스케줄링될 여지를 줌), 완료 대기는 __syncthreads() 하나로
// 통일한다 -- cuda::pipeline(cuda::memcpy_async 기반 진짜 비동기 복사)도
// 시도해봤지만, 이 커널의 워프별 lane-divergent 소프트맥스 구간과 얽히면서
// racecheck/synccheck가 실제 해저드(cp.async 폴백 경로 vs wmma 읽기 충돌,
// mbarrier 배리어 카운트 불일치)를 반복적으로 잡아냈다. __syncthreads() 기반
// 방식은 이 코드베이스 전체에서 이미 검증된 동기화 수단이라 안전하고, 서로
// 다른 stage를 건드리는 로드/연산 명령 사이에는 데이터 의존성이 없어 하드웨어가
// 여전히 자유롭게 겹쳐서 스케줄링할 수 있다 (자세한 배경은 OPTIMIZATION_PLAN.md
// 참고). backward 커널(flash_bwd_dkdv_kernel_tc/flash_bwd_dq_kernel_tc)은 이
// 변경과 무관하게 원래 설계(KV 16개씩, 싱글 버퍼링) 그대로다.
#pragma once
#include "common.cuh"
#include "flash_kernels.cuh"  // flash_bwd_d_kernel (D 계산, 텐서 코어 불필요)을 재사용
#include <mma.h>

using namespace nvcuda;

// 16바이트 경계로 올림 -- WMMA fragment 로드가 정렬된 포인터를 요구하므로,
// shared memory 안에서 서로 다른 타입(half/float) 영역을 이어붙일 때 각
// 영역의 시작 주소를 안전하게 정렬해두기 위한 헬퍼.
__host__ __device__ inline size_t align16(size_t bytes) { return (bytes + 15) & ~size_t(15); }

// [템플릿 매개변수] 이 warp-tiled + 더블 버퍼링 버전은 HEAD_DIM=64 전용이다
// (OPTIMIZATION_PLAN.md 참고 -- 다른 head_dim에 대한 fallback은 이번 범위 밖).
template <int HEAD_DIM>
__global__ void flash_fwd_kernel_tc(const half* __restrict__ Q, const half* __restrict__ K,
                                     const half* __restrict__ V, half* __restrict__ O,
                                     float* __restrict__ L, int seq_len, float scale, bool causal) {
    static_assert(HEAD_DIM == 64, "flash_fwd_kernel_tc (warp-tiled + double-buffered) is only implemented for HEAD_DIM=64 -- see OPTIMIZATION_PLAN.md");
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int BLOCK_M = WMMA_M * WARPS_PER_BLOCK;  // 64: 이 블록이 담당하는 Q 행 개수
    constexpr int BLOCK_N = 64;                         // 한 번에 처리하는 K/V 타일 폭 (warp-tiling: WMMA_N의 4배)
    constexpr int N_SUB = BLOCK_N / WMMA_N;             // 4: 한 반복 안의 16-wide 서브타일 개수
    constexpr int HD_TILES = HEAD_DIM / WMMA_K;         // head_dim을 16단위로 몇 조각으로 자르는지
    constexpr int STAGES = 2;                           // 더블 버퍼링 (shared memory 2벌 + __syncthreads())

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int bh = blockIdx.y;
    const int q_tile_start = blockIdx.x * BLOCK_M;
    const int warp_row_start = q_tile_start + warp_id * WMMA_M;  // 이 워프가 담당하는 16행의 시작 위치

    // ----- shared memory 레이아웃 (16바이트 정렬 경계로 이어붙임) -----
    // Qs/Stile/Ptile/Otile은 한 반복 안에서 만들어지고 소비되는(Qs는 커널 진입 시
    // 딱 한 번만 로드) 데이터라 더블 버퍼링이 필요 없다. Ks/Vs만 다음 타일 로드와
    // 현재 타일 연산을 겹치기 위해 STAGES=2개씩 갖는다.
    extern __shared__ char smem_raw[];
    size_t off = 0;
    half* Qs = reinterpret_cast<half*>(smem_raw + off);       off += align16((size_t)BLOCK_M * HEAD_DIM * sizeof(half));            // [BLOCK_M][HEAD_DIM]
    half* Ks = reinterpret_cast<half*>(smem_raw + off);       off += align16((size_t)STAGES * BLOCK_N * HEAD_DIM * sizeof(half));    // [STAGES][BLOCK_N][HEAD_DIM]
    half* Vs = reinterpret_cast<half*>(smem_raw + off);       off += align16((size_t)STAGES * BLOCK_N * HEAD_DIM * sizeof(half));    // [STAGES][BLOCK_N][HEAD_DIM]
    float* Stile = reinterpret_cast<float*>(smem_raw + off);  off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * BLOCK_N * sizeof(float)); // 워프별 [16][BLOCK_N] S 스크래치
    half* Ptile = reinterpret_cast<half*>(smem_raw + off);    off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * BLOCK_N * sizeof(half));  // 워프별 [16][BLOCK_N] P 스크래치
    float* Otile = reinterpret_cast<float*>(smem_raw + off);  off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * HEAD_DIM * sizeof(float)); // 워프별 [16][HEAD_DIM] P@V 결과 스크래치

    const half* Qbh = Q + (size_t)bh * seq_len * HEAD_DIM;
    const half* Kbh = K + (size_t)bh * seq_len * HEAD_DIM;
    const half* Vbh = V + (size_t)bh * seq_len * HEAD_DIM;

    // Q 타일(64행) 전체를 블록의 모든 스레드가 협력해서 shared memory로 복사.
    for (int idx = threadIdx.x; idx < BLOCK_M * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM, c = idx % HEAD_DIM;
        int gq = q_tile_start + r;
        Qs[idx] = (gq < seq_len) ? Qbh[(size_t)gq * HEAD_DIM + c] : __float2half(0.0f);
    }

    // 이 워프가 담당하는 16행에 대한 온라인 소프트맥스 상태.
    // lane_id < 16인 스레드만 의미 있는 값을 가진다 (그 레인이 담당하는 행 번호 = lane_id).
    float m_i = -INFINITY, l_i = 0.0f;
    float acc[HEAD_DIM];
#pragma unroll
    for (int t = 0; t < HEAD_DIM; t++) acc[t] = 0.0f;

    const int hi = causal ? min(seq_len, q_tile_start + BLOCK_M) : seq_len;
    __syncthreads();  // Qs 복사 완료 대기

    const int n_tiles = (hi + BLOCK_N - 1) / BLOCK_N;

    // KV 타일 하나(64개, 마지막이면 부분적일 수 있음)를 stage 버퍼로 협력 복사.
    // seq_len 경계를 넘는 부분은 0으로 채운다 (원본 커널과 동일한 방식 -- 이
    // 타일이 마지막이 아니면 항상 전부 유효하고, 마지막일 때만 일부가 0으로 채워짐).
    auto load_tile = [&](int stage, int kv0) {
        half* Kbuf = Ks + (size_t)stage * BLOCK_N * HEAD_DIM;
        half* Vbuf = Vs + (size_t)stage * BLOCK_N * HEAD_DIM;
        for (int idx = threadIdx.x; idx < BLOCK_N * HEAD_DIM; idx += blockDim.x) {
            int r = idx / HEAD_DIM, c = idx % HEAD_DIM;
            int gk = kv0 + r;
            Kbuf[idx] = (gk < seq_len) ? Kbh[(size_t)gk * HEAD_DIM + c] : __float2half(0.0f);
            Vbuf[idx] = (gk < seq_len) ? Vbh[(size_t)gk * HEAD_DIM + c] : __float2half(0.0f);
        }
    };

    // 이미 로드가 끝난 stage 버퍼(Kcur/Vcur)로 S/softmax/P@V를 계산하고 그 결과를
    // 이 워프의 acc[]/m_i/l_i에 누적한다.
    auto compute_tile = [&](int stage, int kv0) {
        half* Kcur = Ks + (size_t)stage * BLOCK_N * HEAD_DIM;
        half* Vcur = Vs + (size_t)stage * BLOCK_N * HEAD_DIM;

        // ===== S = Q_i @ K_j^T, N_SUB=4개의 16-wide 서브타일로 나눠 텐서 코어로 계산 =====
        // K를 col_major로 읽으면(row-major로 저장된 [BLOCK_N][HEAD_DIM]을 그대로) 물리적
        // 전치 없이 K^T를 얻는다 -- 서브타일 n은 그 안의 16행짜리 창을 고르는 것뿐.
        float* my_Stile = Stile + (size_t)warp_id * WMMA_M * BLOCK_N;
#pragma unroll
        for (int n = 0; n < N_SUB; n++) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
            wmma::fill_fragment(s_frag, 0.0f);
#pragma unroll
            for (int kk = 0; kk < HD_TILES; kk++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, Qs + (size_t)warp_id * WMMA_M * HEAD_DIM + kk * WMMA_K, HEAD_DIM);
                wmma::load_matrix_sync(b_frag, Kcur + (size_t)n * WMMA_N * HEAD_DIM + kk * WMMA_K, HEAD_DIM);
                wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
            }
            wmma::store_matrix_sync(my_Stile + n * WMMA_N, s_frag, BLOCK_N, wmma::mem_row_major);
        }
        __syncwarp();

        // ===== 온라인 소프트맥스 갱신 (BLOCK_N=64 전체에 대해 한 번에): 행렬곱이 아니므로
        // 텐서 코어 대상이 아님. 이 워프의 32레인 중 16개(lane_id<16)만 활성화해서
        // flash_fwd_kernel과 동일한 스칼라 방식으로 계산. =====
        half* my_Ptile = Ptile + (size_t)warp_id * WMMA_M * BLOCK_N;
        if (lane_id < 16) {
            const int row = lane_id;
            const int q_idx = warp_row_start + row;
            float scores[BLOCK_N];
            float blk_max = -INFINITY;
#pragma unroll
            for (int c = 0; c < BLOCK_N; c++) {
                int kv_idx = kv0 + c;
                float s = -INFINITY;
                if (q_idx < seq_len && kv_idx < seq_len && !(causal && kv_idx > q_idx))
                    s = my_Stile[row * BLOCK_N + c] * scale;
                scores[c] = s;
                blk_max = fmaxf(blk_max, s);
            }
            float m_new = fmaxf(m_i, blk_max);
            float alpha = expf(m_i - m_new);
            l_i *= alpha;
#pragma unroll
            for (int t = 0; t < HEAD_DIM; t++) acc[t] *= alpha;
            float lsum = 0.0f;
#pragma unroll
            for (int c = 0; c < BLOCK_N; c++) {
                float p = expf(scores[c] - m_new);
                lsum += p;
                my_Ptile[row * BLOCK_N + c] = __float2half(p);  // 다음 P@V 행렬곱 입력용으로 fp16 캐스팅
            }
            l_i += lsum;
            m_i = m_new;
        }
        __syncwarp();  // my_Ptile 쓰기 완료를 워프 전체(WMMA 호출은 32레인 전부 참여)에 알림

        // ===== O_partial = P_ij @ V_j, N_SUB=4개의 16-wide KV 서브청크에 대해 reduction 후
        // head_dim 슬라이스(nn)당 하나의 o_frag에 누적 -- Otile은 여전히 [16][HEAD_DIM]만 필요. =====
        float* my_Otile = Otile + (size_t)warp_id * WMMA_M * HEAD_DIM;
#pragma unroll
        for (int nn = 0; nn < HD_TILES; nn++) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::fill_fragment(o_frag, 0.0f);
#pragma unroll
            for (int sc = 0; sc < N_SUB; sc++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
                wmma::load_matrix_sync(p_frag, my_Ptile + sc * WMMA_N, BLOCK_N);
                wmma::load_matrix_sync(v_frag, Vcur + (size_t)sc * WMMA_N * HEAD_DIM + nn * WMMA_N, HEAD_DIM);
                wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            }
            wmma::store_matrix_sync(my_Otile + nn * WMMA_N, o_frag, HEAD_DIM, wmma::mem_row_major);
        }
        __syncwarp();

        // 이 타일의 P@V 결과(Otile)를 온라인 누적값(acc)에 더한다 -- 마찬가지로
        // 행렬곱이 아니라 단순 원소별 덧셈이라 텐서 코어 대상이 아님.
        if (lane_id < 16) {
            const int row = lane_id;
            const float* my_Otile_row = my_Otile + row * HEAD_DIM;
#pragma unroll
            for (int t = 0; t < HEAD_DIM; t++) acc[t] += my_Otile_row[t];
        }
    };

    if (n_tiles > 0) load_tile(0, 0);  // 프롤로그: 타일 0 로드
    __syncthreads();                   // 타일 0 로드 완료 대기

    for (int t = 0; t < n_tiles; t++) {
        const int stage = t % STAGES;
        const int kv0 = t * BLOCK_N;
        // 다음 타일 로드를 이번 타일 연산보다 먼저 "발행"한다 -- 서로 다른 stage
        // 버퍼를 건드리므로 데이터 의존성이 없어, 하드웨어가 이 로드와 아래
        // compute_tile()의 텐서 코어 연산을 자유롭게 겹쳐서 스케줄링할 수 있다.
        if (t + 1 < n_tiles) load_tile((t + 1) % STAGES, (t + 1) * BLOCK_N);
        compute_tile(stage, kv0);
        __syncthreads();  // 이번 반복의 로드+연산이 모두 끝났는지 확인 (다음 반복이 다른
                          // stage를 읽거나, 두 반복 뒤에 이 stage를 다시 덮어쓰기 전에)
    }

    if (lane_id < 16) {
        const int row = lane_id;
        const int q_idx = warp_row_start + row;
        if (q_idx < seq_len) {
            for (int t = 0; t < HEAD_DIM; t++) O[((size_t)bh * seq_len + q_idx) * HEAD_DIM + t] = __float2half(acc[t] / l_i);
            L[(size_t)bh * seq_len + q_idx] = m_i + logf(l_i);
        }
    }
}

template <int HEAD_DIM>
void flash_forward_launch_tc(const __half* Q, const __half* K, const __half* V, __half* O, float* L,
                              int BH, int seq_len, bool causal) {
    static_assert(HEAD_DIM == 64, "flash_forward_launch_tc (warp-tiled + double-buffered) is only implemented for HEAD_DIM=64 -- see OPTIMIZATION_PLAN.md");
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int BLOCK_M = 16 * WARPS_PER_BLOCK;  // 64
    constexpr int BLOCK_N = 64;                     // warp-tiled KV 타일 폭 (OPTIMIZATION_PLAN.md)
    constexpr int STAGES = 2;                       // 더블 버퍼링
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    size_t smem_bytes = align16((size_t)BLOCK_M * HEAD_DIM * sizeof(half))
                       + align16((size_t)STAGES * BLOCK_N * HEAD_DIM * sizeof(half))
                       + align16((size_t)STAGES * BLOCK_N * HEAD_DIM * sizeof(half))
                       + align16((size_t)WARPS_PER_BLOCK * 16 * BLOCK_N * sizeof(float))
                       + align16((size_t)WARPS_PER_BLOCK * 16 * BLOCK_N * sizeof(half))
                       + align16((size_t)WARPS_PER_BLOCK * 16 * HEAD_DIM * sizeof(float));

    auto kernel = flash_fwd_kernel_tc<HEAD_DIM>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    dim3 grid((seq_len + BLOCK_M - 1) / BLOCK_M, BH);
    dim3 block(32 * WARPS_PER_BLOCK);  // 128 threads = 4 warps
    kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half*>(Q), reinterpret_cast<const half*>(K), reinterpret_cast<const half*>(V),
        reinterpret_cast<half*>(O), L, seq_len, scale, causal);
}

// ===========================================================================
// [텐서 코어 버전 backward] flash_kernels.cuh의 flash_bwd_dkdv_kernel /
// flash_bwd_dq_kernel과 같은 두 커널 분리(atomic 회피) 구조를 유지하되,
// 그 안의 행렬곱들을 WMMA로 바꾼다. forward는 행렬곱이 2개(Q@K^T, P@V)뿐인데
// backward는 4개(S=Q@K^T 재계산, dP=dO@V^T, dV=P^T@dO, dK=dS^T@Q  /  또는
// dQ 커널 쪽은 S, dP 재계산 + dQ=dS@K)라서 forward보다 코드가 두 배 이상 커진다.
//
// [전치가 "공짜"인 곳들] forward에서 K를 col_major로 재해석해서 K^T를
// 얻었던 것과 같은 트릭을 backward에서도 두 번 더 쓴다:
//   - dV = P^T @ dO: P가 [Q행][KV행] row-major로 저장되어 있으므로,
//     col_major로 읽으면 그대로 P^T가 된다 (물리적 전치 불필요).
//   - dK = dS^T @ Q: dS도 마찬가지로 [Q행][KV행] row-major 저장이라
//     col_major 재해석으로 dS^T를 얻는다.
// 반면 dQ = dS @ K, dK 커널의 S/dP 재계산(Q@K^T, dO@V^T)은 forward와 동일한
// 패턴이고, dQ = dS @ K는 애초에 전치가 필요 없어서(둘 다 자연스러운
// row-major 방향) 그냥 바로 쓴다.
// ===========================================================================

// [dK, dV 계산, WMMA] "워프 하나 = KV 행 16개 전담"은 유지하되, WMMA 제약상
// Q를 16행씩 잘라 순회한다 (FMA 버전은 64행씩 잘라 순회했음 -- WMMA 타일
// 크기가 16이라 더 잘게 쪼갠 것). causal skip 최적화는 이 버전에서는 생략
// (원소별 마스킹으로 정확성은 유지하되, 완전히 마스킹된 타일도 계산은
// 함 -- FMA 버전 대비 성능 여지가 남아있는 부분).
template <int HEAD_DIM>
__global__ void flash_bwd_dkdv_kernel_tc(const half* __restrict__ Q, const half* __restrict__ K,
                                          const half* __restrict__ V, const half* __restrict__ DO,
                                          const float* __restrict__ L, const float* __restrict__ D,
                                          half* __restrict__ DK, half* __restrict__ DV,
                                          int seq_len, float scale, bool causal) {
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int BLOCK_N = WMMA_M * WARPS_PER_BLOCK;  // 64: 이 블록이 담당하는 KV 행 개수
    constexpr int Q_CHUNK = 16;                          // 반복마다 처리하는 Q 행 개수 (WMMA 타일 크기)
    constexpr int HD_TILES = HEAD_DIM / WMMA_K;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int bh = blockIdx.y;
    const int kv_tile_start = blockIdx.x * BLOCK_N;
    const int warp_kv_start = kv_tile_start + warp_id * WMMA_M;  // 이 워프가 담당하는 16개 KV 행의 시작 위치

    extern __shared__ char smem_raw[];
    size_t off = 0;
    half* Qs = reinterpret_cast<half*>(smem_raw + off);        off += align16((size_t)Q_CHUNK * HEAD_DIM * sizeof(half));   // 블록 공유: 현재 Q 청크
    half* DOs = reinterpret_cast<half*>(smem_raw + off);       off += align16((size_t)Q_CHUNK * HEAD_DIM * sizeof(half));
    float* Ls = reinterpret_cast<float*>(smem_raw + off);      off += align16((size_t)Q_CHUNK * sizeof(float));
    float* Ds = reinterpret_cast<float*>(smem_raw + off);      off += align16((size_t)Q_CHUNK * sizeof(float));
    half* Ks_w = reinterpret_cast<half*>(smem_raw + off);      off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * HEAD_DIM * sizeof(half));  // 워프별: 이 워프가 담당하는 KV 16행 (커널 시작 시 한 번만 로드)
    half* Vs_w = reinterpret_cast<half*>(smem_raw + off);      off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * HEAD_DIM * sizeof(half));
    float* Stile = reinterpret_cast<float*>(smem_raw + off);   off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * WMMA_N * sizeof(float));    // 워프별 스크래치
    float* dPtile = reinterpret_cast<float*>(smem_raw + off);  off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * WMMA_N * sizeof(float));
    half* Ptile = reinterpret_cast<half*>(smem_raw + off);     off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * WMMA_N * sizeof(half));
    half* dStile = reinterpret_cast<half*>(smem_raw + off);    off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * WMMA_N * sizeof(half));
    float* dVscratch = reinterpret_cast<float*>(smem_raw + off); off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * HEAD_DIM * sizeof(float));
    float* dKscratch = reinterpret_cast<float*>(smem_raw + off); off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * HEAD_DIM * sizeof(float));

    const half* Kbh = K + (size_t)bh * seq_len * HEAD_DIM;
    const half* Vbh = V + (size_t)bh * seq_len * HEAD_DIM;
    const half* Qbh = Q + (size_t)bh * seq_len * HEAD_DIM;
    const half* DObh = DO + (size_t)bh * seq_len * HEAD_DIM;
    const float* Lbh = L + (size_t)bh * seq_len;
    const float* Dbh = D + (size_t)bh * seq_len;

    half* my_Ks = Ks_w + (size_t)warp_id * WMMA_M * HEAD_DIM;
    half* my_Vs = Vs_w + (size_t)warp_id * WMMA_M * HEAD_DIM;
    // 이 워프가 담당하는 K,V 16행은 커널 내내 안 바뀌므로 루프 시작 전 한 번만 로드 (32레인 협력).
    for (int idx = lane_id; idx < WMMA_M * HEAD_DIM; idx += 32) {
        int r = idx / HEAD_DIM, c = idx % HEAD_DIM;
        int gk = warp_kv_start + r;
        my_Ks[idx] = (gk < seq_len) ? Kbh[(size_t)gk * HEAD_DIM + c] : __float2half(0.0f);
        my_Vs[idx] = (gk < seq_len) ? Vbh[(size_t)gk * HEAD_DIM + c] : __float2half(0.0f);
    }

    float dv_acc[HEAD_DIM], dk_acc[HEAD_DIM];
#pragma unroll
    for (int t = 0; t < HEAD_DIM; t++) { dv_acc[t] = 0.0f; dk_acc[t] = 0.0f; }

    __syncthreads();  // 모든 워프의 K,V 로드 완료 대기

    for (int q0 = 0; q0 < seq_len; q0 += Q_CHUNK) {
        // Q 청크(16행), dO, L, D를 블록 전체가 협력해서 로드 (모든 워프가 공유해서 읽음).
        for (int idx = threadIdx.x; idx < Q_CHUNK * HEAD_DIM; idx += blockDim.x) {
            int r = idx / HEAD_DIM, c = idx % HEAD_DIM;
            int gq = q0 + r;
            Qs[idx] = (gq < seq_len) ? Qbh[(size_t)gq * HEAD_DIM + c] : __float2half(0.0f);
            DOs[idx] = (gq < seq_len) ? DObh[(size_t)gq * HEAD_DIM + c] : __float2half(0.0f);
        }
        for (int idx = threadIdx.x; idx < Q_CHUNK; idx += blockDim.x) {
            int gq = q0 + idx;
            Ls[idx] = (gq < seq_len) ? Lbh[gq] : 0.0f;
            Ds[idx] = (gq < seq_len) ? Dbh[gq] : 0.0f;
        }
        __syncthreads();

        // ===== S = Q_chunk @ K_warp^T =====
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
        wmma::fill_fragment(s_frag, 0.0f);
#pragma unroll
        for (int kk = 0; kk < HD_TILES; kk++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, Qs + kk * WMMA_K, HEAD_DIM);
            wmma::load_matrix_sync(b_frag, my_Ks + kk * WMMA_K, HEAD_DIM);
            wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
        }
        float* my_Stile = Stile + (size_t)warp_id * WMMA_M * WMMA_N;
        wmma::store_matrix_sync(my_Stile, s_frag, WMMA_N, wmma::mem_row_major);

        // ===== dP = DO_chunk @ V_warp^T =====
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dp_frag;
        wmma::fill_fragment(dp_frag, 0.0f);
#pragma unroll
        for (int kk = 0; kk < HD_TILES; kk++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, DOs + kk * WMMA_K, HEAD_DIM);
            wmma::load_matrix_sync(b_frag, my_Vs + kk * WMMA_K, HEAD_DIM);
            wmma::mma_sync(dp_frag, a_frag, b_frag, dp_frag);
        }
        float* my_dPtile = dPtile + (size_t)warp_id * WMMA_M * WMMA_N;
        wmma::store_matrix_sync(my_dPtile, dp_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();

        // ===== 원소별: p = exp(S*scale - L), dS = p*(dP-D)*scale (lane = Q행, 0..15) =====
        half* my_Ptile = Ptile + (size_t)warp_id * WMMA_M * WMMA_N;
        half* my_dStile = dStile + (size_t)warp_id * WMMA_M * WMMA_N;
        if (lane_id < 16) {
            const int qr = lane_id;
            const int q_idx = q0 + qr;
#pragma unroll
            for (int c = 0; c < 16; c++) {
                const int kv_idx = warp_kv_start + c;
                const bool valid = q_idx < seq_len && kv_idx < seq_len && (!causal || q_idx >= kv_idx);
                float p = 0.0f, ds = 0.0f;
                if (valid) {
                    float s = my_Stile[qr * 16 + c] * scale;
                    p = expf(s - Ls[qr]);
                    float dp = my_dPtile[qr * 16 + c];
                    ds = p * (dp - Ds[qr]) * scale;
                }
                my_Ptile[qr * 16 + c] = __float2half(p);
                my_dStile[qr * 16 + c] = __float2half(ds);
            }
        }
        __syncwarp();

        // ===== dV += P^T @ DO_chunk (P를 col_major로 재해석해서 전치 대신 사용) =====
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> pT_frag;
        wmma::load_matrix_sync(pT_frag, my_Ptile, 16);
        float* my_dVscratch = dVscratch + (size_t)warp_id * WMMA_M * HEAD_DIM;
#pragma unroll
        for (int nn = 0; nn < HD_TILES; nn++) {
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> do_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dv_frag;
            wmma::fill_fragment(dv_frag, 0.0f);
            wmma::load_matrix_sync(do_frag, DOs + nn * WMMA_N, HEAD_DIM);
            wmma::mma_sync(dv_frag, pT_frag, do_frag, dv_frag);
            wmma::store_matrix_sync(my_dVscratch + nn * WMMA_N, dv_frag, HEAD_DIM, wmma::mem_row_major);
        }

        // ===== dK += dS^T @ Q_chunk (dS도 col_major 재해석으로 전치) =====
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> dsT_frag;
        wmma::load_matrix_sync(dsT_frag, my_dStile, 16);
        float* my_dKscratch = dKscratch + (size_t)warp_id * WMMA_M * HEAD_DIM;
#pragma unroll
        for (int nn = 0; nn < HD_TILES; nn++) {
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dk_frag;
            wmma::fill_fragment(dk_frag, 0.0f);
            wmma::load_matrix_sync(q_frag, Qs + nn * WMMA_N, HEAD_DIM);
            wmma::mma_sync(dk_frag, dsT_frag, q_frag, dk_frag);
            wmma::store_matrix_sync(my_dKscratch + nn * WMMA_N, dk_frag, HEAD_DIM, wmma::mem_row_major);
        }
        __syncwarp();

        // WMMA 결과(dVscratch/dKscratch)를 이 워프가 지금까지 누적해온 레지스터 배열에 더한다
        // (lane = KV행, 0..15) -- forward에서 acc[]를 누적하던 것과 같은 relay 패턴.
        if (lane_id < 16) {
            const int kvr = lane_id;
            const float* dv_src = my_dVscratch + (size_t)kvr * HEAD_DIM;
            const float* dk_src = my_dKscratch + (size_t)kvr * HEAD_DIM;
#pragma unroll
            for (int t = 0; t < HEAD_DIM; t++) { dv_acc[t] += dv_src[t]; dk_acc[t] += dk_src[t]; }
        }
        __syncthreads();  // 다음 Q 청크가 Qs/DOs/Ls/Ds를 덮어쓰기 전 대기
    }

    if (lane_id < 16) {
        const int kvr = lane_id;
        const int kv_idx = warp_kv_start + kvr;
        if (kv_idx < seq_len) {
#pragma unroll
            for (int t = 0; t < HEAD_DIM; t++) {
                DK[((size_t)bh * seq_len + kv_idx) * HEAD_DIM + t] = __float2half(dk_acc[t]);
                DV[((size_t)bh * seq_len + kv_idx) * HEAD_DIM + t] = __float2half(dv_acc[t]);
            }
        }
    }
}

// [dQ 계산, WMMA] forward 커널과 구조가 가장 비슷하다 (워프 하나 = Q 행 16개
// 전담, KV를 16행씩 순회). dQ = dS @ K는 전치가 필요 없어서 (둘 다 자연스러운
// row-major 방향) forward의 P@V 계산과 거의 동일한 형태.
template <int HEAD_DIM>
__global__ void flash_bwd_dq_kernel_tc(const half* __restrict__ Q, const half* __restrict__ K,
                                        const half* __restrict__ V, const half* __restrict__ DO,
                                        const float* __restrict__ L, const float* __restrict__ D,
                                        half* __restrict__ DQ, int seq_len, float scale, bool causal) {
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int BLOCK_M = WMMA_M * WARPS_PER_BLOCK;  // 64: 이 블록이 담당하는 Q 행 개수
    constexpr int KV_CHUNK = 16;
    constexpr int HD_TILES = HEAD_DIM / WMMA_K;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int bh = blockIdx.y;
    const int q_tile_start = blockIdx.x * BLOCK_M;
    const int warp_q_start = q_tile_start + warp_id * WMMA_M;

    extern __shared__ char smem_raw[];
    size_t off = 0;
    half* Qs_w = reinterpret_cast<half*>(smem_raw + off);   off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * HEAD_DIM * sizeof(half));  // 워프별: 이 워프의 Q 16행 (한 번만 로드)
    half* DOs_w = reinterpret_cast<half*>(smem_raw + off);  off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * HEAD_DIM * sizeof(half));
    float* Ls_w = reinterpret_cast<float*>(smem_raw + off); off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * sizeof(float));
    float* Ds_w = reinterpret_cast<float*>(smem_raw + off); off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * sizeof(float));
    half* Ks = reinterpret_cast<half*>(smem_raw + off);     off += align16((size_t)KV_CHUNK * HEAD_DIM * sizeof(half));  // 블록 공유: 현재 KV 청크
    half* Vs = reinterpret_cast<half*>(smem_raw + off);     off += align16((size_t)KV_CHUNK * HEAD_DIM * sizeof(half));
    float* Stile = reinterpret_cast<float*>(smem_raw + off);  off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * WMMA_N * sizeof(float));  // 워프별 스크래치
    float* dPtile = reinterpret_cast<float*>(smem_raw + off); off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * WMMA_N * sizeof(float));
    half* dStile = reinterpret_cast<half*>(smem_raw + off);   off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * WMMA_N * sizeof(half));
    float* dQscratch = reinterpret_cast<float*>(smem_raw + off); off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * HEAD_DIM * sizeof(float));

    const half* Qbh = Q + (size_t)bh * seq_len * HEAD_DIM;
    const half* DObh = DO + (size_t)bh * seq_len * HEAD_DIM;
    const half* Kbh = K + (size_t)bh * seq_len * HEAD_DIM;
    const half* Vbh = V + (size_t)bh * seq_len * HEAD_DIM;
    const float* Lbh = L + (size_t)bh * seq_len;
    const float* Dbh = D + (size_t)bh * seq_len;

    half* my_Qs = Qs_w + (size_t)warp_id * WMMA_M * HEAD_DIM;
    half* my_DOs = DOs_w + (size_t)warp_id * WMMA_M * HEAD_DIM;
    float* my_Ls = Ls_w + (size_t)warp_id * WMMA_M;
    float* my_Ds = Ds_w + (size_t)warp_id * WMMA_M;

    // 이 워프가 담당하는 Q,dO 16행 + L,D는 커널 내내 안 바뀌므로 루프 시작 전 한 번만 로드.
    for (int idx = lane_id; idx < WMMA_M * HEAD_DIM; idx += 32) {
        int r = idx / HEAD_DIM, c = idx % HEAD_DIM;
        int gq = warp_q_start + r;
        my_Qs[idx] = (gq < seq_len) ? Qbh[(size_t)gq * HEAD_DIM + c] : __float2half(0.0f);
        my_DOs[idx] = (gq < seq_len) ? DObh[(size_t)gq * HEAD_DIM + c] : __float2half(0.0f);
    }
    if (lane_id < 16) {
        int gq = warp_q_start + lane_id;
        my_Ls[lane_id] = (gq < seq_len) ? Lbh[gq] : 0.0f;
        my_Ds[lane_id] = (gq < seq_len) ? Dbh[gq] : 0.0f;
    }

    float dq_acc[HEAD_DIM];
#pragma unroll
    for (int t = 0; t < HEAD_DIM; t++) dq_acc[t] = 0.0f;

    const int hi = causal ? min(seq_len, q_tile_start + BLOCK_M) : seq_len;  // forward와 동일한 causal 스킵
    __syncthreads();

    for (int kv0 = 0; kv0 < hi; kv0 += KV_CHUNK) {
        for (int idx = threadIdx.x; idx < KV_CHUNK * HEAD_DIM; idx += blockDim.x) {
            int r = idx / HEAD_DIM, c = idx % HEAD_DIM;
            int gk = kv0 + r;
            Ks[idx] = (gk < seq_len) ? Kbh[(size_t)gk * HEAD_DIM + c] : __float2half(0.0f);
            Vs[idx] = (gk < seq_len) ? Vbh[(size_t)gk * HEAD_DIM + c] : __float2half(0.0f);
        }
        __syncthreads();

        // ===== S = Q_warp @ K_chunk^T =====
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
        wmma::fill_fragment(s_frag, 0.0f);
#pragma unroll
        for (int kk = 0; kk < HD_TILES; kk++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, my_Qs + kk * WMMA_K, HEAD_DIM);
            wmma::load_matrix_sync(b_frag, Ks + kk * WMMA_K, HEAD_DIM);
            wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
        }
        float* my_Stile = Stile + (size_t)warp_id * WMMA_M * WMMA_N;
        wmma::store_matrix_sync(my_Stile, s_frag, WMMA_N, wmma::mem_row_major);

        // ===== dP = DO_warp @ V_chunk^T =====
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dp_frag;
        wmma::fill_fragment(dp_frag, 0.0f);
#pragma unroll
        for (int kk = 0; kk < HD_TILES; kk++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, my_DOs + kk * WMMA_K, HEAD_DIM);
            wmma::load_matrix_sync(b_frag, Vs + kk * WMMA_K, HEAD_DIM);
            wmma::mma_sync(dp_frag, a_frag, b_frag, dp_frag);
        }
        float* my_dPtile = dPtile + (size_t)warp_id * WMMA_M * WMMA_N;
        wmma::store_matrix_sync(my_dPtile, dp_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();

        // ===== 원소별: dS = P*(dP-D)*scale (lane = Q행, 0..15) =====
        half* my_dStile = dStile + (size_t)warp_id * WMMA_M * WMMA_N;
        if (lane_id < 16) {
            const int qr = lane_id;
            const int q_idx = warp_q_start + qr;
#pragma unroll
            for (int c = 0; c < 16; c++) {
                const int kv_idx = kv0 + c;
                const bool valid = q_idx < seq_len && kv_idx < seq_len && (!causal || q_idx >= kv_idx);
                float ds = 0.0f;
                if (valid) {
                    float s = my_Stile[qr * 16 + c] * scale;
                    float p = expf(s - my_Ls[qr]);
                    float dp = my_dPtile[qr * 16 + c];
                    ds = p * (dp - my_Ds[qr]) * scale;
                }
                my_dStile[qr * 16 + c] = __float2half(ds);
            }
        }
        __syncwarp();

        // ===== dQ += dS @ K_chunk (전치 불필요 -- 둘 다 자연스러운 방향) =====
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> ds_frag;
        wmma::load_matrix_sync(ds_frag, my_dStile, 16);
        float* my_dQscratch = dQscratch + (size_t)warp_id * WMMA_M * HEAD_DIM;
#pragma unroll
        for (int nn = 0; nn < HD_TILES; nn++) {
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> k_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dq_frag;
            wmma::fill_fragment(dq_frag, 0.0f);
            wmma::load_matrix_sync(k_frag, Ks + nn * WMMA_N, HEAD_DIM);
            wmma::mma_sync(dq_frag, ds_frag, k_frag, dq_frag);
            wmma::store_matrix_sync(my_dQscratch + nn * WMMA_N, dq_frag, HEAD_DIM, wmma::mem_row_major);
        }
        __syncwarp();

        if (lane_id < 16) {
            const int qr = lane_id;
            const float* src = my_dQscratch + (size_t)qr * HEAD_DIM;
#pragma unroll
            for (int t = 0; t < HEAD_DIM; t++) dq_acc[t] += src[t];
        }
        __syncthreads();
    }

    if (lane_id < 16) {
        const int qr = lane_id;
        const int q_idx = warp_q_start + qr;
        if (q_idx < seq_len)
#pragma unroll
            for (int t = 0; t < HEAD_DIM; t++) DQ[((size_t)bh * seq_len + q_idx) * HEAD_DIM + t] = __float2half(dq_acc[t]);
    }
}

// [순수 커널 실행 버전] D는 flash_kernels.cuh의 flash_bwd_d_kernel(텐서 코어
// 불필요한 O(seq_len) 원소별 작업)을 그대로 재사용한다.
template <int HEAD_DIM>
void flash_backward_launch_tc_buf(const __half* Q, const __half* K, const __half* V, const __half* O,
                                   const __half* DO, const float* L, float* D, __half* DQ, __half* DK,
                                   __half* DV, int BH, int seq_len, bool causal) {
    constexpr int WARPS_PER_BLOCK = 4;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    dim3 grid_d((seq_len + 255) / 256, BH);
    flash_bwd_d_kernel<<<grid_d, 256>>>(O, DO, D, seq_len, HEAD_DIM);

    const half* Qh = reinterpret_cast<const half*>(Q);
    const half* Kh = reinterpret_cast<const half*>(K);
    const half* Vh = reinterpret_cast<const half*>(V);
    const half* DOh = reinterpret_cast<const half*>(DO);
    half* DQh = reinterpret_cast<half*>(DQ);
    half* DKh = reinterpret_cast<half*>(DK);
    half* DVh = reinterpret_cast<half*>(DV);

    // dK, dV: 블록당 64 KV행 (워프 4개 x 16행)
    {
        size_t smem = align16((size_t)16 * HEAD_DIM * sizeof(half)) * 2                          // Qs, DOs
                    + align16((size_t)16 * sizeof(float)) * 2                                     // Ls, Ds
                    + align16((size_t)WARPS_PER_BLOCK * 16 * HEAD_DIM * sizeof(half)) * 2          // Ks_w, Vs_w
                    + align16((size_t)WARPS_PER_BLOCK * 16 * 16 * sizeof(float)) * 2               // Stile, dPtile
                    + align16((size_t)WARPS_PER_BLOCK * 16 * 16 * sizeof(half)) * 2                // Ptile, dStile
                    + align16((size_t)WARPS_PER_BLOCK * 16 * HEAD_DIM * sizeof(float)) * 2;        // dVscratch, dKscratch
        auto kernel = flash_bwd_dkdv_kernel_tc<HEAD_DIM>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        dim3 grid((seq_len + 63) / 64, BH);
        dim3 block(32 * WARPS_PER_BLOCK);
        kernel<<<grid, block, smem>>>(Qh, Kh, Vh, DOh, L, D, DKh, DVh, seq_len, scale, causal);
    }

    // dQ: 블록당 64 Q행 (워프 4개 x 16행), forward와 같은 grid 구성
    {
        size_t smem = align16((size_t)WARPS_PER_BLOCK * 16 * HEAD_DIM * sizeof(half)) * 2          // Qs_w, DOs_w
                    + align16((size_t)WARPS_PER_BLOCK * 16 * sizeof(float)) * 2                     // Ls_w, Ds_w
                    + align16((size_t)16 * HEAD_DIM * sizeof(half)) * 2                             // Ks, Vs
                    + align16((size_t)WARPS_PER_BLOCK * 16 * 16 * sizeof(float)) * 2                // Stile, dPtile
                    + align16((size_t)WARPS_PER_BLOCK * 16 * 16 * sizeof(half))                     // dStile
                    + align16((size_t)WARPS_PER_BLOCK * 16 * HEAD_DIM * sizeof(float));             // dQscratch
        auto kernel = flash_bwd_dq_kernel_tc<HEAD_DIM>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        dim3 grid((seq_len + 63) / 64, BH);
        dim3 block(32 * WARPS_PER_BLOCK);
        kernel<<<grid, block, smem>>>(Qh, Kh, Vh, DOh, L, D, DQh, seq_len, scale, causal);
    }
}

// [편의용 wrapper] D 스크래치 버퍼를 alloc으로 직접 할당해주는 버전 (test 용).
template <int HEAD_DIM>
void flash_backward_launch_tc(DeviceAllocTracker& alloc, const __half* Q, const __half* K, const __half* V,
                               const __half* O, const __half* DO, const float* L, __half* DQ, __half* DK,
                               __half* DV, int BH, int seq_len, bool causal) {
    size_t d_bytes = (size_t)BH * seq_len * sizeof(float);
    float* D = (float*)alloc.alloc(d_bytes);
    flash_backward_launch_tc_buf<HEAD_DIM>(Q, K, V, O, DO, L, D, DQ, DK, DV, BH, seq_len, causal);
    alloc.free(D, d_bytes);
}
