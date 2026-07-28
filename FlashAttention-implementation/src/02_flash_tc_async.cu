// [02] tensor-core forward, 01_flash_tc_naive 대비 cp.async 비동기 벡터 로드 도입.
// 참고: github.com/sonnyli/flash_attention_from_scratch 의 kernel 1
// (`async+load_0_0_0_tiles`) -- 그 프로젝트는 kernel 1부터 이미 GMEM->SMEM
// 로드를 스칼라 대입이 아니라 하드웨어 cp.async(비동기 카피 유닛)로 처리한다.
// 01은 스레드마다 half 원소 하나씩 "읽어서 레지스터를 거쳐 쓰는" 완전 동기
// 로드였는데, 이번엔 그 부분만 16바이트(half 8개) 단위 cp.async로 바꾼다.
//
// [바뀐 것] Q/K/V GMEM->SMEM 로드: 원소별 대입 -> __pipeline_memcpy_async
// 16B 청크. sonnyli kernel 1처럼 eager(다음 타일 프리페치)는 아직 없다 --
// 매 반복 "issue -> commit -> wait_prior(0) -> syncthreads" 순서로, 로드
// 자체의 메커니즘만 바꾼 것(레지스터를 거치지 않고 GMEM->SMEM 직접 전송,
// 스레드당 8배 더 큰 트랜잭션)이지 로드-연산 오버랩은 아직 없다.
//
// [안 바꾼 것] KV 타일 크기(16, WMMA_N 그대로), 워프 배치, WMMA 사용, smem
// 레이아웃/크기. README에 기록된 occupancy 함정(KV 16->64가 SM당 상주
// 블록을 2->1로 떨어뜨려 오히려 느려졌던 사례) 때문에, 이번 단계는 smem
// footprint를 전혀 늘리지 않는 축(로드 메커니즘)만 건드린다.
//
// [다음 후보] 여기서 얻는 이득이 확인되면, eager 프리페치(다음 KV 타일을
// 현재 타일 연산과 오버랩) -> K/V 버퍼 공유로 footprint 절감 -> 그 다음에야
// KV 타일 확장 순서로 간다 (README의 occupancy 경고 참고).
#include "flash_tc.h"
#include <cuda_pipeline.h>
#include <fstream>
#include <string>

template <int HEAD_DIM>
__global__ void flash_tc_async_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                                       const half* __restrict__ V, half* __restrict__ O,
                                       int seq_len, float scale, bool causal) {
    static_assert(HEAD_DIM == 64, "flash_tc_async_kernel is only implemented for HEAD_DIM=64");
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int BLOCK_M = WMMA_M * WARPS_PER_BLOCK;  // 64
    constexpr int BLOCK_N = WMMA_N;                      // 16: 01과 동일, 아직 안 넓힘
    constexpr int HD_TILES = HEAD_DIM / WMMA_K;
    constexpr int VEC = 16 / sizeof(half);  // 16B cp.async 청크당 half 개수 (=8)
    static_assert(HEAD_DIM % VEC == 0, "HEAD_DIM must be a multiple of VEC for 16B cp.async");

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int bh = blockIdx.y;
    const int q_tile_start = blockIdx.x * BLOCK_M;
    const int warp_row_start = q_tile_start + warp_id * WMMA_M;

    extern __shared__ char smem_raw[];
    size_t off = 0;
    half* Qs = reinterpret_cast<half*>(smem_raw + off);       off += align16((size_t)BLOCK_M * HEAD_DIM * sizeof(half));
    half* Ks = reinterpret_cast<half*>(smem_raw + off);       off += align16((size_t)BLOCK_N * HEAD_DIM * sizeof(half));
    half* Vs = reinterpret_cast<half*>(smem_raw + off);       off += align16((size_t)BLOCK_N * HEAD_DIM * sizeof(half));
    float* Stile = reinterpret_cast<float*>(smem_raw + off);  off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * BLOCK_N * sizeof(float));
    half* Ptile = reinterpret_cast<half*>(smem_raw + off);    off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * BLOCK_N * sizeof(half));
    float* Otile = reinterpret_cast<float*>(smem_raw + off);  off += align16((size_t)WARPS_PER_BLOCK * WMMA_M * HEAD_DIM * sizeof(float));

    const half* Qbh = Q + (size_t)bh * seq_len * HEAD_DIM;
    const half* Kbh = K + (size_t)bh * seq_len * HEAD_DIM;
    const half* Vbh = V + (size_t)bh * seq_len * HEAD_DIM;

    // Q 타일: 16B(half 8개) 단위 cp.async. 한 청크는 항상 한 행 안에 있으므로
    // (HEAD_DIM % VEC == 0) 행 전체가 범위 밖이면 그 청크 전체를 0으로 채운다.
    for (int idx = threadIdx.x; idx < (BLOCK_M * HEAD_DIM) / VEC; idx += blockDim.x) {
        int elem = idx * VEC;
        int r = elem / HEAD_DIM, c = elem % HEAD_DIM;
        int gq = q_tile_start + r;
        if (gq < seq_len) {
            __pipeline_memcpy_async(&Qs[elem], &Qbh[(size_t)gq * HEAD_DIM + c], 16);
        } else {
            *reinterpret_cast<uint4*>(&Qs[elem]) = make_uint4(0, 0, 0, 0);
        }
    }
    __pipeline_commit();

    float m_i = -INFINITY, l_i = 0.0f;
    float acc[HEAD_DIM];
#pragma unroll
    for (int t = 0; t < HEAD_DIM; t++) acc[t] = 0.0f;

    const int hi = causal ? min(seq_len, q_tile_start + BLOCK_M) : seq_len;
    __pipeline_wait_prior(0);
    __syncthreads();

    for (int kv0 = 0; kv0 < hi; kv0 += BLOCK_N) {
        // K/V 타일을 16B cp.async로 로드 (여전히 단일 버퍼 -- 이번 반복이
        // 쓸 데이터를 issue한 직후 바로 wait, 프리페치는 다음 단계).
        for (int idx = threadIdx.x; idx < (BLOCK_N * HEAD_DIM) / VEC; idx += blockDim.x) {
            int elem = idx * VEC;
            int r = elem / HEAD_DIM, c = elem % HEAD_DIM;
            int gk = kv0 + r;
            if (gk < seq_len) {
                __pipeline_memcpy_async(&Ks[elem], &Kbh[(size_t)gk * HEAD_DIM + c], 16);
                __pipeline_memcpy_async(&Vs[elem], &Vbh[(size_t)gk * HEAD_DIM + c], 16);
            } else {
                *reinterpret_cast<uint4*>(&Ks[elem]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&Vs[elem]) = make_uint4(0, 0, 0, 0);
            }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // ===== S = Q_warp @ K_tile^T (K를 col_major로 읽어 물리적 전치 없이 K^T) =====
        float* my_Stile = Stile + (size_t)warp_id * WMMA_M * BLOCK_N;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
        wmma::fill_fragment(s_frag, 0.0f);
#pragma unroll
        for (int kk = 0; kk < HD_TILES; kk++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, Qs + (size_t)warp_id * WMMA_M * HEAD_DIM + kk * WMMA_K, HEAD_DIM);
            wmma::load_matrix_sync(b_frag, Ks + kk * WMMA_K, HEAD_DIM);
            wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
        }
        wmma::store_matrix_sync(my_Stile, s_frag, BLOCK_N, wmma::mem_row_major);
        __syncwarp();

        // ===== online softmax 갱신 (행렬곱이 아니므로 lane_id<16만 사용) =====
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
                my_Ptile[row * BLOCK_N + c] = __float2half(p);
            }
            l_i += lsum;
            m_i = m_new;
        }
        __syncwarp();

        // ===== O_partial = P_tile @ V_tile =====
        float* my_Otile = Otile + (size_t)warp_id * WMMA_M * HEAD_DIM;
#pragma unroll
        for (int nn = 0; nn < HD_TILES; nn++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::fill_fragment(o_frag, 0.0f);
            wmma::load_matrix_sync(p_frag, my_Ptile, BLOCK_N);
            wmma::load_matrix_sync(v_frag, Vs + nn * WMMA_N, HEAD_DIM);
            wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            wmma::store_matrix_sync(my_Otile + nn * WMMA_N, o_frag, HEAD_DIM, wmma::mem_row_major);
        }
        __syncwarp();

        if (lane_id < 16) {
            const int row = lane_id;
            const float* my_Otile_row = my_Otile + row * HEAD_DIM;
#pragma unroll
            for (int t = 0; t < HEAD_DIM; t++) acc[t] += my_Otile_row[t];
        }
        __syncthreads();  // 다음 반복이 Ks/Vs(단일 버퍼)를 덮어쓰기 전 대기
    }

    if (lane_id < 16) {
        const int row = lane_id;
        const int q_idx = warp_row_start + row;
        if (q_idx < seq_len)
            for (int t = 0; t < HEAD_DIM; t++) O[((size_t)bh * seq_len + q_idx) * HEAD_DIM + t] = __float2half(acc[t] / l_i);
    }
}

template <int HEAD_DIM>
void flash_tc_async_launch(const __half* Q, const __half* K, const __half* V, __half* O,
                            int BH, int seq_len, bool causal) {
    static_assert(HEAD_DIM == 64, "flash_tc_async_launch is only implemented for HEAD_DIM=64");
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int BLOCK_M = WMMA_M * WARPS_PER_BLOCK;
    constexpr int BLOCK_N = WMMA_N;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    size_t smem_bytes = align16((size_t)BLOCK_M * HEAD_DIM * sizeof(half))
                       + align16((size_t)BLOCK_N * HEAD_DIM * sizeof(half))
                       + align16((size_t)BLOCK_N * HEAD_DIM * sizeof(half))
                       + align16((size_t)WARPS_PER_BLOCK * WMMA_M * BLOCK_N * sizeof(float))
                       + align16((size_t)WARPS_PER_BLOCK * WMMA_M * BLOCK_N * sizeof(half))
                       + align16((size_t)WARPS_PER_BLOCK * WMMA_M * HEAD_DIM * sizeof(float));

    auto kernel = flash_tc_async_kernel<HEAD_DIM>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    dim3 grid((seq_len + BLOCK_M - 1) / BLOCK_M, BH);
    dim3 block(32 * WARPS_PER_BLOCK);
    kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half*>(Q), reinterpret_cast<const half*>(K), reinterpret_cast<const half*>(V),
        reinterpret_cast<half*>(O), seq_len, scale, causal);
}

// ─────────────────────────────────────────────────────────────
constexpr int HEAD_DIM = DEFAULT_HEAD_DIM;
constexpr int FMA_BLOCK_N = 32;
constexpr int FMA_BLOCK_M = 64;

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
    __half *Q = (__half*)alloc.alloc(qkv_bytes), *K = (__half*)alloc.alloc(qkv_bytes);
    __half *V = (__half*)alloc.alloc(qkv_bytes), *O_tc = (__half*)alloc.alloc(qkv_bytes);
    __half *O_fma = (__half*)alloc.alloc(qkv_bytes);
    CUDA_CHECK(cudaMemcpy(Q, hQh.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(K, hKh.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V, hVh.data(), qkv_bytes, cudaMemcpyHostToDevice));

    flash_tc_async_launch<HEAD_DIM>(Q, K, V, O_tc, BH, seq_len, causal);
    int block_m = min(FMA_BLOCK_M, seq_len);
    flash_tiled_launch<HEAD_DIM, FMA_BLOCK_N>(Q, K, V, O_fma, BH, seq_len, causal, block_m);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> gOh(qkv_elems), gOfmaH(qkv_elems);
    CUDA_CHECK(cudaMemcpy(gOh.data(), O_tc, qkv_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gOfmaH.data(), O_fma, qkv_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> gO = to_float(gOh), gOfma = to_float(gOfmaH);

    char label[128];
    snprintf(label, sizeof(label), "BH=%d seq_len=%d d=%d causal=%d (TC vs CPU ref)", BH, seq_len, HEAD_DIM, causal);
    bool ok = verify_against_cpu(gO, refO, 3e-2f, label);  // WMMA는 축소 정밀도 누적이라 tol을 조금 더 둠
    ok &= verify_against_cpu(gO, gOfma, 3e-2f, "TC vs FMA-tiled kernel");

    alloc.free(Q, qkv_bytes); alloc.free(K, qkv_bytes); alloc.free(V, qkv_bytes);
    alloc.free(O_tc, qkv_bytes); alloc.free(O_fma, qkv_bytes);
    return ok;
}

int main(int argc, char** argv) {
    bool causal = argc > 1 && std::string(argv[1]) == "causal";

    printf("=== [02] tensor-core forward (cp.async vectorized loads, KV tile=16, single buffer) ===\n");
    bool all_ok = true;
    all_ok &= run_correctness_case(2, 128, false);
    all_ok &= run_correctness_case(2, 128, true);
    all_ok &= run_correctness_case(2, 130, false);  // seq_len이 BLOCK_M(64)로 안 나누어떨어지는 경계 케이스
    printf("%s\n\n", all_ok ? "All correctness cases OK" : "SOME CASES FAILED");

    printf("=== [02] benchmark vs. tiled-FMA baseline (BH=%d, head_dim=%d, causal=%d) ===\n",
           DEFAULT_BH, HEAD_DIM, causal);
    printf("%8s | %10s %10s %8s %10s\n", "seq_len", "tc_ms", "fma_ms", "speedup", "gflops");

    std::string csv_path = causal ? "results/results_02_tc_async_causal.csv" : "results/results_02_tc_async_noncausal.csv";
    std::ofstream csv(csv_path);
    csv << "seq_len,time_ms,fma_ms,speedup_vs_fma,gflops\n";

    for (int seq_len : default_seq_lens()) {
        size_t qkv_elems = (size_t)DEFAULT_BH * seq_len * HEAD_DIM;
        size_t qkv_bytes = qkv_elems * sizeof(__half);

        std::vector<float> hQ(qkv_elems), hK(qkv_elems), hV(qkv_elems);
        fill_random(hQ, 1); fill_random(hK, 2); fill_random(hV, 3);
        std::vector<__half> hQh = to_half(hQ), hKh = to_half(hK), hVh = to_half(hV);

        DeviceAllocTracker alloc;
        __half *Q = (__half*)alloc.alloc(qkv_bytes), *K = (__half*)alloc.alloc(qkv_bytes);
        __half *V = (__half*)alloc.alloc(qkv_bytes), *O = (__half*)alloc.alloc(qkv_bytes);
        CUDA_CHECK(cudaMemcpy(Q, hQh.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(K, hKh.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(V, hVh.data(), qkv_bytes, cudaMemcpyHostToDevice));

        auto run_tc = [&]() { flash_tc_async_launch<HEAD_DIM>(Q, K, V, O, DEFAULT_BH, seq_len, causal); };
        float tc_ms = time_ms_avg(run_tc, WARM_UP, iters_for(seq_len));

        auto run_fma = [&]() { flash_tiled_launch<HEAD_DIM, FMA_BLOCK_N>(Q, K, V, O, DEFAULT_BH, seq_len, causal, FMA_BLOCK_M); };
        float fma_ms = time_ms_avg(run_fma, WARM_UP, iters_for(seq_len));

        double gflops = attn_gflops(DEFAULT_BH, seq_len, HEAD_DIM, causal, tc_ms);
        double speedup = fma_ms / tc_ms;

        printf("%8d | %10.4f %10.4f %7.2fx %10.2f\n", seq_len, tc_ms, fma_ms, speedup, gflops);
        csv << seq_len << "," << tc_ms << "," << fma_ms << "," << speedup << "," << gflops << "\n";

        alloc.free(Q, qkv_bytes); alloc.free(K, qkv_bytes); alloc.free(V, qkv_bytes); alloc.free(O, qkv_bytes);
    }

    printf("\nSaved %s\n", csv_path.c_str());
    return all_ok ? 0 : 1;
}
