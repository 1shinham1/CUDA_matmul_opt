// [벤치마크] naive(텐서 코어 없음) vs flash FMA(텐서 코어 없음) vs
// flash WMMA(텐서 코어 사용) 실행 시간 비교, forward만 그리고 forward+backward
// 둘 다. 세 구현 모두 Q,K,V,O(,dQ,dK,dV,dO)는 fp16 저장이라 이 벤치마크는
// "텐서 코어를 실제로 켰을 때"의 순수한 속도 차이를 보여준다 (fp32 원시
// CUDA 버전의 naive vs flash 비교가 "메모리 접근 패턴만의 차이"를 보여줬던
// 것과 대비됨). backward의 텐서 코어 버전은 flash_kernels_tc.cuh의
// flash_bwd_dkdv_kernel_tc / flash_bwd_dq_kernel_tc를 사용.
#include "naive_kernels.cuh"
#include "flash_kernels.cuh"
#include "flash_kernels_tc.cuh"
#include <cstdio>
#include <fstream>
#include <string>

constexpr int HEAD_DIM = 64;
constexpr int FLASH_BLOCK_N = 32;
constexpr int FLASH_BLOCK_M = 64;

static bool fits_in_gpu(size_t bytes_needed) {
    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    return (double)bytes_needed * 1.2 < (double)free_b;
}

static int iters_for(int seq_len) {
    if (seq_len <= 1024) return 20;
    if (seq_len <= 4096) return 10;
    return 5;
}

int main(int argc, char** argv) {
    bool causal = argc > 1 && std::string(argv[1]) == "causal";
    int BH = 32;
    printf("FA_with_cuda: naive vs flash(FMA) vs flash(WMMA tensor core), fwd and fwd+bwd (BH=%d, head_dim=%d, causal=%d, fp16 storage)\n\n",
           BH, HEAD_DIM, causal);

    int seq_lens[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    bool naive_fwd_feasible = true, naive_fb_feasible = true;
    std::ofstream csv(causal ? "results/results_tc_causal.csv" : "results/results_tc_noncausal.csv");
    csv << "seq_len,naive_fwd_ms,fma_fwd_ms,tc_fwd_ms,naive_fb_ms,fma_fb_ms,tc_fb_ms,"
           "tc_fwd_vs_naive,tc_fwd_vs_fma,tc_fb_vs_naive,tc_fb_vs_fma\n";
    printf("%8s | %9s %9s %9s | %10s %10s %10s | %8s %8s\n",
           "seq_len", "naive_fwd", "fma_fwd", "tc_fwd", "naive_fb", "fma_fb", "tc_fb", "tc/naive", "tc/fma");

    for (int seq_len : seq_lens) {
        size_t qkv_elems = (size_t)BH * seq_len * HEAD_DIM;
        size_t qkv_bytes = qkv_elems * sizeof(__half);
        size_t sp_bytes = (size_t)BH * seq_len * seq_len * sizeof(float);
        size_t l_bytes = (size_t)BH * seq_len * sizeof(float);
        printf("seq_len=%d ...\n", seq_len); fflush(stdout);

        std::vector<float> hQ(qkv_elems), hK(qkv_elems), hV(qkv_elems), hdO(qkv_elems);
        fill_random(hQ, 1); fill_random(hK, 2); fill_random(hV, 3); fill_random(hdO, 4);
        std::vector<__half> hQh = to_half(hQ), hKh = to_half(hK), hVh = to_half(hV), hdOh = to_half(hdO);

        DeviceAllocTracker alloc;
        __half *Q = (__half*)alloc.alloc(qkv_bytes), *K = (__half*)alloc.alloc(qkv_bytes), *V = (__half*)alloc.alloc(qkv_bytes);
        __half *O = (__half*)alloc.alloc(qkv_bytes);
        __half *dO = (__half*)alloc.alloc(qkv_bytes);
        __half *dQ = (__half*)alloc.alloc(qkv_bytes), *dK = (__half*)alloc.alloc(qkv_bytes), *dV = (__half*)alloc.alloc(qkv_bytes);
        CUDA_CHECK(cudaMemcpy(Q, hQh.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(K, hKh.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(V, hVh.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dO, hdOh.data(), qkv_bytes, cudaMemcpyHostToDevice));

        double naive_fwd_ms = -1, naive_fb_ms = -1, fma_fwd_ms, fma_fb_ms, tc_fwd_ms, tc_fb_ms;

        // ---- naive ----
        if (naive_fwd_feasible && fits_in_gpu(2 * sp_bytes)) {
            float* S = (float*)alloc.alloc(sp_bytes);
            float* P = (float*)alloc.alloc(sp_bytes);
            auto run = [&]() { naive_forward_launch(Q, K, V, O, S, P, BH, seq_len, HEAD_DIM, causal); };
            naive_fwd_ms = time_ms_median(run, 3, iters_for(seq_len));

            if (naive_fb_feasible && fits_in_gpu(2 * sp_bytes)) {
                float* dP = (float*)alloc.alloc(sp_bytes);
                float* dS = (float*)alloc.alloc(sp_bytes);
                auto run_fb = [&]() {
                    naive_forward_launch(Q, K, V, O, S, P, BH, seq_len, HEAD_DIM, causal);
                    naive_backward_launch(Q, K, V, dO, P, dP, dS, dQ, dK, dV, BH, seq_len, HEAD_DIM, causal);
                };
                naive_fb_ms = time_ms_median(run_fb, 3, iters_for(seq_len));
                alloc.free(dP, sp_bytes); alloc.free(dS, sp_bytes);
            } else {
                naive_fb_feasible = false;
            }
            alloc.free(S, sp_bytes); alloc.free(P, sp_bytes);
        } else {
            naive_fwd_feasible = false;
            naive_fb_feasible = false;
        }

        // ---- flash FMA ----
        {
            float* L = (float*)alloc.alloc(l_bytes);
            auto run = [&]() { flash_forward_launch<HEAD_DIM, FLASH_BLOCK_N>(Q, K, V, O, L, BH, seq_len, causal, FLASH_BLOCK_M); };
            fma_fwd_ms = time_ms_median(run, 3, iters_for(seq_len));

            float* D = (float*)alloc.alloc(l_bytes);
            auto run_fb = [&]() {
                flash_forward_launch<HEAD_DIM, FLASH_BLOCK_N>(Q, K, V, O, L, BH, seq_len, causal, FLASH_BLOCK_M);
                flash_backward_launch_buf<HEAD_DIM, FLASH_BLOCK_N>(Q, K, V, O, dO, L, D, dQ, dK, dV, BH, seq_len, causal, FLASH_BLOCK_M);
            };
            fma_fb_ms = time_ms_median(run_fb, 3, iters_for(seq_len));
            alloc.free(D, l_bytes);
            alloc.free(L, l_bytes);
        }

        // ---- flash WMMA tensor core ----
        {
            float* L = (float*)alloc.alloc(l_bytes);
            auto run = [&]() { flash_forward_launch_tc<HEAD_DIM>(Q, K, V, O, L, BH, seq_len, causal); };
            tc_fwd_ms = time_ms_median(run, 3, iters_for(seq_len));

            float* D = (float*)alloc.alloc(l_bytes);
            auto run_fb = [&]() {
                flash_forward_launch_tc<HEAD_DIM>(Q, K, V, O, L, BH, seq_len, causal);
                flash_backward_launch_tc_buf<HEAD_DIM>(Q, K, V, O, dO, L, D, dQ, dK, dV, BH, seq_len, causal);
            };
            tc_fb_ms = time_ms_median(run_fb, 3, iters_for(seq_len));
            alloc.free(D, l_bytes);
            alloc.free(L, l_bytes);
        }

        double tc_fwd_vs_naive = naive_fwd_ms > 0 ? naive_fwd_ms / tc_fwd_ms : -1;
        double tc_fwd_vs_fma = fma_fwd_ms > 0 ? fma_fwd_ms / tc_fwd_ms : -1;
        double tc_fb_vs_naive = naive_fb_ms > 0 ? naive_fb_ms / tc_fb_ms : -1;
        double tc_fb_vs_fma = fma_fb_ms > 0 ? fma_fb_ms / tc_fb_ms : -1;

        printf("%8d | %9.3f %9.3f %9.3f | %10.3f %10.3f %10.3f | %7.2fx %7.2fx\n",
               seq_len, naive_fwd_ms, fma_fwd_ms, tc_fwd_ms, naive_fb_ms, fma_fb_ms, tc_fb_ms,
               tc_fb_vs_naive, tc_fb_vs_fma);
        csv << seq_len << "," << naive_fwd_ms << "," << fma_fwd_ms << "," << tc_fwd_ms << ","
            << naive_fb_ms << "," << fma_fb_ms << "," << tc_fb_ms << ","
            << tc_fwd_vs_naive << "," << tc_fwd_vs_fma << "," << tc_fb_vs_naive << "," << tc_fb_vs_fma << "\n";

        alloc.free(Q, qkv_bytes); alloc.free(K, qkv_bytes); alloc.free(V, qkv_bytes); alloc.free(O, qkv_bytes);
        alloc.free(dO, qkv_bytes); alloc.free(dQ, qkv_bytes); alloc.free(dK, qkv_bytes); alloc.free(dV, qkv_bytes);
    }
    printf("\nSaved.\n");
    return 0;
}
