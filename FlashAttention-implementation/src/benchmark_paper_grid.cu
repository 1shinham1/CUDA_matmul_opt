// [벤치마크] flash WMMA(텐서 코어) forward만, python2/논문 벤치마크 그리드로.
// head_dim=64 고정(flash_forward_launch_tc는 HEAD_DIM=64 전용 -- OPTIMIZATION_PLAN.md),
// hidden_dim=2048 -> heads=32, batch=16384/seq_len. BH=batch*heads로 커널에 넘김
// (Q/K/V가 [BH][seq_len][HEAD_DIM]로 flatten되어 있어 batch/heads를 따로 구분하지 않음).
#include "flash_kernels_tc.cuh"
#include <cstdio>
#include <fstream>

constexpr int HEAD_DIM = 64;
constexpr int HEADS = 2048 / HEAD_DIM;  // 32

static int iters_for(int seq_len) {
    if (seq_len <= 1024) return 20;
    if (seq_len <= 4096) return 10;
    return 5;
}

int main() {
    int seq_lens[] = {512, 1024, 2048, 4096, 8192, 16384};
    std::ofstream csv("results/results_tc_paper_grid.csv");
    csv << "seq_len,batch,heads,tc_fwd_ms\n";
    printf("head_dim=%d, heads=%d (hidden_dim=2048)\n", HEAD_DIM, HEADS);
    printf("%8s %6s %6s %10s\n", "seq_len", "batch", "heads", "tc_fwd_ms");

    for (int seq_len : seq_lens) {
        int batch = 16384 / seq_len;
        int BH = batch * HEADS;

        size_t qkv_elems = (size_t)BH * seq_len * HEAD_DIM;
        size_t qkv_bytes = qkv_elems * sizeof(__half);
        size_t l_bytes = (size_t)BH * seq_len * sizeof(float);

        std::vector<float> hQ(qkv_elems), hK(qkv_elems), hV(qkv_elems);
        fill_random(hQ, 1); fill_random(hK, 2); fill_random(hV, 3);
        std::vector<__half> hQh = to_half(hQ), hKh = to_half(hK), hVh = to_half(hV);

        DeviceAllocTracker alloc;
        __half *Q = (__half*)alloc.alloc(qkv_bytes), *K = (__half*)alloc.alloc(qkv_bytes), *V = (__half*)alloc.alloc(qkv_bytes);
        __half *O = (__half*)alloc.alloc(qkv_bytes);
        float *L = (float*)alloc.alloc(l_bytes);

        CUDA_CHECK(cudaMemcpy(Q, hQh.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(K, hKh.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(V, hVh.data(), qkv_bytes, cudaMemcpyHostToDevice));

        auto run = [&]() { flash_forward_launch_tc<HEAD_DIM>(Q, K, V, O, L, BH, seq_len, /*causal=*/false); };
        double ms = time_ms_median(run, 3, iters_for(seq_len));

        printf("%8d %6d %6d %10.4f\n", seq_len, batch, HEADS, ms);
        csv << seq_len << "," << batch << "," << HEADS << "," << ms << "\n";

        alloc.free(Q, qkv_bytes); alloc.free(K, qkv_bytes); alloc.free(V, qkv_bytes);
        alloc.free(O, qkv_bytes); alloc.free(L, l_bytes);
    }
    return 0;
}
