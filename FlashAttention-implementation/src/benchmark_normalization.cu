// [벤치마크] "정규화를 맨 끝까지 미루는" 최적화(flash_fwd_kernel, 사실상
// FlashAttention-2가 자기 기여로 내세우는 기법 중 하나) vs. "논문 Algorithm
// 1/2를 글자 그대로 따라서 매 K/V 타일마다 정규화해서 HBM에 쓰는"
// flash_fwd_kernel_strict의 forward 실행 시간만 비교한다.
//
// 두 커널은 최종 결과가 수학적으로 완전히 동일하고(test_flash.cu에서 이미
// 검증함), 유일한 차이는 "루프 안에서 매번 나눗셈+HBM 쓰기를 하느냐"뿐이라,
// 이 벤치마크의 시간 차이는 오직 그 최적화 하나의 순수한 효과다.
#include "flash_kernels.cuh"
#include <cstdio>
#include <fstream>
#include <string>

constexpr int HEAD_DIM = 64;
constexpr int FLASH_BLOCK_N = 32;
constexpr int FLASH_BLOCK_M = 64;

static int iters_for(int seq_len) {
    if (seq_len <= 1024) return 20;
    if (seq_len <= 4096) return 10;
    return 5;
}

int main(int argc, char** argv) {
    bool causal = argc > 1 && std::string(argv[1]) == "causal";
    int BH = 32;
    printf("FA_with_cuda: deferred-normalization vs literal-Algorithm-1/2 (BH=%d, head_dim=%d, causal=%d, Q/K/V/O=fp16, compute=fp32)\n\n",
           BH, HEAD_DIM, causal);

    int seq_lens[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    std::ofstream csv(causal ? "results/results_norm_causal.csv" : "results/results_norm_noncausal.csv");
    csv << "seq_len,deferred_ms,strict_ms,slowdown\n";
    printf("%8s | %12s %12s | %10s\n", "seq_len", "deferred(ms)", "strict(ms)", "slowdown");

    for (int seq_len : seq_lens) {
        size_t qkv_elems = (size_t)BH * seq_len * HEAD_DIM;
        size_t qkv_bytes = qkv_elems * sizeof(__half);  // Q,K,V,O는 fp16 저장
        size_t l_bytes = (size_t)BH * seq_len * sizeof(float);  // L은 fp32 통계량

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

        int it = iters_for(seq_len);
        auto run_deferred = [&]() { flash_forward_launch<HEAD_DIM, FLASH_BLOCK_N>(Q, K, V, O, L, BH, seq_len, causal, FLASH_BLOCK_M); };
        auto run_strict = [&]() { flash_forward_launch_strict<HEAD_DIM, FLASH_BLOCK_N>(Q, K, V, O, L, BH, seq_len, causal, FLASH_BLOCK_M); };

        float t_deferred = time_ms_median(run_deferred, 3, it);
        float t_strict = time_ms_median(run_strict, 3, it);

        printf("%8d | %12.4f %12.4f | %9.2fx\n", seq_len, t_deferred, t_strict, t_strict / t_deferred);
        csv << seq_len << "," << t_deferred << "," << t_strict << "," << (t_strict / t_deferred) << "\n";

        alloc.free(Q, qkv_bytes); alloc.free(K, qkv_bytes); alloc.free(V, qkv_bytes);
        alloc.free(O, qkv_bytes); alloc.free(L, l_bytes);
    }
    return 0;
}
