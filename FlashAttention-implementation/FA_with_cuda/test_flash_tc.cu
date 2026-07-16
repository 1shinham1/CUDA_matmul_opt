// [정확성 테스트] flash_kernels_tc.cuh의 텐서 코어(WMMA) FlashAttention
// forward+backward 커널이 common.cuh의 fp64 CPU 레퍼런스, 그리고
// flash_kernels.cuh의 (텐서 코어 없는) FMA 커널과 같은 결과를 내는지 확인한다.
#include "flash_kernels.cuh"
#include "flash_kernels_tc.cuh"
#include <cstdio>

template <int HEAD_DIM>
static void run_case(int BH, int seq_len, bool causal) {
    size_t qkv_elems = (size_t)BH * seq_len * HEAD_DIM;
    std::vector<float> hQ(qkv_elems), hK(qkv_elems), hV(qkv_elems), hdO(qkv_elems);
    fill_random(hQ, 1); fill_random(hK, 2); fill_random(hV, 3); fill_random(hdO, 4);
    snap_to_half(hQ); snap_to_half(hK); snap_to_half(hV); snap_to_half(hdO);

    std::vector<float> refO(qkv_elems), refdQ(qkv_elems), refdK(qkv_elems), refdV(qkv_elems);
    ref_forward(hQ, hK, hV, refO, BH, seq_len, HEAD_DIM, causal);
    ref_backward(hQ, hK, hV, hdO, refdQ, refdK, refdV, BH, seq_len, HEAD_DIM, causal);

    std::vector<__half> hQh = to_half(hQ), hKh = to_half(hK), hVh = to_half(hV), hdOh = to_half(hdO);

    DeviceAllocTracker alloc;
    size_t qkv_bytes = qkv_elems * sizeof(__half);
    size_t l_bytes = (size_t)BH * seq_len * sizeof(float);
    __half *Q = (__half*)alloc.alloc(qkv_bytes), *K = (__half*)alloc.alloc(qkv_bytes), *V = (__half*)alloc.alloc(qkv_bytes);
    __half *dO = (__half*)alloc.alloc(qkv_bytes);
    __half *O_tc = (__half*)alloc.alloc(qkv_bytes);
    float *L_tc = (float*)alloc.alloc(l_bytes);
    CUDA_CHECK(cudaMemcpy(Q, hQh.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(K, hKh.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V, hVh.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dO, hdOh.data(), qkv_bytes, cudaMemcpyHostToDevice));

    flash_forward_launch_tc<HEAD_DIM>(Q, K, V, O_tc, L_tc, BH, seq_len, causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> gOh(qkv_elems);
    CUDA_CHECK(cudaMemcpy(gOh.data(), O_tc, qkv_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> gO = to_float(gOh);
    float diffVsRef = max_abs_diff(gO, refO);

    // 같은 입력으로 텐서 코어 없는 flash_fwd_kernel도 돌려서 서로 일치하는지 추가 확인.
    __half *O_fma = (__half*)alloc.alloc(qkv_bytes);
    float *L_fma = (float*)alloc.alloc(l_bytes);
    int block_m = min(64, seq_len);
    flash_forward_launch<HEAD_DIM, 32>(Q, K, V, O_fma, L_fma, BH, seq_len, causal, block_m);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__half> gOfmaH(qkv_elems);
    CUDA_CHECK(cudaMemcpy(gOfmaH.data(), O_fma, qkv_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> gOfma = to_float(gOfmaH);
    float diffVsFma = max_abs_diff(gO, gOfma);

    float tol = 3e-2f;  // 텐서 코어 mma는 fp16 입력을 내부적으로 축소 정밀도로 누적할 수 있어 여유를 조금 더 둠
    bool okFwd = diffVsRef < tol && diffVsFma < tol;
    printf("BH=%d seq_len=%5d d=%3d causal=%d | fwd: TC vs CPU-ref %.6f | TC vs FMA-kernel %.6f  %s\n",
           BH, seq_len, HEAD_DIM, causal, diffVsRef, diffVsFma, okFwd ? "OK" : "FAIL");

    // ----- backward: WMMA 텐서 코어 커널을 CPU 정답, FMA 커널과 비교 -----
    __half *dQ_tc = (__half*)alloc.alloc(qkv_bytes), *dK_tc = (__half*)alloc.alloc(qkv_bytes), *dV_tc = (__half*)alloc.alloc(qkv_bytes);
    flash_backward_launch_tc<HEAD_DIM>(alloc, Q, K, V, O_tc, dO, L_tc, dQ_tc, dK_tc, dV_tc, BH, seq_len, causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__half> gdQh(qkv_elems), gdKh(qkv_elems), gdVh(qkv_elems);
    CUDA_CHECK(cudaMemcpy(gdQh.data(), dQ_tc, qkv_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gdKh.data(), dK_tc, qkv_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gdVh.data(), dV_tc, qkv_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> gdQ = to_float(gdQh), gdK = to_float(gdKh), gdV = to_float(gdVh);

    float dQDiffRef = max_abs_diff(gdQ, refdQ), dKDiffRef = max_abs_diff(gdK, refdK), dVDiffRef = max_abs_diff(gdV, refdV);

    __half *dQ_fma = (__half*)alloc.alloc(qkv_bytes), *dK_fma = (__half*)alloc.alloc(qkv_bytes), *dV_fma = (__half*)alloc.alloc(qkv_bytes);
    flash_backward_launch<HEAD_DIM, 32>(alloc, Q, K, V, O_fma, dO, L_fma, dQ_fma, dK_fma, dV_fma, BH, seq_len, causal, block_m);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__half> gdQfmaH(qkv_elems), gdKfmaH(qkv_elems), gdVfmaH(qkv_elems);
    CUDA_CHECK(cudaMemcpy(gdQfmaH.data(), dQ_fma, qkv_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gdKfmaH.data(), dK_fma, qkv_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gdVfmaH.data(), dV_fma, qkv_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> gdQfma = to_float(gdQfmaH), gdKfma = to_float(gdKfmaH), gdVfma = to_float(gdVfmaH);
    float dQDiffFma = max_abs_diff(gdQ, gdQfma), dKDiffFma = max_abs_diff(gdK, gdKfma), dVDiffFma = max_abs_diff(gdV, gdVfma);

    bool okBwd = dQDiffRef < tol && dKDiffRef < tol && dVDiffRef < tol
              && dQDiffFma < tol && dKDiffFma < tol && dVDiffFma < tol;
    printf("  bwd: dQ ref=%.6f fma=%.6f | dK ref=%.6f fma=%.6f | dV ref=%.6f fma=%.6f  %s\n",
           dQDiffRef, dQDiffFma, dKDiffRef, dKDiffFma, dVDiffRef, dVDiffFma, okBwd ? "OK" : "FAIL");

    alloc.free(Q, qkv_bytes); alloc.free(K, qkv_bytes); alloc.free(V, qkv_bytes); alloc.free(dO, qkv_bytes);
    alloc.free(O_tc, qkv_bytes); alloc.free(L_tc, l_bytes);
    alloc.free(O_fma, qkv_bytes); alloc.free(L_fma, l_bytes);
    alloc.free(dQ_tc, qkv_bytes); alloc.free(dK_tc, qkv_bytes); alloc.free(dV_tc, qkv_bytes);
    alloc.free(dQ_fma, qkv_bytes); alloc.free(dK_fma, qkv_bytes); alloc.free(dV_fma, qkv_bytes);
}

int main() {
    printf("=== FlashAttention WMMA tensor-core fwd+bwd kernels vs. CPU reference + FMA kernel ===\n");
    run_case<64>(2, 64, false);
    run_case<64>(2, 64, true);
    run_case<64>(1, 128, false);
    run_case<64>(2, 130, false);   // seq_len not a multiple of BLOCK_M(64)
    run_case<64>(2, 130, true);
    run_case<64>(1, 1024, false);
    run_case<64>(1, 1024, true);
    return 0;
}
