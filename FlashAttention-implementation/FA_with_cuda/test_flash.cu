// [정확성 테스트] flash_kernels.cuh의 FlashAttention 커널(Q,K,V,O,dQ,dK,dV,dO는
// fp16, 나머지 계산은 fp32) 결과가 common.cuh의 double precision CPU
// 레퍼런스와 일치하는지 확인. test_naive.cu와 구조는 거의 동일하고,
// 대상 커널만 flash로 바뀐 것.
#include "flash_kernels.cuh"
#include <cstdio>

// HEAD_DIM은 커널의 템플릿 매개변수라 함수 템플릿으로 받는다
// (run_case<64>(...)처럼 호출 시점에 컴파일타임 값으로 지정).
template <int HEAD_DIM>
static void run_case(int BH, int seq_len, bool causal) {
    const int BLOCK_N = 32;        // K/V 타일 폭 (flash_kernels.cuh 템플릿 매개변수)
    int BLOCK_M = min(64, seq_len);      // Q 타일 폭 (실행 시점 값). seq_len이 64보다 작은 케이스도 있어서 min 처리.

    size_t qkv_elems = (size_t)BH * seq_len * HEAD_DIM;
    std::vector<float> hQ(qkv_elems), hK(qkv_elems), hV(qkv_elems), hdO(qkv_elems);
    fill_random(hQ, 1);
    fill_random(hK, 2);
    fill_random(hV, 3);
    fill_random(hdO, 4);
    // fp16으로 반올림했을 때의 값으로 스냅 -- CPU 정답이 GPU가 실제로 보는
    // 입력과 동일한 값을 쓰도록 맞춰서, "계산 방식의 차이"만 순수하게 비교.
    snap_to_half(hQ); snap_to_half(hK); snap_to_half(hV); snap_to_half(hdO);

    // ----- CPU 정답 -----
    std::vector<float> refO(qkv_elems), refdQ(qkv_elems), refdK(qkv_elems), refdV(qkv_elems);
    ref_forward(hQ, hK, hV, refO, BH, seq_len, HEAD_DIM, causal);
    ref_backward(hQ, hK, hV, hdO, refdQ, refdK, refdV, BH, seq_len, HEAD_DIM, causal);

    // ----- GPU 계산 (Q,K,V,O,dQ,dK,dV,dO는 fp16 저장) -----
    std::vector<__half> hQh = to_half(hQ), hKh = to_half(hK), hVh = to_half(hV), hdOh = to_half(hdO);

    DeviceAllocTracker alloc;
    size_t qkv_bytes = qkv_elems * sizeof(__half);
    size_t l_bytes = (size_t)BH * seq_len * sizeof(float);  // logsumexp L 버퍼 크기 (fp32 통계량, naive의 S,P와 달리 O(seq_len)만 필요!)
    __half *Q = (__half*)alloc.alloc(qkv_bytes), *K = (__half*)alloc.alloc(qkv_bytes), *V = (__half*)alloc.alloc(qkv_bytes);
    __half *O = (__half*)alloc.alloc(qkv_bytes);
    __half *dO = (__half*)alloc.alloc(qkv_bytes);
    __half *dQ = (__half*)alloc.alloc(qkv_bytes), *dK = (__half*)alloc.alloc(qkv_bytes), *dV = (__half*)alloc.alloc(qkv_bytes);
    float *L = (float*)alloc.alloc(l_bytes);

    CUDA_CHECK(cudaMemcpy(Q, hQh.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(K, hKh.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V, hVh.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dO, hdOh.data(), qkv_bytes, cudaMemcpyHostToDevice));

    // forward를 먼저 돌려서 O, L을 채운 다음, 그 결과(O, L)를 backward가 이어받아 사용.
    // (naive_forward가 S,P를 만들어 backward에 넘기는 것과 같은 관계, 다만 여기선 O(seq_len)짜리 L 하나뿐)
    flash_forward_launch<HEAD_DIM, BLOCK_N>(Q, K, V, O, L, BH, seq_len, causal, BLOCK_M);
    flash_backward_launch<HEAD_DIM, BLOCK_N>(alloc, Q, K, V, O, dO, L, dQ, dK, dV, BH, seq_len, causal, BLOCK_M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> gOh(qkv_elems), gdQh(qkv_elems), gdKh(qkv_elems), gdVh(qkv_elems);
    CUDA_CHECK(cudaMemcpy(gOh.data(), O, qkv_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gdQh.data(), dQ, qkv_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gdKh.data(), dK, qkv_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gdVh.data(), dV, qkv_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> gO = to_float(gOh), gdQ = to_float(gdQh), gdK = to_float(gdKh), gdV = to_float(gdVh);

    float oDiff = max_abs_diff(gO, refO);
    float dQDiff = max_abs_diff(gdQ, refdQ);
    float dKDiff = max_abs_diff(gdK, refdK);
    float dVDiff = max_abs_diff(gdV, refdV);
    float tol = 2e-2f;  // fp16 저장에서 오는 반올림 오차를 감안한 허용치 (naive 테스트와 동일)
    bool ok = oDiff < tol && dQDiff < tol && dKDiff < tol && dVDiff < tol;
    printf("BH=%d seq_len=%5d d=%3d causal=%d | O %.6f dQ %.6f dK %.6f dV %.6f  %s\n",
           BH, seq_len, HEAD_DIM, causal, oDiff, dQDiff, dKDiff, dVDiff, ok ? "OK" : "FAIL");

    // ----- Algorithm 1/2를 글자 그대로 재현한 "strict" 커널도 같은 정답과 비교 -----
    // (매 타일마다 정규화해서 HBM에 쓰지만, 최종값은 flash_fwd_kernel과 수학적으로 동일해야 함)
    __half* Ostrict = (__half*)alloc.alloc(qkv_bytes);
    float* Lstrict = (float*)alloc.alloc(l_bytes);
    flash_forward_launch_strict<HEAD_DIM, BLOCK_N>(Q, K, V, Ostrict, Lstrict, BH, seq_len, causal, BLOCK_M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__half> gOstrictH(qkv_elems);
    CUDA_CHECK(cudaMemcpy(gOstrictH.data(), Ostrict, qkv_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> gOstrict = to_float(gOstrictH);
    float oStrictDiff = max_abs_diff(gOstrict, refO);
    printf("  strict(Algorithm 1/2 literal) forward vs same ref: O %.6f  %s\n",
           oStrictDiff, oStrictDiff < tol ? "OK" : "FAIL");
    alloc.free(Ostrict, qkv_bytes);
    alloc.free(Lstrict, l_bytes);

    alloc.free(Q, qkv_bytes); alloc.free(K, qkv_bytes); alloc.free(V, qkv_bytes);
    alloc.free(O, qkv_bytes); alloc.free(dO, qkv_bytes);
    alloc.free(dQ, qkv_bytes); alloc.free(dK, qkv_bytes); alloc.free(dV, qkv_bytes);
    alloc.free(L, l_bytes);
}

int main() {
    printf("=== FlashAttention CUDA forward+backward kernels (fp16 storage / fp32 compute) vs. CPU reference ===\n");
    run_case<64>(2, 64, false);
    run_case<64>(2, 64, true);
    run_case<64>(1, 128, false);
    run_case<64>(2, 130, false);   // seq_len이 블록 크기로 안 나누어떨어지는 경우 -- 경계 마스킹 검증
    run_case<64>(2, 130, true);
    run_case<128>(2, 128, false);  // head_dim=128 -- 48KB 정적 shared memory 한도를 넘어서 동적 shared memory 경로를 검증
    run_case<32>(1, 256, false);   // head_dim=32 -- 더 작은 head_dim
    return 0;
}
