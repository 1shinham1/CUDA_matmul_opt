// [정확성 테스트] naive_kernels.cuh의 GPU 커널(Q,K,V,O,dQ,dK,dV,dO는 fp16,
// 나머지 계산은 fp32) 결과가 common.cuh의 double precision CPU 레퍼런스와
// (반올림 오차 수준까지) 일치하는지 확인한다. causal/non-causal, 그리고
// seq_len이 블록 크기로 안 나누어떨어지는 경우(경계 처리 검증용, seq_len=130)
// 까지 포함해서 여러 케이스를 돌린다.
#include "naive_kernels.cuh"
#include <cstdio>

// 케이스 하나(배치*헤드=BH, 시퀀스길이=seq_len, head_dim=d, causal 여부)를 돌려서
// GPU forward+backward 결과를 CPU 정답과 비교하고 결과를 한 줄 출력한다.
static void run_case(int BH, int seq_len, int d, bool causal) {
    size_t qkv_elems = (size_t)BH * seq_len * d;
    // 입력(Q,K,V)과 backward에 필요한 출력 그래디언트(dO)를 CPU 쪽에서 fp32 랜덤 생성한 뒤,
    // fp16으로 반올림했을 때의 값으로 스냅한다 -- GPU가 fp16으로 저장하는 순간 어차피
    // 이 값으로 반올림되므로, CPU 정답도 "GPU가 실제로 보는 입력"과 똑같은 값으로 계산해야
    // 순수하게 "계산 방식의 차이"만 비교할 수 있다 (입력값 자체가 미묘하게 다른 잡음 제거).
    std::vector<float> hQ(qkv_elems), hK(qkv_elems), hV(qkv_elems), hdO(qkv_elems);
    fill_random(hQ, 1);
    fill_random(hK, 2);
    fill_random(hV, 3);
    fill_random(hdO, 4);
    snap_to_half(hQ); snap_to_half(hK); snap_to_half(hV); snap_to_half(hdO);

    // ----- CPU에서 "정답" 계산 (fp64, 느리지만 정확함) -----
    std::vector<float> refO(qkv_elems), refdQ(qkv_elems), refdK(qkv_elems), refdV(qkv_elems);
    ref_forward(hQ, hK, hV, refO, BH, seq_len, d, causal);
    ref_backward(hQ, hK, hV, hdO, refdQ, refdK, refdV, BH, seq_len, d, causal);

    // ----- GPU에서 같은 계산 (Q,K,V,O,dQ,dK,dV,dO는 fp16 저장) -----
    std::vector<__half> hQh = to_half(hQ), hKh = to_half(hK), hVh = to_half(hV), hdOh = to_half(hdO);

    DeviceAllocTracker alloc;
    size_t qkv_bytes = qkv_elems * sizeof(__half);
    __half *Q = (__half*)alloc.alloc(qkv_bytes), *K = (__half*)alloc.alloc(qkv_bytes);
    __half *V = (__half*)alloc.alloc(qkv_bytes), *dO = (__half*)alloc.alloc(qkv_bytes);
    __half *O = (__half*)alloc.alloc(qkv_bytes);
    __half *dQ = (__half*)alloc.alloc(qkv_bytes), *dK = (__half*)alloc.alloc(qkv_bytes), *dV = (__half*)alloc.alloc(qkv_bytes);

    // 호스트(CPU)에서 만든 fp16 값을 디바이스(GPU) 메모리로 복사
    CUDA_CHECK(cudaMemcpy(Q, hQh.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(K, hKh.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V, hVh.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dO, hdOh.data(), qkv_bytes, cudaMemcpyHostToDevice));

    // naive_forward/naive_backward는 "편의용 wrapper"라서 S,P,dP,dS 버퍼(fp32)
    // 할당까지 알아서 해준다 (여기서는 반복 측정이 아니라 한 번만 돌리므로
    // 할당 오버헤드가 문제되지 않음. benchmark.cu와 다른 점).
    NaiveForwardBuffers fwd = naive_forward(alloc, Q, K, V, O, BH, seq_len, d, causal);
    naive_backward(alloc, Q, K, V, dO, fwd, dQ, dK, dV, BH, seq_len, d, causal);
    CUDA_CHECK(cudaGetLastError());       // 커널 launch 자체가 실패하지 않았는지 확인
    CUDA_CHECK(cudaDeviceSynchronize());  // 모든 커널이 실제로 끝날 때까지 대기

    // GPU 결과(fp16)를 다시 CPU로 복사해서 fp32로 변환 후 비교
    std::vector<__half> gOh(qkv_elems), gdQh(qkv_elems), gdKh(qkv_elems), gdVh(qkv_elems);
    CUDA_CHECK(cudaMemcpy(gOh.data(), O, qkv_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gdQh.data(), dQ, qkv_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gdKh.data(), dK, qkv_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gdVh.data(), dV, qkv_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> gO = to_float(gOh), gdQ = to_float(gdQh), gdK = to_float(gdKh), gdV = to_float(gdVh);

    // GPU(fp16 저장 + fp32 계산) vs CPU(fp64) 결과의 최대 절댓값 오차.
    // 출력값을 fp16으로 반올림하는 데서 오는 오차가 지배적이라, fp32일 때보다
    // 허용치를 넉넉히 잡는다 (fp16 가수부 10비트 => 상대오차 ~1e-3 수준).
    float dO_diff = max_abs_diff(gO, refO);
    float dQ_diff = max_abs_diff(gdQ, refdQ);
    float dK_diff = max_abs_diff(gdK, refdK);
    float dV_diff = max_abs_diff(gdV, refdV);
    float tol = 2e-2f;
    bool ok = dO_diff < tol && dQ_diff < tol && dK_diff < tol && dV_diff < tol;

    printf("BH=%d seq_len=%5d d=%3d causal=%d | O %.6f dQ %.6f dK %.6f dV %.6f  %s\n",
           BH, seq_len, d, causal, dO_diff, dQ_diff, dK_diff, dV_diff, ok ? "OK" : "FAIL");

    // 다음 케이스를 위해 이번에 쓴 GPU 메모리를 전부 반납
    alloc.free(Q, qkv_bytes); alloc.free(K, qkv_bytes); alloc.free(V, qkv_bytes);
    alloc.free(dO, qkv_bytes); alloc.free(O, qkv_bytes);
    alloc.free(dQ, qkv_bytes); alloc.free(dK, qkv_bytes); alloc.free(dV, qkv_bytes);
    alloc.free(fwd.S, fwd.sp_bytes); alloc.free(fwd.P, fwd.sp_bytes);
}

int main() {
    printf("=== naive CUDA kernels (fp16 storage / fp32 compute) vs. double-precision CPU reference ===\n");
    run_case(2, 64, 64, false);
    run_case(2, 64, 64, true);
    run_case(1, 128, 64, false);
    run_case(2, 130, 64, false);   // seq_len이 블록 크기(16의 배수 등)로 안 나누어떨어지는 경우 -- 경계 마스킹 검증
    run_case(2, 130, 64, true);
    run_case(2, 128, 128, false);  // head_dim이 다른 경우
    run_case(1, 256, 32, false);   // head_dim이 더 작은 경우
    return 0;
}
