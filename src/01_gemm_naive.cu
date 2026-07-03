#include "gemm.h"

// Naive GEMM kernel - thread 하나가 C의 원소 하나 담당
__global__ void gemm_naive(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // C의 행 coalescing 안되게하려고 x사용
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // C의 열

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {          // 내적 계산
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    std::vector<float> A(M * K), B(K * N), C(M * N);
    float *d_A, *d_B, *d_C;

    // 랜덤 초기화
    init_host_matrices(A.data(), B.data(), M, K, N);

    // GPU 메모리 할당
    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);

    // CPU → GPU 복사
    cudaMemcpy(d_A, A.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice);


    // kernel 실행
    dim3 blockDim(32, 32);          // block당 1024개 thread  (1024 스레드/블록)
    dim3 gridDim((N + 31) / 32, (M + 31) / 32);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 워밍업
    for (int i = 0; i < WARM_UP; ++i)
        gemm_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < N_ITERS; ++i)
        gemm_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaDeviceSynchronize();
    
    // GPU → CPU 복사
    cudaMemcpy(C.data(), d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= N_ITERS;

    double flops  = 2.0 * M * N * K;
    double tflops = flops / (ms / 1000.0) / 1e12;

    printf("Time: %.3f ms\n", ms);
    printf("TFLOPS: %.2f\n", tflops);

    // CPU 참조값과 비교
    std::vector<float> C_ref(M * N);
    gemm_cpu_cached(A.data(), B.data(), C_ref.data(), M, K, N);
    verify_against_cpu(C.data(), C_ref.data(), (size_t)M * N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

// nvcc -arch=sm_89 matmul_naive.cu -o matmul_naive 이렇게 해야 kernel error가 안남
