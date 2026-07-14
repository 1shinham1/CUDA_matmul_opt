#include "gemm.h"

int main() {
    std::vector<float> A(M * K), B(K * N), C(M * N);
    float *d_A, *d_B, *d_C;

    init_host_matrices(A.data(), B.data(), M, K, N);

    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, A.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta  = 0.0f;


    // Warmup
    for (int i = 0; i < WARM_UP; ++i) {
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
    cudaDeviceSynchronize();

    // 측정
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < N_ITERS; ++i) {
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    float avg_time = ms / N_ITERS;
    float gflops   = (2.0f * M * N * K) / (avg_time * 1e6);

    printf("Average kernel execution time: %.3f ms\n", avg_time);
    printf("GFLOPS: %.2f\n", gflops);

    // GPU → CPU 복사 후 CPU 참조값과 비교
    cudaMemcpy(C.data(), d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    std::vector<float> C_ref(M * N);
    gemm_cpu_cached(A.data(), B.data(), C_ref.data(), M, K, N);
    verify_against_cpu(C.data(), C_ref.data(), (size_t)M * N);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

// nvcc -arch=sm_89 matmul_cuBLAS.cu -o matmul_cuBLAS -lcublas