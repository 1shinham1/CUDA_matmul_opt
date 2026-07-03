#include "gemm.h"
#define BLOCKSIZE 32

__global__ void gemm_coalesced(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE); // treadIdx가 증가할때 row 는 일정
    int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE); // col만 1~31로 증가 즉 가로로 쭉 연속된 메모리를 읽음

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
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
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE); //1024 연속된 스레드(warp)가 연속된 메모리 주소에 접근하도록 강제하기 위해서 1D로 만듦
    dim3 gridDim((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 워밍업
    for (int i = 0; i < WARM_UP; ++i)
        gemm_coalesced<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < N_ITERS; ++i)
        gemm_coalesced<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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

    // 메모리 해제
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

// nvcc -arch=sm_89 matmul_native.cu -o matmul_naive 이렇게 해야 kernel error가 안남