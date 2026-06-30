#include <stdio.h>
#include <stdlib.h>

#define M 4096  // A의 행
#define K 4096  // A의 열, B의 행
#define N 4096  // B의 열
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
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    // CPU 메모리 할당
    A     = (float*)malloc(sizeof(float) * M * K);
    B     = (float*)malloc(sizeof(float) * K * N);
    C     = (float*)malloc(sizeof(float) * M * N);

    // 초기화
    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;

    // GPU 메모리 할당
    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);

    // CPU → GPU 복사
    cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    // kernel 실행
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE); //1024 연속된 스레드(warp)가 연속된 메모리 주소에 접근하도록 강제하기 위해서 1D로 만듦
    dim3 gridDim((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 워밍업
    gemm_coalesced<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    gemm_coalesced<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaDeviceSynchronize();
    
    // GPU → CPU 복사
    cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double flops  = 2.0 * M * N * K;
    double tflops = flops / (ms / 1000.0) / 1e12;

    printf("Time: %.3f ms\n", ms);
    printf("TFLOPS: %.2f\n", tflops);
    printf("C[0] = %f (expected: %f)\n", C[0], (float)K);
    
    // 메모리 해제
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return 0;
}

// nvcc -arch=sm_89 matmul_native.cu -o matmul_naive 이렇게 해야 kernel error가 안남