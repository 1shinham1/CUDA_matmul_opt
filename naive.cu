#include <stdio.h>
#include <stdlib.h>

#define M 4096  // A의 행
#define K 4096  // A의 열, B의 행
#define N 4096  // B의 열

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

// CPU에서 정답 계산 (검증용)
void gemm_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++)
                sum += A[i * k + p] * B[p * n + j];
            C[i * n + j] = sum;
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
    dim3 blockDim(32, 32);          // block당 1024개 thread  (1024 스레드/블록)
    dim3 gridDim((N + 31) / 32, (M + 31) / 32);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 워밍업
    gemm_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    gemm_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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
    //printf("C[0] = %f (expected: %f)\n", C[0], (float)K);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return 0;
}

// nvcc -arch=sm_89 matmul_naive.cu -o matmul_naive 이렇게 해야 kernel error가 안남
