#include <stdio.h>
#include <stdlib.h>

#define M 1024  // A의 행
#define K 1024  // A의 열, B의 행
#define N 1024  // B의 열

// Naive GEMM kernel - thread 하나가 C의 원소 하나 담당
__global__ void gemm_naive(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // C의 행
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // C의 열

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {          // 내적 계산
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
    dim3 blockDim(16, 16);          // block당 256개 thread
    dim3 gridDim((N + 15) / 16, (M + 15) / 16); // 이 코드에서는 64 X 64 개의 블록 생성
    gemm_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();
    
    // GPU → CPU 복사
    cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);


    // 메모리 해제
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return 0;
}

// nvcc -arch=sm_89 matmul_native.cu -o matmul_naive 이렇게 해야 kernel error가 안남