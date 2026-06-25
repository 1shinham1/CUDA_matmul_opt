#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

#define M 4096
#define K 4096
#define N 4096

int main() {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    A = (float*)malloc(sizeof(float) * M * K);
    B = (float*)malloc(sizeof(float) * K * N);
    C = (float*)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;

    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    // cuBLAS 핸들 생성
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 워밍업
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);
    cudaDeviceSynchronize();

    // 실제 실행
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);
    cudaDeviceSynchronize();

    // 메모리 해제
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return 0;
}

//nvcc -arch=sm_89 matmul_cuBLAS.cu -o matmul_cuBLAS -lcublas 로 컴파일해야함