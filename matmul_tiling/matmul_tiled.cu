#include <stdio.h>
#include <stdlib.h>

#define M 1024
#define K 1024
#define N 1024
#define TILE 16   // tile 크기 (blockDim과 동일)

__global__ void gemm_tiled(float *A, float *B, float *C, int m, int k, int n) {
    // Shared memory에 tile 선언
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // K 방향으로 tile 단위로 순회
    for (int t = 0; t < (k + TILE - 1) / TILE; t++) {

        // 각 스레드가 tileA, tileB 원소 하나씩 로드
        if (row < m && t * TILE + threadIdx.x < k)
            tileA[threadIdx.y][threadIdx.x] = A[row * k + t * TILE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && t * TILE + threadIdx.y < k)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // 모든 스레드가 로드 완료될 때까지 대기
        __syncthreads();

        // Shared memory에서 내적 계산 (DRAM 접근 없음)
        for (int i = 0; i < TILE; i++)
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        // 다음 tile 로드 전 동기화
        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = sum;
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
    float *A, *B, *C, *C_ref;
    float *d_A, *d_B, *d_C;

    A = (float*)malloc(sizeof(float) * M * K);
    B = (float*)malloc(sizeof(float) * K * N);
    C = (float*)malloc(sizeof(float) * M * N);
    C_ref = (float*)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;

    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);


    // 검증 (CPU 결과와 비교)
    gemm_cpu(A, B, C_ref, M, K, N);
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = C[i] - C_ref[i];
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
    }
    printf("최대 오차: %f\n", max_err);
    printf("C[0] = %.1f (기대값: %.1f)\n", C[0], C_ref[0]);
    printf("C[M*N-1] = %.1f (기대값: %.1f)\n", C[M*N-1], C_ref[M*N-1]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C); free(C_ref);
    return 0;
}