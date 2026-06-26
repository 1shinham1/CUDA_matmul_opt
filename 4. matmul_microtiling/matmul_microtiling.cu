#include <stdio.h>
#include <stdlib.h>

#define M 4096
#define K 4096
#define N 4096

#define BM 64
#define BN 64 // ((BM/TR) × (BN/TC) = 8×8 = 64개) 블록 타일을 더 크게 잡아서 SMEM 재사용을 늘리기 위해서
#define BK 16
// BM, BN -> 클수록 재사용 증가  (but 스레드 수 고정이라 로드 부담 증가)
// BK     -> 클수록 SMEM 증가   (but occupancy 감소)
// tiling에서는 타일의 크기 = thread 수 였어서 As와 Bs의 크기가 같지만 위와 같이 BK는 독립적이기 때문에 작게 잡음
#define TR 8
#define TC 8
#define NUM_THREADS ((BM/TR) * (BN/TC))  // 64

__global__ void gemm_microtiling(float *A, float *B, float *C, int m, int k, int n) {
    int cRow = blockIdx.x;
    int cCol = blockIdx.y;

    int threadRow = threadIdx.x / (BN / TC);
    int threadCol = threadIdx.x % (BN / TC);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    float threadResults[TR * TC] = {0.0f};

    int innerRowA = threadIdx.x / BK;
    int innerColA = threadIdx.x % BK;

    int innerRowB = threadIdx.x / BN;
    int innerColB = threadIdx.x % BN;

    for (int BK_way_Idx = 0; BK_way_Idx < k; BK_way_Idx += BK) {
        // As 로드
        for (int loadOffset = 0; loadOffset < BM; loadOffset += NUM_THREADS / BK) {
            As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * k + innerColA];
        }
        // Bs 로드
        for (int loadOffset = 0; loadOffset < BK; loadOffset += NUM_THREADS / BN) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * n + innerColB];
        }

        __syncthreads();

        A += BK;
        B += BK * n;

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float Atmp[TR], Btmp[TC];

            for (int i = 0; i < TR; i++)
                Atmp[i] = As[(threadRow * TR + i) * BK + dotIdx];
            for (int j = 0; j < TC; j++)
                Btmp[j] = Bs[dotIdx * BN + threadCol * TC + j];

            for (int i = 0; i < TR; i++)
                for (int j = 0; j < TC; j++)
                    threadResults[i * TC + j] += Atmp[i] * Btmp[j];
        }

        __syncthreads();
    }

    for (int i = 0; i < TR; i++)
        for (int j = 0; j < TC; j++)
            C[(threadRow * TR + i) * n + threadCol * TC + j] = threadResults[i * TC + j];
}

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

    dim3 blockDim((BM / TR) * (BN / TC)); // (64/8)*(64/8) = 64 thread
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM); //64 x 64 size

    gemm_microtiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    //검증ㅇ용
    //printf("C[0] = %f (expected: %f)\n", C[0], (float)K);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return 0;
}