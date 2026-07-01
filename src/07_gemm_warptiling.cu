#include "gemm.h"

#define BM 128
#define BN 128
#define BK 16
#define WM 64
#define WN 32
#define WNITER 2
#define TM 4
#define TN 4
#define WMITER ((WM * WN) / (32 * TM * TN * WNITER))  // 2
#define WSUBM (WM / WMITER)   // 32
#define WSUBN (WN / WNITER)   // 16
#define NUM_THREADS ((BM / WM) * (BN / WN) * 32)       // 256

__global__ void gemm_warptiling(float *A, float *B, float *C, int m, int k, int n) {
    int cRow = blockIdx.x;
    int cCol = blockIdx.y;

    // warp 위치
    int warpId  = threadIdx.x / 32;
    int warpRow = warpId / (BN / WN);   // 0~1
    int warpCol = warpId % (BN / WN);   // 0~3

    // warp 안에서 thread 위치
    int threadIdInWarp  = threadIdx.x % 32;
    int threadColInWarp = threadIdInWarp % (WSUBN / TN);  // 0~3
    int threadRowInWarp = threadIdInWarp / (WSUBN / TN);  // 0~7

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * k;
    B += cCol * BN;
    // C 포인터를 warp 위치로 미리 이동
    C += (cRow * BM + warpRow * WM) * n + cCol * BN + warpCol * WN;

    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    // SMEM 로드 인덱스 (float4)
    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);
    int rowStrideA = (NUM_THREADS * 4) / BK;   // 64
    int rowStrideB = NUM_THREADS / (BN / 4);   // 8

    for (int bkIdx = 0; bkIdx < k; bkIdx += BK) {
        // As 로드: float4로 읽고 전치 저장
        for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<float4*>(
                &A[(innerRowA + offset) * k + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }
        // Bs 로드: float4로 읽고 그대로 저장
        for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4*>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0]
                = reinterpret_cast<float4*>(
                &B[(innerRowB + offset) * n + innerColB * 4])[0];
        }

        __syncthreads();

        A += BK;
        B += BK * n;

        // warptiling 계산
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            // regM 로드
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
                for (int i = 0; i < TM; i++) {
                    regM[wSubRowIdx * TM + i] =
                        As[dotIdx * BM + warpRow * WM + wSubRowIdx * WSUBM
                           + threadRowInWarp * TM + i];
                }
            }
            // regN 로드
            for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                for (int j = 0; j < TN; j++) {
                    regN[wSubColIdx * TN + j] =
                        Bs[dotIdx * BN + warpCol * WN + wSubColIdx * WSUBN
                           + threadColInWarp * TN + j];
                }
            }

            // outer product 누적
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
                for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                    for (int i = 0; i < TM; i++) {
                        for (int j = 0; j < TN; j++) {
                            threadResults[(wSubRowIdx * TM + i) * (WNITER * TN)
                                         + wSubColIdx * TN + j]
                                += regM[wSubRowIdx * TM + i] * regN[wSubColIdx * TN + j];
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    // C 저장
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
            float *C_interim = C + (wSubRowIdx * WSUBM) * n + wSubColIdx * WSUBN;
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j += 4) {
                    reinterpret_cast<float4*>(
                        &C_interim[(threadRowInWarp * TM + i) * n
                                    + threadColInWarp * TN + j])[0]
                        = {threadResults[(wSubRowIdx * TM + i) * (WNITER * TN) + wSubColIdx * TN + j],
                           threadResults[(wSubRowIdx * TM + i) * (WNITER * TN) + wSubColIdx * TN + j + 1],
                           threadResults[(wSubRowIdx * TM + i) * (WNITER * TN) + wSubColIdx * TN + j + 2],
                           threadResults[(wSubRowIdx * TM + i) * (WNITER * TN) + wSubColIdx * TN + j + 3]};
                }
            }
        }
    }
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

    dim3 blockDim(NUM_THREADS);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 워밍업
    gemm_warptiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    gemm_warptiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

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
