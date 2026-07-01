#include "gemm.h"



#define BM 128
#define BN 128
#define BK 16
#define WM 64
#define WN 64
#define WNITER 1
#define TM 4
#define TN 4
#define WMITER ((WM * WN) / (32 * TM * TN * WNITER))  // 2
#define WSUBM (WM / WMITER)   // 32
#define WSUBN (WN / WNITER)   // 16
#define NUM_THREADS ((BM / WM) * (BN / WN) * 32)       // 256

__global__ void gemm_doublebuffer(float *A, float *B, float *C, int m, int k, int n) {
    int cRow = blockIdx.x;
    int cCol = blockIdx.y;

    int warpId  = threadIdx.x / 32;
    int warpRow = warpId / (BN / WN);
    int warpCol = warpId % (BN / WN);

    int threadIdInWarp  = threadIdx.x % 32;
    int threadColInWarp = threadIdInWarp % (WSUBN / TN);
    int threadRowInWarp = threadIdInWarp / (WSUBN / TN);

    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    A += cRow * BM * k;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * n + cCol * BN + warpCol * WN;

    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    int innerRowA  = threadIdx.x / (BK / 4);
    int innerColA  = threadIdx.x % (BK / 4);
    int innerRowB  = threadIdx.x / (BN / 4);
    int innerColB  = threadIdx.x % (BN / 4);
    int rowStrideA = (NUM_THREADS * 4) / BK;  // 64
    int rowStrideB = NUM_THREADS / (BN / 4);  // 8

    int current = 0;
    int next    = 1;

    // ---------------------------------------------------------------
    // Prologue: current(0) 버퍼에 첫 번째 청크 로드
    // ---------------------------------------------------------------

    // As: 전치 저장 (블로킹)
    for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        float4 tmp = reinterpret_cast<float4*>(
            &A[(innerRowA + offset) * k + innerColA * 4])[0];
        As[current][(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
        As[current][(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
        As[current][(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
        As[current][(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    // Bs: 비동기 로드
    for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        __pipeline_memcpy_async(
            &Bs[current][(innerRowB + offset) * BN + innerColB * 4],
            &B[(innerRowB + offset) * n + innerColB * 4],
            sizeof(float4)
        );
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    A += BK;
    B += BK * n;

    // ---------------------------------------------------------------
    // Main loop
    // ---------------------------------------------------------------
    for (int bkIdx = BK; bkIdx < k; bkIdx += BK) {

        // [1] next 버퍼에 다음 청크 비동기 로드
        for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<float4*>(
                &A[(innerRowA + offset) * k + innerColA * 4])[0];
            As[next][(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[next][(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[next][(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[next][(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            __pipeline_memcpy_async(
                &Bs[next][(innerRowB + offset) * BN + innerColB * 4],
                &B[(innerRowB + offset) * n + innerColB * 4],
                sizeof(float4)
            );
        }
        __pipeline_commit();

        // [2] wait_prior(1): 이전 commit(current)은 이미 완료, next DMA는 진행 중
        __pipeline_wait_prior(1);

        A += BK;
        B += BK * n;

        // [3] current 버퍼로 연산 (next DMA와 오버랩)
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    regM[wSubRowIdx * TM + i] =
                        As[current][dotIdx * BM + warpRow * WM + wSubRowIdx * WSUBM
                           + threadRowInWarp * TM + i];
                }
            }
            #pragma unroll
            for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    regN[wSubColIdx * TN + j] =
                        Bs[current][dotIdx * BN + warpCol * WN + wSubColIdx * WSUBN
                           + threadColInWarp * TN + j];
                }
            }
            #pragma unroll
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
                #pragma unroll
                for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                    #pragma unroll
                    for (int i = 0; i < TM; i++) {
                        #pragma unroll
                        for (int j = 0; j < TN; j++) {
                            threadResults[(wSubRowIdx * TM + i) * (WNITER * TN)
                                         + wSubColIdx * TN + j]
                                += regM[wSubRowIdx * TM + i] * regN[wSubColIdx * TN + j];
                        }
                    }
                }
            }
        }

        // [4] next DMA 완료 보장 후 버퍼 교체
        __pipeline_wait_prior(0);
        __syncthreads();

        current ^= 1;
        next    ^= 1;
    }

    // ---------------------------------------------------------------
    // Epilogue: 마지막 버퍼 연산
    // ---------------------------------------------------------------
    #pragma unroll
    for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
        #pragma unroll
        for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                regM[wSubRowIdx * TM + i] =
                    As[current][dotIdx * BM + warpRow * WM + wSubRowIdx * WSUBM
                       + threadRowInWarp * TM + i];
            }
        }
        #pragma unroll
        for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                regN[wSubColIdx * TN + j] =
                    Bs[current][dotIdx * BN + warpCol * WN + wSubColIdx * WSUBN
                       + threadColInWarp * TN + j];
            }
        }
        #pragma unroll
        for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
            #pragma unroll
            for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    #pragma unroll
                    for (int j = 0; j < TN; j++) {
                        threadResults[(wSubRowIdx * TM + i) * (WNITER * TN)
                                     + wSubColIdx * TN + j]
                            += regM[wSubRowIdx * TM + i] * regN[wSubColIdx * TN + j];
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // C 저장: float4 벡터 스토어
    // ---------------------------------------------------------------
    #pragma unroll
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
        #pragma unroll
        for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
            float *C_interim = C + (wSubRowIdx * WSUBM) * n + wSubColIdx * WSUBN;
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
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
    gemm_doublebuffer<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    gemm_doublebuffer<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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
