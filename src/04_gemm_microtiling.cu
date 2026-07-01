#include "gemm.h"

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
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    int threadRow = threadIdx.x / (BN / TC);
    int threadCol = threadIdx.x % (BN / TC);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    float threadResults[TR * TC] = {0.0f};

    int innerRowA = threadIdx.x / BK; //0~3 
    int innerColA = threadIdx.x % BK; //0~15.        그렇기 때문에 As 로드시 백터로 못읽음 ex)thread 15번과 16번

    int innerRowB = threadIdx.x / BN; // 0 1개
    int innerColB = threadIdx.x % BN; //0~63

    for (int BK_way_Idx = 0; BK_way_Idx < k; BK_way_Idx += BK) {
        // As 로드
        for (int loadOffset = 0; loadOffset < BM; loadOffset += NUM_THREADS / BK) { // 4 씩 증가 NUM_THREADS / BK = 4
            As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * k + innerColA];
        }
        // Bs 로드
        for (int loadOffset = 0; loadOffset < BK; loadOffset += NUM_THREADS / BN) { // 1씩 증가 NUM_THREADS / BN = 1
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * n + innerColB];
        }

        __syncthreads();

        A += BK;
        B += BK * n;

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float Atmp[TR], Btmp[TC];

            for (int i = 0; i < TR; i++)
                Atmp[i] = As[(threadRow * TR + i) * BK + dotIdx]; // i 가 증가하면서 As가 연속이 아니고 BK(16)만큼 뛰기 때문에 전치를 시켜서 연속으로 만들어 연속으로 읽고 백터로 만들어 16byte씩 읽어온다
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


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 워밍업
    gemm_microtiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    gemm_microtiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaDeviceSynchronize();

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