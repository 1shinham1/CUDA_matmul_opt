#include <stdio.h>
#include <stdlib.h>

#define M 4096
#define K 4096
#define N 4096

#define BM 128
#define BN 128
#define BK 32

#define TR 8
#define TC 8
#define NUM_THREADS ((BM/TR) * (BN/TC))  // 256

__global__ void gemm_vectorize(float *A, float *B, float *C, int m, int k, int n) {
    int cRow = blockIdx.x;
    int cCol = blockIdx.y;

    int threadRow = threadIdx.x / (BN / TC);
    int threadCol = threadIdx.x % (BN / TC);

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    float threadResults[TR * TC] = {0.0f};

    // float4 로드를 위한 인덱스 (4개씩 묶어서)
    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);

    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);

    for (int BK_way_Idx = 0; BK_way_Idx < k; BK_way_Idx += BK) {
        // As 로드: float4로 읽고 전치해서 저장
        for (int loadOffset = 0; loadOffset < BM; loadOffset += NUM_THREADS / (BK / 4)) {
            float4 tmp = reinterpret_cast<float4*>(
                &A[(innerRowA + loadOffset) * k + innerColA * 4])[0];
            // 전치 저장: As[col][row]
            As[(innerColA * 4 + 0) * BM + innerRowA + loadOffset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + loadOffset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + loadOffset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + loadOffset] = tmp.w;
        }
        // Bs 로드: float4로 읽고 그대로 저장 (이미 연속)
        for (int loadOffset = 0; loadOffset < BK; loadOffset += NUM_THREADS / (BN / 4)) {
            reinterpret_cast<float4*>(
                &Bs[(innerRowB + loadOffset) * BN + innerColB * 4])[0]
                = reinterpret_cast<float4*>(
                &B[(innerRowB + loadOffset) * n + innerColB * 4])[0];
        }

        __syncthreads();

        A += BK;
        B += BK * n;

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float Atmp[TR], Btmp[TC];

            for (int i = 0; i < TR; i++)
                Atmp[i] = As[dotIdx * BM + threadRow * TR + i];  // i에 관해서 연속
                //vectorize 전 코드: Atmp[i] = As[(threadRow * TR + i) * BK + dotIdx];
            for (int j = 0; j < TC; j++)
                Btmp[j] = Bs[dotIdx * BN + threadCol * TC + j];

            for (int i = 0; i < TR; i++)
                for (int j = 0; j < TC; j++)
                    threadResults[i * TC + j] += Atmp[i] * Btmp[j];
        }

        __syncthreads();
    }

    // C 저장: float4로 벡터화하여 메모리 대역폭을 더 효율적으로 사용
    for (int i = 0; i < TR; i++) {
        for (int j = 0; j < TC; j += 4) {
            reinterpret_cast<float4*>(
                &C[(threadRow * TR + i) * n + threadCol * TC + j])[0]
                = {threadResults[i * TC + j],
                   threadResults[i * TC + j + 1],
                   threadResults[i * TC + j + 2],
                   threadResults[i * TC + j + 3]};
        }
    }
    /* float1씩 로드해서 계산했을때
    for (int i = 0; i < TR; i++)
        for (int j = 0; j < TC; j++)
            C[(threadRow * TR + i) * n + threadCol * TC + j] = threadResults[i * TC + j];
    */
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

    dim3 blockDim((BM / TR) * (BN / TC)); // (128/8)*(128/8) = 256 thread
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM); //32 x 32 size

    gemm_vectorize<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    //검증ㅇ용
    printf("C[0] = %f (expected: %f)\n", C[0], (float)K);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return 0;
}


//지금까지 우리는 Block tiling + micro tiling(register tiling)을 하고 warp을 따로 지정해준적이 없었다. 그래서 서로 다른 warp이 같은 SMEM을 읽는 (bank conflict가 발생)
// 또 지금까지의 코드는 thread block이 8x8 64개 인데 warp 1개당 thread가 32개로 2개의 warp만 사용하였다.
//