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
    int innerRowA = threadIdx.x / (BK / 4);   // 0~3
    int innerColA = threadIdx.x % (BK / 4);   // 0~3 (float4 단위)

    int innerRowB = threadIdx.x / (BN / 4);   // 0~3
    int innerColB = threadIdx.x % (BN / 4);   // 0~15 (float4 단위)

    for (int BK_way_Idx = 0; BK_way_Idx < k; BK_way_Idx += BK) {
        // As 로드: float4로 읽고 전치해서 저장
        for (int loadOffset = 0; loadOffset < BM; loadOffset += NUM_THREADS / (BK / 4)) {
            float4 tmp = reinterpret_cast<float4*>(&A[(innerRowA + loadOffset) * k + innerColA * 4])[0];
            // 전치 저장: As[col][row]
            As[(innerColA * 4 + 0) * BM + innerRowA + loadOffset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + loadOffset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + loadOffset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + loadOffset] = tmp.w;
        }
        // Bs 로드: float4로 읽고 그대로 저장 (이미 연속)
        for (int loadOffset = 0; loadOffset < BK; loadOffset += NUM_THREADS / (BN / 4)) {
            reinterpret_cast<float4*>(&Bs[(innerRowB + loadOffset) * BN + innerColB * 4])[0]
                = reinterpret_cast<float4*>(&B[(innerRowB + loadOffset) * n + innerColB * 4])[0];
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

    dim3 blockDim((BM / TR) * (BN / TC)); // (64/8)*(64/8) = 64 thread
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM); //64 x 64 size

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 워밍업
    gemm_vectorize<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    gemm_vectorize<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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
    printf("C[0] = %f (expected: %f)\n", C[0], (float)K);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return 0;
}


//지금까지 우리는 Block tiling + micro tiling(register tiling)을 했는 데 4090 의 SMEM크기의 한계는. 48KB이나 지금 코드는. 8KB만 사용하며 최대효율을 뽑지 못했다 그렇기에
//1. BK가 4의 배수                (float4 로드)
//2. NUM_THREADS가 BK/4의 배수    (루프 나누어떨어짐)
//3. NUM_THREADS가 32의 배수      (warp 단위 맞춤)
//와 같은 조건을 만족하는 가장 큰 타일을 만들수 있는 (BM,BN,BK)를 이론적으로 구해서 적용해보려한다.ㄴ