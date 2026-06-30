#include <stdio.h>
#include <stdlib.h>

#define M 4096  // A의 행
#define K 4096  // A의 열, B의 행
#define N 4096  // B의 열
#define BLOCKSIZE 32

__global__ void gemm_smem(float *A, float *B, float *C, int m, int k, int n) {
    // 이 블록이 담당하는 C의 타일 위치
    int cCol = blockIdx.x;
    int cRow = blockIdx.y;

    // 블록 안에서 내 위치 (1D threadIdx → 2D)
    int threadRow = threadIdx.x / BLOCKSIZE;
    int threadCol = threadIdx.x % BLOCKSIZE;

    // Shared MEM 선언
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // 포인터를 내 담당 타일 시작점으로 이동
    A += cRow * BLOCKSIZE * k;   // 내 블록의 row 시작
    B += cCol * BLOCKSIZE;       // 내 블록의 col 시작
    C += cRow * BLOCKSIZE * n + cCol * BLOCKSIZE;

    float sum = 0.0f;

    // K 방향으로 타일 이동
    for (int bkIdx = 0; bkIdx < k; bkIdx += BLOCKSIZE) {
        // GMEM → SMEM 로드 (스레드 하나가 원소 하나씩 담당)
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * k + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * n + threadCol];

        // 모든 스레드가 로드 끝날 때까지 대기
        __syncthreads();

        // 타일 포인터 전진
        A += BLOCKSIZE;
        B += BLOCKSIZE * n;

        // SMEM에서 내적 계산
        for (int i = 0; i < BLOCKSIZE; i++) {
            sum += As[threadRow * BLOCKSIZE + i] * Bs[i * BLOCKSIZE + threadCol];
        }

        // 다음 타일 로드 전 동기화
        // (느린 스레드가 아직 쓰는 SMEM을 빠른 스레드가 덮어쓰면 안됨)
        __syncthreads();
    }

    C[threadRow * n + threadCol] = sum;
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
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE); //1024
    dim3 gridDim((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 워밍업
    gemm_smem<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    gemm_smem<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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
    printf("C[0] = %f (expected: %f)\n", C[0], (float)K);

    // 메모리 해제
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return 0;
}

// nvcc -arch=sm_89 matmul_native.cu -o matmul_naive 이렇게 해야 kernel error가 안남