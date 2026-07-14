#include "gemm.h"
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
    std::vector<float> A(M * K), B(K * N), C(M * N);
    float *d_A, *d_B, *d_C, *d_C_cublas;

    // 랜덤 초기화
    init_host_matrices(A.data(), B.data(), M, K, N);

    // GPU 메모리 할당
    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);
    cudaMalloc((void**)&d_C_cublas, sizeof(float) * M * N);

    // CPU → GPU 복사
    cudaMemcpy(d_A, A.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice);

    // kernel 실행
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE); //
    dim3 gridDim((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 워밍업
    for (int i = 0; i < WARM_UP; ++i)
        gemm_smem<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < N_ITERS; ++i)
        gemm_smem<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaDeviceSynchronize();
    
    // GPU → CPU 복사
    cudaMemcpy(C.data(), d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= N_ITERS;

    double flops  = 2.0 * M * N * K;
    double tflops = flops / (ms / 1000.0) / 1e12;

    printf("Time: %.3f ms\n", ms);
    printf("TFLOPS: %.2f\n", tflops);

    run_cublas_fp32_and_verify(d_A, d_B, d_C_cublas, M, K, N, tflops * 1000.0);

    // CPU 참조값과 비교
    std::vector<float> C_ref(M * N);
    gemm_cpu_cached(A.data(), B.data(), C_ref.data(), M, K, N);
    verify_against_cpu(C.data(), C_ref.data(), (size_t)M * N);

    // 메모리 해제
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_cublas);
    return 0;
}

// nvcc -arch=sm_89 matmul_native.cu -o matmul_naive 이렇게 해야 kernel error가 안남