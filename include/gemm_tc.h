#pragma once

#include <cstdio> //printf
#include <cstdlib> //atoi, exit
#include <cmath> //sqrt
#include <vector> //std::vector
#include <random> // 랜덤 데이터 생성
#include <mma.h> //WMMA API
#include <cublas_v2.h> //cuBLAS
#include <cuda_pipeline.h> //__pipeline_memcpy_async()

// ─── WMMA 타일 크기 (TF32 고정값) ────────────────────────────
// TF32: matrix_a/b = 16x16x8, accumulator = 16x16
static const int WMMA_M = 16; // C, A의 타일 행 수
static const int WMMA_N = 16; // C, B의 타일 열 수
static const int WMMA_K = 8; // A의 타일 열 수 = B의 타일 행 수 (TF32는 8)

// ─── 에러 체크 매크로 ─────────────────────────────────────────
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t st = call;                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error at %s:%d -> status %d\n",          \
                    __FILE__, __LINE__, (int)st);                             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ─── 유틸 함수 ────────────────────────────────────────────────
inline int round_up_multiple(int x, int multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}

inline double compute_gflops(int M, int N, int K, double ms) {
    return 2.0 * (double)M * (double)N * (double)K / (ms / 1000.0) / 1e9;
}

// ─── 호스트 데이터 초기화 ─────────────────────────────────────
// 패딩된 크기로 0 초기화 후 유효 영역만 랜덤으로 채움
inline void init_host_matrices(
    std::vector<float> &h_A, std::vector<float> &h_B,
    int M, int K, int N,
    int M_pad, int K_pad, int N_pad,
    int seed = 42)
{
    h_A.assign(M_pad * K_pad, 0.0f);
    h_B.assign(K_pad * N_pad, 0.0f);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) {
        int r = i / K, c = i % K;
        h_A[r * K_pad + c] = dist(rng);
    }
    for (int i = 0; i < K * N; ++i) {
        int r = i / N, c = i % N;
        h_B[r * N_pad + c] = dist(rng);
    }
}

// ─── cuBLAS TF32 기준값 측정 + 정확도 검증 ───────────────────
inline void run_cublas_and_verify(
    float *d_A, float *d_B, float *d_C, float *d_C_cublas,
    int M_pad, int K_pad, int N_pad,
    double gflops_kernel, int n_iters = 10)
{
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;

    // warmup
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N_pad, M_pad, K_pad,
        &alpha, d_B, CUDA_R_32F, N_pad, d_A, CUDA_R_32F, K_pad,
        &beta,  d_C_cublas, CUDA_R_32F, N_pad,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < n_iters; ++i) {
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, N_pad, M_pad, K_pad,
            &alpha, d_B, CUDA_R_32F, N_pad, d_A, CUDA_R_32F, K_pad,
            &beta,  d_C_cublas, CUDA_R_32F, N_pad,
            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= n_iters;

    double gflops_cublas = compute_gflops(M_pad, N_pad, K_pad, ms);
    printf("[cuBLAS TF32]  time = %.4f ms  |  GFLOPS = %.2f\n", ms, gflops_cublas);
    printf("Efficiency vs cuBLAS: %.1f%%\n", 100.0 * gflops_kernel / gflops_cublas);

    // 정확도 검증
    std::vector<float> h_C(M_pad * N_pad), h_C_ref(M_pad * N_pad);
    CUDA_CHECK(cudaMemcpy(h_C.data(),     d_C,        sizeof(float) * M_pad * N_pad, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_cublas, sizeof(float) * M_pad * N_pad, cudaMemcpyDeviceToHost));

    double diff_norm = 0.0, ref_norm = 0.0;
    for (size_t i = 0; i < h_C.size(); ++i) {
        double d = (double)h_C[i] - (double)h_C_ref[i];
        diff_norm += d * d;
        ref_norm  += (double)h_C_ref[i] * (double)h_C_ref[i];
    }
    double rel_err = std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-12);
    printf("Relative error vs cuBLAS: %.6e %s\n", rel_err,
           (rel_err < 1e-2) ? "(OK)" : "(WARNING: 오차가 큼)");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
