#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_pipeline.h>
#include <cublas_v2.h>

// ─── 기본 행렬 크기 ───────────────────────────────────────────
#define M 4096
#define K 4096
#define N 4096

#define WARM_UP 5
#define N_ITERS  30

// ─── 호스트 데이터 랜덤 초기화 ─────────────────────────────────
inline void init_host_matrices(float *A, float *B, int m, int k, int n, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < m * k; ++i) A[i] = dist(rng);
    for (int i = 0; i < k * n; ++i) B[i] = dist(rng);
}

// ─── CPU 참조 계산 (검증용, OpenMP로 병렬화) ─────────────────────
inline void gemm_cpu(const float *A, const float *B, float *C, int m, int k, int n) {
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
                sum += A[i * k + p] * B[p * n + j];
            C[i * n + j] = sum;
        }
    }
}

// ─── CPU 참조값 캐싱 ────────────────────────────────────────────
// A, B가 항상 같은 seed로 초기화되어 매 커널 실행마다 결과가 동일하므로,
// 최초 1회만 계산해 bin/ 아래에 저장하고 이후에는 파일에서 읽어온다.
inline void gemm_cpu_cached(const float *A, const float *B, float *C, int m, int k, int n) {
    char path[256];
    snprintf(path, sizeof(path), "bin/.cpu_ref_%dx%dx%d.bin", m, k, n);

    FILE *f = fopen(path, "rb");
    if (f) {
        size_t got = fread(C, sizeof(float), (size_t)m * n, f);
        fclose(f);
        if (got == (size_t)m * n) {
            printf("[CPU 참조값 캐시 사용: %s]\n", path);
            return;
        }
    }

    gemm_cpu(A, B, C, m, k, n);

    f = fopen(path, "wb");
    if (f) {
        fwrite(C, sizeof(float), (size_t)m * n, f);
        fclose(f);
    }
}

// ─── GPU 결과 vs CPU 참조값 상대 오차 계산 & 출력 ─────────────────
inline void verify_against_cpu(const float *C, const float *C_ref, size_t size) {
    double diff_norm = 0.0, ref_norm = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double d = (double)C[i] - (double)C_ref[i];
        diff_norm += d * d;
        ref_norm  += (double)C_ref[i] * (double)C_ref[i];
    }
    double rel_err = std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-12);
    printf("Relative error vs CPU: %.6e %s\n", rel_err,
           (rel_err < 1e-3) ? "(OK)" : "(WARNING: 오차가 큼)");
}

// ─── cuBLAS FP32 기준값 측정 (같은 프로세스 안에서 커널 직후 측정) ────
// benchmark.sh가 커널마다 별도 프로세스(예: 09번)의 cuBLAS 측정값을 공유해
// 쓰면 GPU 클럭/부스트 편차가 %에 섞인다. 커널 자신의 프로세스 안에서
// 바로 옆에서 cuBLAS를 재서 비교해야 공정하다(TC 커널의 run_cublas_and_verify
// 와 동일한 목적).
inline void run_cublas_fp32_and_verify(float *d_A, float *d_B, float *d_C_cublas,
                                        int m, int k, int n, double gflops_kernel) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f, beta = 0.0f;

    // 워밍업
    for (int i = 0; i < WARM_UP; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                    &alpha, d_B, n, d_A, k, &beta, d_C_cublas, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < N_ITERS; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                    &alpha, d_B, n, d_A, k, &beta, d_C_cublas, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= N_ITERS;

    double gflops_cublas = 2.0 * (double)m * (double)n * (double)k / (ms / 1000.0) / 1e9;
    printf("[cuBLAS FP32]  time = %.4f ms  |  GFLOPS = %.2f\n", ms, gflops_cublas);
    printf("Efficiency vs cuBLAS: %.1f%%\n", 100.0 * gflops_kernel / gflops_cublas);

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
