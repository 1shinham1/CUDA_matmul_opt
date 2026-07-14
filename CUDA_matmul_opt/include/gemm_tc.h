#pragma once

#include <cstdio> //printf
#include <cstdlib> //atoi, exit
#include <cmath> //sqrt
#include <vector> //std::vector
#include <random> // 랜덤 데이터 생성
#include <mma.h> //WMMA API
#include <cublas_v2.h> //cuBLAS
#include <cuda_pipeline.h> //__pipeline_memcpy_async()

#define WARM_UP 5
#define N_ITERS  30
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
    double gflops_kernel, int n_iters = N_ITERS)
{
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;

    // warmup
    for (int i = 0; i < WARM_UP; ++i) {
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, N_pad, M_pad, K_pad,
            &alpha, d_B, CUDA_R_32F, N_pad, d_A, CUDA_R_32F, K_pad,
            &beta,  d_C_cublas, CUDA_R_32F, N_pad,
            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < N_ITERS; ++i) {
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
    ms /= N_ITERS;

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
// gemm.h의 init_host_matrices와 동일한 seed/순서로 채워지므로 (패딩이 없는
// 경우) CUDA Core 쪽에서 만든 캐시를 그대로 재사용할 수 있다.
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
// TC 커널은 TF32(가수 10비트)로 truncate 후 연산하므로 FP32 CPU 참조값과
// 비교하면 truncation 자체의 오차가 섞여 CUDA Core보다 임계값을 느슨하게 잡는다.
inline void verify_against_cpu(const float *C, const float *C_ref, size_t size) {
    double diff_norm = 0.0, ref_norm = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double d = (double)C[i] - (double)C_ref[i];
        diff_norm += d * d;
        ref_norm  += (double)C_ref[i] * (double)C_ref[i];
    }
    double rel_err = std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-12);
    printf("Relative error vs CPU: %.6e %s\n", rel_err,
           (rel_err < 5e-2) ? "(OK)" : "(WARNING: 오차가 큼)");
}
