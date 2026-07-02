#include "gemm_tc.h"

__global__ void naive_tensor_mat_mul_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols)
{
    int warpM = blockIdx.x;
    int warpN = blockIdx.y;

    // Fragment 선언: 데이터 타입은 float 메모리를 가리키지만, 내부 정밀도는
    // nvcuda::wmma::precision::tf32 로 지정 -> 로드/연산 시 TF32로 처리됨
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                            nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a_frag; // TF32 정밀도로 로드
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                            nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b_frag; // TF32 정밀도로 로드
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag; 

    //c_frag 초기화
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < A_n_cols; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        nvcuda::wmma::load_matrix_sync(a_frag, &d_A_ptr[aRow * A_n_cols + aCol], A_n_cols);
        nvcuda::wmma::load_matrix_sync(b_frag, &d_B_ptr[bRow * C_n_cols + bCol], C_n_cols);

        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); //c_frag = a_frag · b_frag + c_frag 누적
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    nvcuda::wmma::store_matrix_sync(&d_C_ptr[cRow * C_n_cols + cCol], c_frag, C_n_cols, nvcuda::wmma::mem_row_major); //완성된 c_frag를 global memory에 씀
}

// ----------------------------------------------------------------------------
// 호스트 측 wrapper
// ----------------------------------------------------------------------------
void naive_tensor_mat_mul(float *d_A, float *d_B, float *d_C, int M, int N, int K)
{
    dim3 dim_block(32, 1); // 블록당 32 threads = 1 warp
    dim3 dim_grid(M / WMMA_M, N / WMMA_N);

    naive_tensor_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int K = (argc > 2) ? atoi(argv[2]) : M;
    int N = (argc > 3) ? atoi(argv[3]) : M;

    int M_pad = round_up_multiple(M, WMMA_M);
    int K_pad = round_up_multiple(K, WMMA_K);
    int N_pad = round_up_multiple(N, WMMA_N);

    printf("[TF32 Naive] Matrix size (padded): M=%d, K=%d, N=%d\n", M_pad, K_pad, N_pad);

    std::vector<float> h_A, h_B;
    init_host_matrices(h_A, h_B, M, K, N, M_pad, K_pad, N_pad);

    float *d_A, *d_B, *d_C, *d_C_cublas;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * M_pad * K_pad));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * K_pad * N_pad));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * M_pad * N_pad));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, sizeof(float) * M_pad * N_pad));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * M_pad * K_pad, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(float) * K_pad * N_pad, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float) * M_pad * N_pad));
    CUDA_CHECK(cudaMemset(d_C_cublas, 0, sizeof(float) * M_pad * N_pad));

    // warmup + 타이밍
    for (int i = 0; i < WARM_UP; ++i)
        naive_tensor_mat_mul(d_A, d_B, d_C, M_pad, N_pad, K_pad);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < N_ITERS; ++i)
        naive_tensor_mat_mul(d_A, d_B, d_C, M_pad, N_pad, K_pad);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= N_ITERS;

    double gflops = compute_gflops(M_pad, N_pad, K_pad, ms);
    printf("[Naive WMMA TF32]      time = %.4f ms  |  GFLOPS = %.2f\n", ms, gflops);

    run_cublas_and_verify(d_A, d_B, d_C, d_C_cublas, M_pad, K_pad, N_pad, gflops, N_ITERS);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C)); CUDA_CHECK(cudaFree(d_C_cublas));
    return 0;
}
