#include "gemm_tc.h"

#define WARPS_PER_BLOCK_M 2
#define WARPS_PER_BLOCK_N 2
#define TILES_PER_WARP_M  2
#define TILES_PER_WARP_N  2
#define WARPS_PER_BLOCK   (WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N)  // 4
#define WARP_TILE_M       (TILES_PER_WARP_M * WMMA_M)              // 32
#define WARP_TILE_N       (TILES_PER_WARP_N * WMMA_N)              // 32
#define BLOCK_TILE_M      (WARPS_PER_BLOCK_M * WARP_TILE_M)        // 64
#define BLOCK_TILE_N      (WARPS_PER_BLOCK_N * WARP_TILE_N)        // 64
#define THREADS_PER_WARP  32

// float1 + __pipeline_memcpy_async
// As, Bs 각 원소를 비동기로 로드 → DMA 엔진이 백그라운드 처리
__device__ __forceinline__ void load_tile_async(
    float As[][WMMA_K], float Bs[][BLOCK_TILE_N],
    float *d_A_ptr, float *d_B_ptr,
    int blockRow, int blockCol, int k0, int K, int N,
    int flatTid, int threadsPerBlock)
{
    const int As_elems = BLOCK_TILE_M * WMMA_K;
    for (int idx = flatTid; idx < As_elems; idx += threadsPerBlock) {
        int r = idx / WMMA_K;
        int c = idx % WMMA_K;
        __pipeline_memcpy_async(
            &As[r][c],
            &d_A_ptr[(blockRow + r) * K + k0 + c],
            sizeof(float));
    }

    const int Bs_elems = WMMA_K * BLOCK_TILE_N;
    for (int idx = flatTid; idx < Bs_elems; idx += threadsPerBlock) {
        int r = idx / BLOCK_TILE_N;
        int c = idx % BLOCK_TILE_N;
        __pipeline_memcpy_async(
            &Bs[r][c],
            &d_B_ptr[(k0 + r) * N + blockCol + c],
            sizeof(float));
    }
}

__global__ void double_buffer_tensor_mat_mul_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr,
                                                    int M, int N, int K)
{
    int warpId  = threadIdx.y;
    int warpRow = warpId / WARPS_PER_BLOCK_N;
    int warpCol = warpId % WARPS_PER_BLOCK_N;

    __shared__ float As[2][BLOCK_TILE_M][WMMA_K];
    __shared__ float Bs[2][WMMA_K][BLOCK_TILE_N];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                            nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a_frag[TILES_PER_WARP_M];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                            nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b_frag[TILES_PER_WARP_N];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[TILES_PER_WARP_M][TILES_PER_WARP_N];

    for (int i = 0; i < TILES_PER_WARP_M; ++i)
        for (int j = 0; j < TILES_PER_WARP_N; ++j)
            nvcuda::wmma::fill_fragment(c_frag[i][j], 0.0f);

    const int threadsPerBlock = THREADS_PER_WARP * WARPS_PER_BLOCK;
    int flatTid   = threadIdx.y * THREADS_PER_WARP + threadIdx.x;
    int blockRow  = blockIdx.x * BLOCK_TILE_M;
    int blockCol  = blockIdx.y * BLOCK_TILE_N;
    int warpRowOffset = warpRow * WARP_TILE_M;
    int warpColOffset = warpCol * WARP_TILE_N;

    // 프롤로그: 버퍼 0 비동기 로드 후 대기
    load_tile_async(As[0], Bs[0], d_A_ptr, d_B_ptr,
                    blockRow, blockCol, 0, K, N, flatTid, threadsPerBlock);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    int curBuf    = 0;
    int numKSteps = K / WMMA_K;

    for (int step = 0; step < numKSteps; ++step) {
        int nextBuf = 1 - curBuf;
        int k0_next = (step + 1) * WMMA_K;

        // 다음 버퍼 비동기 로드 시작 (compute와 오버랩)
        if (step + 1 < numKSteps) {
            load_tile_async(As[nextBuf], Bs[nextBuf], d_A_ptr, d_B_ptr,
                            blockRow, blockCol, k0_next, K, N, flatTid, threadsPerBlock);
            __pipeline_commit();
        }

        // 현재 버퍼로 compute
        for (int i = 0; i < TILES_PER_WARP_M; ++i) {
            float *a_tile_ptr = &As[curBuf][warpRowOffset + i * WMMA_M][0];
            nvcuda::wmma::load_matrix_sync(a_frag[i], a_tile_ptr, WMMA_K);
        }
        for (int j = 0; j < TILES_PER_WARP_N; ++j) {
            float *b_tile_ptr = &Bs[curBuf][0][warpColOffset + j * WMMA_N];
            nvcuda::wmma::load_matrix_sync(b_frag[j], b_tile_ptr, BLOCK_TILE_N);
        }
        for (int i = 0; i < TILES_PER_WARP_M; ++i)
            for (int j = 0; j < TILES_PER_WARP_N; ++j)
                nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);

        // 다음 버퍼 로드 완료 대기 후 swap
        __pipeline_wait_prior(0);
        __syncthreads();
        curBuf = nextBuf;
    }

    for (int i = 0; i < TILES_PER_WARP_M; ++i)
        for (int j = 0; j < TILES_PER_WARP_N; ++j) {
            int cRow = blockRow + warpRowOffset + i * WMMA_M;
            int cCol = blockCol + warpColOffset + j * WMMA_N;
            nvcuda::wmma::store_matrix_sync(&d_C_ptr[cRow * N + cCol],
                                             c_frag[i][j], N, nvcuda::wmma::mem_row_major);
        }
}

void double_buffer_tensor_mat_mul(float *d_A, float *d_B, float *d_C, int M, int N, int K)
{
    dim3 dim_block(THREADS_PER_WARP, WARPS_PER_BLOCK);
    dim3 dim_grid(M / BLOCK_TILE_M, N / BLOCK_TILE_N);
    double_buffer_tensor_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int K = (argc > 2) ? atoi(argv[2]) : M;
    int N = (argc > 3) ? atoi(argv[3]) : M;

    int M_pad = round_up_multiple(M, BLOCK_TILE_M);
    int K_pad = round_up_multiple(K, WMMA_K);
    int N_pad = round_up_multiple(N, BLOCK_TILE_N);

    printf("[TF32 Double Buffer] WARPS=%dx%d, TILES_PER_WARP=%dx%d, BLOCK_TILE=%dx%d\n",
           WARPS_PER_BLOCK_M, WARPS_PER_BLOCK_N, TILES_PER_WARP_M, TILES_PER_WARP_N,
           BLOCK_TILE_M, BLOCK_TILE_N);
    printf("Matrix size (padded): M=%d, K=%d, N=%d\n", M_pad, K_pad, N_pad);

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

    for (int i = 0; i < WARM_UP; ++i)
        double_buffer_tensor_mat_mul(d_A, d_B, d_C, M_pad, N_pad, K_pad);
    CUDA_CHECK(cudaDeviceSynchronize());


    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < N_ITERS; ++i)
        double_buffer_tensor_mat_mul(d_A, d_B, d_C, M_pad, N_pad, K_pad);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= N_ITERS;

    double gflops = compute_gflops(M_pad, N_pad, K_pad, ms);
    printf("[Double Buffer WMMA TF32] time = %.4f ms  |  GFLOPS = %.2f\n", ms, gflops);

    run_cublas_and_verify(d_A, d_B, d_C, d_C_cublas, M_pad, K_pad, N_pad, gflops, N_ITERS);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C)); CUDA_CHECK(cudaFree(d_C_cublas));
    return 0;
}
