#include "gemm_tc.h"

// 블록당 워프 배치: 2x2 = 4개 워프.
#define WARPS_PER_BLOCK_M 2
#define WARPS_PER_BLOCK_N 2
// 변경 불가 parameter
#define WARPS_PER_BLOCK   (WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N)  // 4
#define BLOCK_TILE_M      (WARPS_PER_BLOCK_M * WMMA_M)             // 32
#define BLOCK_TILE_N      (WARPS_PER_BLOCK_N * WMMA_N)             // 32
#define THREADS_PER_WARP  32

// ============================================================================
// Shared Memory 기반 Tensor Core 행렬곱 커널 (TF32)
//
// 조건:
//  - A, B, C 모두 float(FP32) 메모리, row-major
//  - M, N은 BLOCK_TILE_M / BLOCK_TILE_N의 배수, K는 WMMA_K(8)의 배수로 패딩됨
// ============================================================================

__global__ void shared_mem_tensor_mat_mul_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int M, int N, int K)
{
    int warpId   = threadIdx.y; // 0~ WARPS_PER_BLOCK로  ㅡ매핑
    int warpRow  = warpId / WARPS_PER_BLOCK_N;
    int warpCol  = warpId % WARPS_PER_BLOCK_N;

    int warpM = blockIdx.x * WARPS_PER_BLOCK_M + warpRow;
    int warpN = blockIdx.y * WARPS_PER_BLOCK_N + warpCol;

    // Shared memory: float로 저장 (TF32는 메모리상 FP32 그대로, 연산 시에만 truncate)
    __shared__ float As[BLOCK_TILE_M][WMMA_K];
    __shared__ float Bs[WMMA_K][BLOCK_TILE_N];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                            nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                            nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    const int threadsPerBlock = THREADS_PER_WARP * WARPS_PER_BLOCK;
    int flatTid = threadIdx.y * THREADS_PER_WARP + threadIdx.x;

    int blockRow = blockIdx.x * BLOCK_TILE_M;
    int blockCol = blockIdx.y * BLOCK_TILE_N;

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {

        // ---- global -> shared 로드 ---- CUDA core가 하기 때문에 thread 단위 연산
        const int As_elems = BLOCK_TILE_M * WMMA_K;
        for (int idx = flatTid; idx < As_elems; idx += threadsPerBlock) {
            int r = idx / WMMA_K;
            int c = idx % WMMA_K;
            int globalRow = blockRow + r;
            int globalCol = k0 + c;
            As[r][c] = d_A_ptr[globalRow * K + globalCol];
        }

        const int Bs_elems = WMMA_K * BLOCK_TILE_N;
        for (int idx = flatTid; idx < Bs_elems; idx += threadsPerBlock) {
            int r = idx / BLOCK_TILE_N;
            int c = idx % BLOCK_TILE_N;
            int globalRow = k0 + r;
            int globalCol = blockCol + c;
            Bs[r][c] = d_B_ptr[globalRow * N + globalCol];
        }

        __syncthreads();

        // ---- shared -> fragment 로드 & 누산 ----
        float *a_tile_ptr = &As[warpRow * WMMA_M][0];
        float *b_tile_ptr = &Bs[0][warpCol * WMMA_N];

        nvcuda::wmma::load_matrix_sync(a_frag, a_tile_ptr, WMMA_K); //a_frag 선언할 떄 TF32로 선언해서 다시 명시 필요 없음
        nvcuda::wmma::load_matrix_sync(b_frag, b_tile_ptr, BLOCK_TILE_N);

        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    nvcuda::wmma::store_matrix_sync(&d_C_ptr[cRow * N + cCol], c_frag, N, nvcuda::wmma::mem_row_major);
}

// ============================================================================
// 호스트 wrapper
// ============================================================================
void shared_mem_tensor_mat_mul(float *d_A, float *d_B, float *d_C, int M, int N, int K)
{
    dim3 dim_block(THREADS_PER_WARP, WARPS_PER_BLOCK);
    dim3 dim_grid(M / BLOCK_TILE_M, N / BLOCK_TILE_N);

    shared_mem_tensor_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096; //defalt 4096
    int K = (argc > 2) ? atoi(argv[2]) : M;
    int N = (argc > 3) ? atoi(argv[3]) : M;

    int M_pad = round_up_multiple(M, BLOCK_TILE_M);
    int K_pad = round_up_multiple(K, WMMA_K);
    int N_pad = round_up_multiple(N, BLOCK_TILE_N);

    printf("[TF32 Shared Mem] WARPS_PER_BLOCK=%dx%d, BLOCK_TILE=%dx%d\n",
           WARPS_PER_BLOCK_M, WARPS_PER_BLOCK_N, BLOCK_TILE_M, BLOCK_TILE_N);
    printf("Matrix size (padded): M=%d, K=%d, N=%d\n", M_pad, K_pad, N_pad);

    // ---- 호스트 데이터 (FP32 그대로) ----
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
        shared_mem_tensor_mat_mul(d_A, d_B, d_C, M_pad, N_pad, K_pad);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < N_ITERS; ++i)
        shared_mem_tensor_mat_mul(d_A, d_B, d_C, M_pad, N_pad, K_pad);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= N_ITERS;

    double gflops = compute_gflops(M_pad, N_pad, K_pad, ms);
    printf("[Shared Mem WMMA TF32] time = %.4f ms  |  GFLOPS = %.2f\n", ms, gflops);

    run_cublas_and_verify(d_A, d_B, d_C, d_C_cublas, M_pad, K_pad, N_pad, gflops, N_ITERS);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C)); CUDA_CHECK(cudaFree(d_C_cublas));
    return 0;
}
