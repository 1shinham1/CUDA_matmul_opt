#include "gemm_tc.h"

// ── Shared memory XOR swizzle + 수동 fragment 로드 ─────────────────────────
//
// 확인된 매핑 (lane=threadIdx.x, g=lane>>2, t=lane&3):
//   matrix_a (16x8 row_major):  x0=(g,t)   x1=(g+8,t)   x2=(g,t+4)   x3=(g+8,t+4)
//   matrix_b (8x16 row_major):  x0=(t,g)   x1=(t+4,g)   x2=(t,g+8)   x3=(t+4,g+8)

#define WARPS_PER_BLOCK_M 2
#define WARPS_PER_BLOCK_N 2
#define TILES_PER_WARP_M  4
#define TILES_PER_WARP_N  4
#define WARPS_PER_BLOCK   (WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N)  // 4
#define WARP_TILE_M       (TILES_PER_WARP_M * WMMA_M)              // 64
#define WARP_TILE_N       (TILES_PER_WARP_N * WMMA_N)              // 64
#define BLOCK_TILE_M      (WARPS_PER_BLOCK_M * WARP_TILE_M)        // 128
#define BLOCK_TILE_N      (WARPS_PER_BLOCK_N * WARP_TILE_N)        // 128
#define THREADS_PER_WARP  32

// As[BLOCK_TILE_M][WMMA_K=8]: row stride 8 -> 4행마다 같은 bank window에
// 겹침(row, row+4, row+8, row+12가 전부 같은 window). 실제 fragment 로드는
// 한 m-tile 안에서 g=0..7(=row..row+7) 범위가 한 번에 접근되는데, 그 안의
// 충돌쌍은 (g,g+4) 뿐이다. (row>>2)&1 로 이 두 그룹을 구분해 4(=bit2)만큼
// 확실히 다른 반쪽 window로 옮겨야 실제로 풀린다.
__device__ __forceinline__ int swizzle_a(int row, int col) {
    return col ^ (((row >> 2) & 1) << 2);
}
// Bs[WMMA_K=8][BLOCK_TILE_N]: row stride가 32의 배수라 모든 row가 같은
// bank window. 한 fragment 로드에서 row 4개(t=0~3 또는 4~7)가 동시에 접근
//되는데, 32열 band 안에서 row마다 8칸씩 순환 이동(row*8 mod 32)시키면
// 4개 row가 32뱅크를 8칸씩 정확히 나눠 가져서 겹치지 않는다. (XOR로는
// row/col 조합에 따라 우연히 같은 뱅크로 되돌아가는 경우가 남아있었음)
__device__ __forceinline__ int swizzle_b(int row, int col) {
    int band = col & ~31, within = col & 31;
    return band + (within + row * 8) % 32;
}

// global -> shared 비동기 로드 (scalar, swizzle 적용). swizzle이 4개 원소
// 단위 순서를 바꿔놓기 때문에(특히 As의 XOR) float4 벡터화 스토어는 순서가
// 틀어져서 못 쓰고, __pipeline_memcpy_async를 4byte 단위로 걸어 overlap
__device__ __forceinline__ void load_tile_async_swizzled(
    float As[][WMMA_K], float Bs[][BLOCK_TILE_N],
    float *d_A_ptr, float *d_B_ptr,
    int blockRow, int blockCol, int k0, int K, int N,
    int flatTid, int threadsPerBlock)
{
    const int As_elems = BLOCK_TILE_M * WMMA_K;
    for (int idx = flatTid; idx < As_elems; idx += threadsPerBlock) {
        int r = idx / WMMA_K, c = idx % WMMA_K;
        __pipeline_memcpy_async(&As[r][swizzle_a(r, c)],
                                 &d_A_ptr[(blockRow + r) * K + k0 + c], sizeof(float));
    }
    const int Bs_elems = WMMA_K * BLOCK_TILE_N;
    for (int idx = flatTid; idx < Bs_elems; idx += threadsPerBlock) {
        int r = idx / BLOCK_TILE_N, c = idx % BLOCK_TILE_N;
        __pipeline_memcpy_async(&Bs[r][swizzle_b(r, c)],
                                 &d_B_ptr[(k0 + r) * N + blockCol + c], sizeof(float));
    }
}

__global__ void tc_swizzle_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr,
                                   int M, int N, int K)
{
    int warpId  = threadIdx.y;
    int warpRow = warpId / WARPS_PER_BLOCK_N;
    int warpCol = warpId % WARPS_PER_BLOCK_N;
    int lane    = threadIdx.x;
    int g = lane >> 2, t = lane & 3;

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
    int flatTid  = threadIdx.y * THREADS_PER_WARP + threadIdx.x;
    int blockRow = blockIdx.x * BLOCK_TILE_M;
    int blockCol = blockIdx.y * BLOCK_TILE_N;
    int warpRowOffset = warpRow * WARP_TILE_M;
    int warpColOffset = warpCol * WARP_TILE_N;

    // 프롤로그: 버퍼 0 비동기 로드 후 대기
    load_tile_async_swizzled(As[0], Bs[0], d_A_ptr, d_B_ptr,
                              blockRow, blockCol, 0, K, N, flatTid, threadsPerBlock);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    int curBuf    = 0;
    int numKSteps = K / WMMA_K;

    for (int step = 0; step < numKSteps; ++step) {
        int nextBuf = 1 - curBuf;
        int k0_next = (step + 1) * WMMA_K;

        if (step + 1 < numKSteps) {
            load_tile_async_swizzled(As[nextBuf], Bs[nextBuf], d_A_ptr, d_B_ptr,
                                      blockRow, blockCol, k0_next, K, N, flatTid, threadsPerBlock);
            __pipeline_commit();
        }

        // ---- shared -> fragment (수동, swizzle 적용, curBuf) ----
        for (int i = 0; i < TILES_PER_WARP_M; ++i) {
            int rowBase = warpRowOffset + i * WMMA_M;
            int r0 = rowBase + g, r1 = rowBase + g + 8;
            a_frag[i].x[0] = As[curBuf][r0][swizzle_a(r0, t)];
            a_frag[i].x[1] = As[curBuf][r1][swizzle_a(r1, t)];
            a_frag[i].x[2] = As[curBuf][r0][swizzle_a(r0, t + 4)];
            a_frag[i].x[3] = As[curBuf][r1][swizzle_a(r1, t + 4)];
        }
        for (int j = 0; j < TILES_PER_WARP_N; ++j) {
            int colBase = warpColOffset + j * WMMA_N;
            int c0 = colBase + g, c1 = colBase + g + 8;
            b_frag[j].x[0] = Bs[curBuf][t][swizzle_b(t, c0)];
            b_frag[j].x[1] = Bs[curBuf][t + 4][swizzle_b(t + 4, c0)];
            b_frag[j].x[2] = Bs[curBuf][t][swizzle_b(t, c1)];
            b_frag[j].x[3] = Bs[curBuf][t + 4][swizzle_b(t + 4, c1)];
        }

        for (int i = 0; i < TILES_PER_WARP_M; ++i)
            for (int j = 0; j < TILES_PER_WARP_N; ++j)
                nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);

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

void tc_swizzle(float *d_A, float *d_B, float *d_C, int M, int N, int K)
{
    dim3 dim_block(THREADS_PER_WARP, WARPS_PER_BLOCK);
    dim3 dim_grid(M / BLOCK_TILE_M, N / BLOCK_TILE_N);
    tc_swizzle_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int K = (argc > 2) ? atoi(argv[2]) : M;
    int N = (argc > 3) ? atoi(argv[3]) : M;

    int M_pad = round_up_multiple(M, BLOCK_TILE_M);
    int K_pad = round_up_multiple(K, WMMA_K);
    int N_pad = round_up_multiple(N, BLOCK_TILE_N);

    printf("[TF32 Swizzle (manual fragment x[] + XOR swizzle)] BLOCK_TILE=%dx%d\n",
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
        tc_swizzle(d_A, d_B, d_C, M_pad, N_pad, K_pad);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < N_ITERS; ++i)
        tc_swizzle(d_A, d_B, d_C, M_pad, N_pad, K_pad);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= N_ITERS;

    double gflops = compute_gflops(M_pad, N_pad, K_pad, ms);
    printf("[Swizzle TF32] time = %.4f ms  |  GFLOPS = %.2f\n", ms, gflops);

    run_cublas_and_verify(d_A, d_B, d_C, d_C_cublas, M_pad, K_pad, N_pad, gflops, N_ITERS);

    std::vector<float> h_C(M_pad * N_pad), C_ref(M_pad * N_pad);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeof(float) * M_pad * N_pad, cudaMemcpyDeviceToHost));
    gemm_cpu_cached(h_A.data(), h_B.data(), C_ref.data(), M_pad, K_pad, N_pad);
    verify_against_cpu(h_C.data(), C_ref.data(), (size_t)M_pad * N_pad);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C)); CUDA_CHECK(cudaFree(d_C_cublas));
    return 0;
}
