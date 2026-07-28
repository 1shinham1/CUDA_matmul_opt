#include "gemm_tc.h"

// ── Shared memory XOR swizzle + 수동 fragment 로드 ─────────────────────────
//
// 확인된 매핑 (lane=threadIdx.x, g=lane>>2, t=lane&3):
//   matrix_a (16x8 row_major):  x0=(g,t)   x1=(g+8,t)   x2=(g,t+4)   x3=(g+8,t+4)
//   matrix_b (8x16 row_major):  x0=(t,g)   x1=(t+4,g)   x2=(t,g+8)   x3=(t+4,g+8)
//   (두 번째 인덱스가 K 방향. BK 안에서 wmma k-step을 kk=0..BK/8-1 로 돈다)
//
// [BK 실험 -- ncu 권고를 따라가 봤지만 기각한 기록]
// ncu가 BK=8에서 "uncoalesced shared accesses, 50,331,648 excessive wavefronts
// (전체의 33%), Est. Speedup 33%"를 지목했다. 수를 세보면 원인이 정확히
// As 스토어다:
//   store 명령 = 8회/thread x 4warp x 512step x 1024block = 16,777,216 (As, Bs 각각)
//   wavefront  = As 4/명령 + Bs 1/명령 -> 83,886,080
//   ncu 정의(wavefronts - requests) = 83,886,080 - 33,554,432 = 50,331,648
// 즉 리포트된 충돌 100%가 As 스토어 하나당 초과 wavefront 3개, Bs는 깨끗했다.
//
// 원인은 shared 레이아웃이 아니라 글로벌 쪽 세그먼트 분할이다. BK=8이면 A 타일
// 한 행이 8 float = 32B뿐이라, warp 32레인이 서로 16KB(K*4) 떨어진 행 4개를
// 걸친다. LDGSTS는 흩어진 소스 세그먼트를 합칠 수 없어 4번으로 쪼개지고, 그
// 조각들이 shared에 32B씩 쓰면서 wavefront가 4배가 된다. BK를 키우면 A 타일
// 한 행이 64B(BK=16)/128B(BK=32)가 되어 이 분할이 사라진다.
//
// 실제로 해봤더니 진단은 맞았지만 성능은 반대로 갔다 (RTX 4090, 4096^3):
//   BK   시간      vs cuBLAS   shared wavefront   occupancy   register
//    8   2.11 ms     74.5%      176,706,395        16.67%       204
//   16   2.27 ms     69.3%      147,828,972        16.67%       186
//   32   2.55 ms     61.3%      125,973,504         8.33%       210
// BK=16은 wavefront를 16% 줄이고 occupancy도 지키고 레지스터까지 줄였는데도
// 8% 느렸다(static/dynamic shared 차이는 아님 -- 둘 다 재봤고 2.27 vs 2.28).
// 이 커널은 Compute 39% / Memory 48% / DRAM 16%로 어느 쪽도 포화가 아니라
// wavefront(처리량 지표)가 병목이 아니었던 것. BK=32는 거기에 shared 64KB로
// SM당 블록이 2->1이 되며 occupancy까지 반토막 났다.
//
// 결론: BK=8 유지. 진짜 제약은 레지스터 204개가 SM당 블록을 2개로 묶는 것이고
// (ncu도 "occupancy 16.7%"를 최상위 권고로 지목), 3블록이 되려면 <=170까지
// 줄여야 한다. 아래 BK는 8/16/32를 그대로 바꿔 끼워 재현할 수 있다.
#define BK                8                               // K 방향 타일 (WMMA_K의 배수)
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
#define K_STEPS_PER_TILE  (BK / WMMA_K)                            // BK=8이면 1
// 4 = sizeof(float). #if 에서 평가돼야 해서 sizeof 대신 리터럴을 쓴다.
#define SMEM_BYTES        (4 * 2 * (BLOCK_TILE_M * BK + BK * BLOCK_TILE_N))

static_assert(BK == 32 || BK == 16 || BK == 8, "swizzle_a가 이 세 경우로만 유도돼 있다");

// As의 swizzle은 BK(=행 stride)에 따라 달라진다. bank = (row*BK + col) % 32
// 이므로 BK=32면 row가 뱅크에 영향을 안 주고, BK=16이면 row의 최하위 비트가
// 상/하 16뱅크를 가른다. 어느 쪽이든 fragment 로드가 동시에 잡는
// 행 8개(g=0..7) x 열 4개(t) = 32레인을 32뱅크로 갈라야 한다.
#if BK == 32
// 모든 행이 같은 bank window(bank = col)라, 스와즐이 없으면 열 4개 = 뱅크
// 4개에 8-way로 몰린다. (row&7)<<2 로 행마다 4칸씩 밀면 8개 행이 bit2~4를
// 나눠 가져 8그룹 x 4칸 = 32뱅크로 정확히 갈린다. col의 bit0~1은 t,
// bit3~4는 kbase라 마스크(bit2~4)와 t가 안 겹친다.
__device__ __forceinline__ int swizzle_a(int row, int col) {
    return col ^ ((row & 7) << 2);
}
// row & 7 : 동시에 접근되는 행 8개(g=0..7)를 구분
// << 2    : 4칸 단위로 밀기 (0,4,...,28 = bit2~4, col<32라 행을 안 벗어남)
#elif BK == 16
// bank = 16*(row&1) + col 이라 row&1이 이미 상/하 16뱅크를 갈라준다. 남은
// 일은 같은 패리티인 행 4개(g>>1 = 0..3)를 16칸 안에서 겹치지 않게 놓는 것
// 뿐이라, (row>>1)&3 으로 4칸씩 민다. col<16이라 bit2~3만 건드리면 되고
// bit0~1(=t)과 안 겹친다.
__device__ __forceinline__ int swizzle_a(int row, int col) {
    return col ^ (((row >> 1) & 3) << 2);
}
// (row>>1)&3 : 같은 패리티 행 4개를 구분
// << 2       : 4칸 단위로 밀기 (0,4,8,12 = bit2~3, col<16이라 행을 안 벗어남)
#else   // BK == 8
// row stride 8 -> 4행마다 같은 bank window에 겹침(row, row+4, row+8, row+12가
// 전부 같은 window). 실제 fragment 로드는 한 m-tile 안에서 g=0..7(=row..row+7)
// 범위가 한 번에 접근되는데, 그 안의 충돌쌍은 (g,g+4) 뿐이다. (row>>2)&1 로
// 이 두 그룹을 구분해 4(=bit2)만큼 다른 반쪽 window로 옮기면 풀린다.
__device__ __forceinline__ int swizzle_a(int row, int col) {
    return col ^ (((row >> 2) & 1) << 2);
}
// (row>>2)&1 : 충돌하는 두 그룹(g, g+4)을 구분
// << 2       : 4칸 밀기 (0 또는 4, col<8이라 행을 안 벗어남)
#endif

// Bs[BK][BLOCK_TILE_N]: row stride(BLOCK_TILE_N=128)가 32의 배수라 모든 row가 같은 bank
// window. 한 fragment 로드에서 row 4개(kbase+t, t=0~3 또는 +4)가 동시에 접근
// 되는데, row마다 8칸씩(bit3~4) 어긋나게 놓으면 4개 row가 32뱅크를 8칸씩
// 정확히 나눠 가져서 겹치지 않는다. 접근되는 col의 하위 3비트는 g(0~7)뿐이라
// 8*row와 비트 영역이 안 겹치므로 XOR 한 방으로 같은 배치가 나온다.
// kbase가 8의 배수라 (kbase+t)&3 == t -- BK를 키워도 마스크는 그대로다.
//
// (실측은 순환 이동(band + (within + row*8) % 32) 쪽이 0.9% 빨랐다 -- XOR이
// 명령어는 10% 적은데 레지스터가 늘고 stall이 커진 탓. 차이가 작아 읽기 쉬운
// XOR을 택했다.)
__device__ __forceinline__ int swizzle_b(int row, int col) {
    return col ^ ((row & 3) << 3);
}
// row & 3 : 동시에 접근되는 행 4개를 구분 (band를 벗어나지 않게 제한)
// << 3    : 그걸 8배로 (0, 8, 16, 24 = bit3~4에만 걸침)

// global -> shared 비동기 로드 (scalar, swizzle 적용). swizzle이 원소 순서를
// 바꿔놓기 때문에 float4 벡터화 스토어는 순서가 틀어져서 못 쓰고,
// __pipeline_memcpy_async를 4byte 단위로 걸어 overlap 한다.
// BK가 32면 warp 하나가 As/Bs 모두 한 행의 연속 32 float(=128B)를 맡아 글로벌이
// 한 세그먼트로 합쳐지지만, 위 헤더 주석대로 그 이득보다 손해가 커서 BK=8을 쓴다.
__device__ __forceinline__ void load_tile_async_swizzled(
    float As[][BK], float Bs[][BLOCK_TILE_N],
    float *d_A_ptr, float *d_B_ptr,
    int blockRow, int blockCol, int k0, int K, int N,
    int flatTid, int threadsPerBlock)
{
    const int As_elems = BLOCK_TILE_M * BK;
    for (int idx = flatTid; idx < As_elems; idx += threadsPerBlock) {
        int r = idx / BK, c = idx % BK;
        __pipeline_memcpy_async(&As[r][swizzle_a(r, c)],
                                 &d_A_ptr[(blockRow + r) * K + k0 + c], sizeof(float));
    }
    const int Bs_elems = BK * BLOCK_TILE_N;
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

    // BK=32면 double buffer 포함 64KB라 static __shared__ 한도(48KB)를 넘어
    // dynamic + opt-in이 필요하다. 다만 dynamic은 base 포인터가 런타임 값이라
    // 컴파일러가 주소를 상수로 접지 못하므로, 들어가는 크기면 static을 쓴다.
#if SMEM_BYTES <= 48 * 1024
    __shared__ float As_[2][BLOCK_TILE_M][BK];
    __shared__ float Bs_[2][BK][BLOCK_TILE_N];
    auto As = As_;
    auto Bs = Bs_;
#else
    extern __shared__ float smem[];
    float (*As)[BLOCK_TILE_M][BK] = (float (*)[BLOCK_TILE_M][BK])smem;
    float (*Bs)[BK][BLOCK_TILE_N] =
        (float (*)[BK][BLOCK_TILE_N])(smem + 2 * BLOCK_TILE_M * BK);
#endif

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
    int numKSteps = K / BK;

    for (int step = 0; step < numKSteps; ++step) {
        int nextBuf = 1 - curBuf;
        int k0_next = (step + 1) * BK;

        if (step + 1 < numKSteps) {
            load_tile_async_swizzled(As[nextBuf], Bs[nextBuf], d_A_ptr, d_B_ptr,
                                      blockRow, blockCol, k0_next, K, N, flatTid, threadsPerBlock);
            __pipeline_commit();
        }

        // ---- 타일 하나 안에서 wmma k-step 4번 (shared -> fragment, swizzle 적용) ----
        for (int kk = 0; kk < K_STEPS_PER_TILE; ++kk) {
            const int kbase = kk * WMMA_K;

            for (int i = 0; i < TILES_PER_WARP_M; ++i) {
                int rowBase = warpRowOffset + i * WMMA_M;
                int r0 = rowBase + g, r1 = rowBase + g + 8;
                a_frag[i].x[0] = As[curBuf][r0][swizzle_a(r0, kbase + t)];
                a_frag[i].x[1] = As[curBuf][r1][swizzle_a(r1, kbase + t)];
                a_frag[i].x[2] = As[curBuf][r0][swizzle_a(r0, kbase + t + 4)];
                a_frag[i].x[3] = As[curBuf][r1][swizzle_a(r1, kbase + t + 4)];
            }
            for (int j = 0; j < TILES_PER_WARP_N; ++j) {
                int colBase = warpColOffset + j * WMMA_N;
                int c0 = colBase + g, c1 = colBase + g + 8;
                int rb0 = kbase + t, rb1 = kbase + t + 4;
                b_frag[j].x[0] = Bs[curBuf][rb0][swizzle_b(rb0, c0)];
                b_frag[j].x[1] = Bs[curBuf][rb1][swizzle_b(rb1, c0)];
                b_frag[j].x[2] = Bs[curBuf][rb0][swizzle_b(rb0, c1)];
                b_frag[j].x[3] = Bs[curBuf][rb1][swizzle_b(rb1, c1)];
            }

            for (int i = 0; i < TILES_PER_WARP_M; ++i)
                for (int j = 0; j < TILES_PER_WARP_N; ++j)
                    nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }

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
    // 48KB 이하면 static __shared__로 잡히므로 dynamic 크기는 0이다.
    constexpr size_t dyn_smem = (SMEM_BYTES <= 48 * 1024) ? 0 : SMEM_BYTES;

    // 48KB를 넘는 dynamic shared는 opt-in이 필요하고, L1/shared carveout도
    // shared 쪽으로 최대한 밀어줘야 SM에 블록이 들어간다.
    static bool attr_set = false;
    if (!attr_set) {
        if (dyn_smem > 0)
            CUDA_CHECK(cudaFuncSetAttribute(tc_swizzle_kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem));
        CUDA_CHECK(cudaFuncSetAttribute(tc_swizzle_kernel,
                    cudaFuncAttributePreferredSharedMemoryCarveout,
                    cudaSharedmemCarveoutMaxShared));
        attr_set = true;
    }

    dim3 dim_block(THREADS_PER_WARP, WARPS_PER_BLOCK);
    dim3 dim_grid(M / BLOCK_TILE_M, N / BLOCK_TILE_N);
    tc_swizzle_kernel<<<dim_grid, dim_block, dyn_smem>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int K = (argc > 2) ? atoi(argv[2]) : M;
    int N = (argc > 3) ? atoi(argv[3]) : M;

    int M_pad = round_up_multiple(M, BLOCK_TILE_M);
    int K_pad = round_up_multiple(K, BK);
    int N_pad = round_up_multiple(N, BLOCK_TILE_N);

    printf("[TF32 Swizzle (manual fragment x[] + XOR swizzle)] BLOCK_TILE=%dx%d, BK=%d\n",
           BLOCK_TILE_M, BLOCK_TILE_N, BK);
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
