// ═══════════════════════════════════════════════════════════════════════════
//  Kernel 2 — Swizzling
//
//  kernel_1 대비 변경점:  SWIZZLE = false -> true.  그게 전부다.
//      $ diff kernel_1.cu kernel_2.cu
//
//  성능: RTX4090 47.1% -> 95.4%   [기준선: torch SDPA = 100%, seq_len=4096, fp16]
//
//  ── 왜 이렇게 커지는가 ─────────────────────────────────────────────────────
//  smem 한 행 = 128 x b16 = 256 B = 64 word. 뱅크 = (addr/4) % 32 인데
//  64 % 32 == 0 이므로, 스위즐이 없으면 모든 행의 같은 열이 같은 뱅크에 앉는다.
//
//    ldmatrix.x4 는 lane 0..15 가 행 0..15 를 같은 열에서 읽는다.
//      스위즐 X : 8개 행이 같은 4개 뱅크를 때린다 -> 8-way 충돌
//      스위즐 O : 행 r 의 청크가 열 (r^c) 로 이동 -> 8청크가 32뱅크에 1회씩
//
//  주목할 점: 명령어 수는 오히려 늘어난다. SASS 기준 HMMA 128 / LDSM 72 로
//  kernel_1 과 동일한데 LOP3 만 24 -> 119 로 증가한다. 순수하게 런타임 뱅크
//  충돌만 사라진 결과다. (이 LOP3 오버헤드를 걷어내는 것이 다음 최적화 후보)
// ═══════════════════════════════════════════════════════════════════════════

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

#include "common.h"
#include "copy.cuh"
#include "kernel_args.h"
#include "mma.cuh"
#include "ptx.cuh"
#include "softmax.cuh"
#include "tile_config.cuh"

// 파이썬 확장으로 빌드할 때만 torch 바인딩을 끌어온다.
// profile/harness.cu 는 MK_STANDALONE 을 정의해 __global__ 커널만 가져간다.
#ifndef MK_STANDALONE
#include "entry.h"
#include "launch.cuh"
#endif

namespace kernel_2 {
using namespace mk;

// ── 이 커널의 설정 ──────────────────────────────────────────────────────────
using Cfg = TileConfig</*B_r=*/64, /*B_c=*/64, /*n_warps=*/4>;

constexpr bool ASYNC = true;        // cp.async 사용 (1번부터 켜져 있다)
constexpr bool SWIZZLE = true;      // ★ kernel 1 대비 여기만 바뀌었다
constexpr bool OPT_SOFTMAX = false; // exp2 융합 경로 스위치 (softmax.cuh)

// ── 타일 상수 (TileConfig<64,64,4>, d_head=128) ─────────────────────────────
//    d_frags          = 16    d_head / 8
//    qo_frags         =  2    (B_r / n_warps) / 8   워프당 Q 16행
//    kv_frags         =  8    B_c / 8               계산은 블록 전체 64행
//    kv_rows_per_warp = 16    B_c / n_warps         복사는 워프당 16행
//    smem_bytes       = 49152 (64 + 128) * 128 * 2  = 48 KB
//
//  스레드 하나가 드는 레지스터 (32-bit):
//    rQ 2x16=32   rK 8x16=128   rV 16x8=128   rS 2x16=32   rP 2x8=16   rO 2x32=64

template <typename value_t>
__global__ void flash_forward(__grid_constant__ const KernelArgs args) {
    constexpr int D = Cfg::d_head;

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    // ── 이 CTA 가 담당할 (batch, head, Q 블록) ──────────────────────────────
    // grid = (seq_len/B_r, n_heads, batch)
    const int64_t head_off =
        static_cast<int64_t>(blockIdx.z) * args.batch_stride +
        static_cast<int64_t>(blockIdx.y) * args.head_stride;
    const int64_t qo_off =
        head_off + static_cast<int64_t>(blockIdx.x) * Cfg::B_r * args.seq_stride;

    const value_t *gQ = static_cast<const value_t *>(args.Q) + qo_off;
    value_t *gO = static_cast<value_t *>(args.O) + qo_off;
    const value_t *gK = static_cast<const value_t *>(args.K) + head_off;
    const value_t *gV = static_cast<const value_t *>(args.V) + head_off;

    // ── smem 분할: Q | K | V   (O 는 Q 영역을 재사용한다) ───────────────────
    extern __shared__ __align__(16) char smem_raw[];
    value_t *sQ = reinterpret_cast<value_t *>(smem_raw);
    value_t *sK = sQ + Cfg::B_r * D;
    value_t *sV = sK + Cfg::B_c * D;

    // ── 워프별 포인터 ───────────────────────────────────────────────────────
    // 복사는 각자 자기 몫만, 계산은 Q/O 만 자기 몫 / K/V 는 블록 전체.
    const int64_t q_rows = static_cast<int64_t>(warp) * Cfg::qo_rows_per_warp;
    const int64_t kv_rows = static_cast<int64_t>(warp) * Cfg::kv_rows_per_warp;

    const value_t *gQ_w = gQ + q_rows * args.seq_stride;
    value_t *gO_w = gO + q_rows * args.seq_stride;
    const value_t *gK_w = gK + kv_rows * args.seq_stride;
    const value_t *gV_w = gV + kv_rows * args.seq_stride;

    value_t *sQ_w = sQ + q_rows * D; // 에필로그에서 sO_w 로 재사용
    value_t *sK_w = sK + kv_rows * D;
    value_t *sV_w = sV + kv_rows * D;

    const int64_t kv_block_stride =
        static_cast<int64_t>(Cfg::B_c) * args.seq_stride;

    // ── 레지스터 타일 ───────────────────────────────────────────────────────
    uint32_t rQ[Cfg::qo_frags][Cfg::d_frags];
    uint32_t rK[Cfg::kv_frags][Cfg::d_frags];
    uint32_t rV[Cfg::d_frags][Cfg::kv_frags];
    float rS[Cfg::qo_frags][Cfg::kv_frags * 2];
    uint32_t rP[Cfg::qo_frags][Cfg::kv_frags];
    float rO[Cfg::qo_frags][Cfg::d_frags * 2];

    float m[Cfg::qo_frags]; // 행별 running max
    float l[Cfg::qo_frags]; // 행별 running exp 합

    zero_accum(rO);
    FA_UNROLL
    for (int q = 0; q < Cfg::qo_frags; ++q) {
        m[q] = -INFINITY;
        l[q] = 0.0f;
    }

    const float scale =
        rsqrtf(static_cast<float>(D)) * (OPT_SOFTMAX ? LOG2E : 1.0f);

    // ── 프롤로그: Q 를 gmem -> smem -> RF (커널 전체에서 딱 한 번) ──────────
    copy_rows_gmem_to_smem<Cfg::qo_rows_per_warp, D, SWIZZLE, ASYNC>(
        gQ_w, sQ_w, args.seq_stride, lane);
    cp_async_commit();
    cp_async_wait<0>();
    // cp_async_wait 는 "이 스레드"의 복사만 보장한다. 워프 전체가 sQ_w 를
    // 읽으므로 워프 배리어가 필요하다. (Q 는 워프별 영역이라 syncwarp 로 충분)
    __syncwarp();
    load_fragments<D, SWIZZLE>(rQ, sQ_w, lane);

    // ── 메인 루프: KV 블록 하나씩 ───────────────────────────────────────────
    for (int j = 0; j < args.n_kv_blocks; ++j) {

        // K 블록: gmem -> smem. 발사하고 곧바로 기다린다 (오버랩 없음).
        copy_rows_gmem_to_smem<Cfg::kv_rows_per_warp, D, SWIZZLE, ASYNC>(
            gK_w + j * kv_block_stride, sK_w, args.seq_stride, lane);
        cp_async_commit();
        cp_async_wait<0>();
        // 배리어 (1): sK 쓰기 완료 + 지난 반복의 sV 읽기 완료를 함께 보장
        __syncthreads();

        // S = Q K^T
        load_fragments<D, SWIZZLE>(rK, sK, lane); // 블록 전체 64행
        zero_accum(rS);
        warp_mma<Cfg::d_frags, value_t>(rQ, rK, rS);

        // online softmax
        float m_new[Cfg::qo_frags];
        if constexpr (!OPT_SOFTMAX) {
            scale_accum(rS, scale); // 별도 FMUL 패스 (OPT_SOFTMAX 면 사라진다)
        }
        row_max(rS, m, m_new);
        rescale_l_and_O<OPT_SOFTMAX>(m_new, m, l, rO, scale);
        exp_accum<OPT_SOFTMAX>(rS, m, scale);
        accumulate_l(rS, l);

        // S(f32 누산기) -> P(b16 mma 입력). 레지스터 안에서 끝난다.
        convert_f32_to_b16<value_t>(rS, rP);

        // V 블록: gmem -> smem
        copy_rows_gmem_to_smem<Cfg::kv_rows_per_warp, D, SWIZZLE, ASYNC>(
            gV_w + j * kv_block_stride, sV_w, args.seq_stride, lane);
        cp_async_commit();
        cp_async_wait<0>();
        // 배리어 (2): sV 쓰기 완료 + 이번 반복의 sK 읽기 완료를 함께 보장
        __syncthreads();

        // O += P V   (V 는 ldmatrix.trans 로 전치하며 읽는다)
        load_fragments_trans<D, SWIZZLE>(rV, sV, lane);
        warp_mma<Cfg::kv_frags, value_t>(rP, rV, rO);
    }

    // ── 에필로그 ────────────────────────────────────────────────────────────
    normalize_O(rO, l);

    uint32_t rO_b16[Cfg::qo_frags][Cfg::d_frags];
    convert_f32_to_b16<value_t>(rO, rO_b16);

    // 누산기 레이아웃 그대로 gmem 에 쓰면 워프당 흩어진 4B 스토어 8개가 된다.
    // smem 을 한 번 경유하면 128 B 정렬 라인 4개(512 B/warp)로 나간다.
    // sQ_w 는 Q 를 RF 에 올린 뒤 죽었으므로 그대로 쓴다.
    value_t *sO_w = sQ_w;
    store_fragments<D, SWIZZLE>(rO_b16, sO_w, lane);
    __syncwarp(); // 워프별 독립 영역이므로 __syncthreads() 는 불필요
    copy_rows_smem_to_gmem<Cfg::qo_rows_per_warp, D, SWIZZLE>(
        gO_w, sO_w, args.seq_stride, lane);
}

} // namespace kernel_2

#ifndef MK_STANDALONE
std::tuple<torch::Tensor, float>
forward_2(const torch::Tensor &Q, const torch::Tensor &K,
          const torch::Tensor &V, std::optional<torch::Tensor> out,
          bool benchmark) {
    return mk::launch_flash_forward<kernel_2::Cfg>(
        &kernel_2::flash_forward<half>, &kernel_2::flash_forward<nv_bfloat16>,
        Q, K, V, out, benchmark);
}
#endif // MK_STANDALONE
