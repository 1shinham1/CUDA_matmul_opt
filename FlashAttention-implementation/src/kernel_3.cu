// ═══════════════════════════════════════════════════════════════════════════
//  Kernel 3 — Eagerly Loading K & V Blocks
//
//  kernel_2 대비 변경점:  cp.async 를 "issue만" 해두고 나중에 기다린다.
//                         플래그가 아니라 루프 구조 자체가 바뀐다.
//      $ diff kernel_2.cu kernel_3.cu
//
//  성능: RTX4090 95.4% -> 97.7%   [기준선: torch SDPA = 100%, seq_len=4096, fp16]
//
//  ── 무엇이 달라지는가 ──────────────────────────────────────────────────────
//  kernel_2 까지는 copy -> commit -> wait 가 한 덩어리라 gmem 지연이 그대로
//  노출된다. 여기서는 세 지점에서 미리 발사한다:
//
//     프롤로그       : Q 와 K[0] 을 함께 발사. wait<1> 로 Q 만 기다린다.
//     배리어 A 직후  : V[j] 발사   -> QK matmul 과 겹친다
//     배리어 B 직후  : K[j+1] 발사 -> softmax + PV matmul 과 겹친다
//
//  softmax 를 배리어 B "뒤"로 옮긴 것도 의도적이다. K[j+1] 전송과 겹치라고.
//
//  ── 동기화 ────────────────────────────────────────────────────────────────
//  cp_async_wait<N> 은 호출한 스레드의 복사만 보장한다. K/V 는 블록 전체가
//  읽으므로 반드시 __syncthreads() 가 따라와야 한다. 배리어 2개가
//  "이번 반복 도착 보장" 과 "지난 반복 읽기 완료 보장" 을 함께 처리한다.
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

namespace kernel_3 {
using namespace mk;

// ── 이 커널의 설정 ──────────────────────────────────────────────────────────
using Cfg = TileConfig</*B_r=*/64, /*B_c=*/64, /*n_warps=*/4>;

constexpr bool ASYNC = true;        // cp.async 사용 (1번부터 켜져 있다)
constexpr bool SWIZZLE = true;      // kernel 2 에서 켜졌다
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

    // ── 프롤로그: Q 와 K[0] 을 함께 발사한다 ────────────────────────────────
    copy_rows_gmem_to_smem<Cfg::qo_rows_per_warp, D, SWIZZLE, ASYNC>(
        gQ_w, sQ_w, args.seq_stride, lane);
    cp_async_commit(); // 그룹 1 = Q

    copy_rows_gmem_to_smem<Cfg::kv_rows_per_warp, D, SWIZZLE, ASYNC>(
        gK_w, sK_w, args.seq_stride, lane);
    cp_async_commit(); // 그룹 2 = K[0]

    // wait<1> : 미완료 그룹이 1개 이하가 될 때까지. 즉 Q 만 기다리고
    //           K[0] 은 계속 날려둔 채로 Q 를 레지스터에 올린다.
    cp_async_wait<1>();
    // Q 는 워프별 영역이므로 워프 배리어로 충분하다.
    __syncwarp();
    load_fragments<D, SWIZZLE>(rQ, sQ_w, lane);

    // ── 메인 루프: KV 블록 하나씩 ───────────────────────────────────────────
    for (int j = 0; j < args.n_kv_blocks; ++j) {

        // 배리어 A: K[j] 도착 보장 + 지난 반복의 sV 읽기 완료 보장
        cp_async_wait<0>();
        __syncthreads();

        // V[j] 를 지금 발사해두면 아래 QK matmul 과 겹쳐서 전송된다.
        copy_rows_gmem_to_smem<Cfg::kv_rows_per_warp, D, SWIZZLE, ASYNC>(
            gV_w + j * kv_block_stride, sV_w, args.seq_stride, lane);
        cp_async_commit();

        // S = Q K^T
        load_fragments<D, SWIZZLE>(rK, sK, lane); // 블록 전체 64행
        zero_accum(rS);
        warp_mma<Cfg::d_frags, value_t>(rQ, rK, rS);

        // 배리어 B: V[j] 도착 보장 + 방금 끝난 sK 읽기 완료 보장
        cp_async_wait<0>();
        __syncthreads();

        // 다음 반복의 K 를 지금 발사한다. 아래 softmax + PV matmul 과 겹친다.
        if (j + 1 < args.n_kv_blocks) {
            copy_rows_gmem_to_smem<Cfg::kv_rows_per_warp, D, SWIZZLE, ASYNC>(
                gK_w + (j + 1) * kv_block_stride, sK_w, args.seq_stride, lane);
            cp_async_commit();
        }

        // online softmax  (K[j+1] 전송과 겹치도록 배리어 B 뒤에 둔다)
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

} // namespace kernel_3

#ifndef MK_STANDALONE
std::tuple<torch::Tensor, float>
forward_3(const torch::Tensor &Q, const torch::Tensor &K,
          const torch::Tensor &V, std::optional<torch::Tensor> out,
          bool benchmark) {
    return mk::launch_flash_forward<kernel_3::Cfg>(
        &kernel_3::flash_forward<half>, &kernel_3::flash_forward<nv_bfloat16>,
        Q, K, V, out, benchmark);
}
#endif // MK_STANDALONE
