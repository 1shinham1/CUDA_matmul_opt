#pragma once

#include "common.h"

// ─────────────────────────────────────────────────────────────────────────────
//  타일 설정
//
//  B_r, B_c, n_warps 세 개만 정하면 나머지 개수는 전부 여기서 파생된다.
//
//  기본값 TileConfig<64, 64, 4> (d_head = 128) 기준:
//        B_r = 64    B_c = 64    n_warps = 4    d_head = 128
//        ─────────────────────────────────────────────────
//        n_threads             = 128        (4 warps)
//        d_frags               = 16         (d_head = 128) / 8
//        qo_rows_per_warp      = 16         B_r / n_warps
//        qo_frags              = 2          (qo_rows_per_warp = 16) / 8
//        kv_frags              = 8          B_c / 8      <- "계산" 단위
//        kv_rows_per_warp      = 16         B_c / n_warps
//        smem_bytes            = 49152      (B_r + 2 * B_c) * (d_head = 128) * 2  = 48 KB
//
//  >>> 워프 분업이 Q/O 와 K/V 에서 다르다. 이게 이 커널의 핵심 설계다. <<<
//
//      Q, O : 각 워프가 자기 몫 16행만 로드하고, 자기 16행만 계산한다.
//      K, V : 각 워프가 자기 몫 16행을 smem 에 로드하지만,
//             계산은 블록 전체 64행에 대해 수행한다.
//
//  => softmax 의 행 리덕션이 워프 안에서 닫힌다. 워프 간 통신이 0이고,
//     루프 안 __syncthreads() 는 순전히 smem 재사용 보호용이다.
// ─────────────────────────────────────────────────────────────────────────────

namespace mk {
//TileConfig<B_r = 64, B_c = 64, n_warps = 4>   // d_head = 128 (기본값)
template <int B_r_, int B_c_, int n_warps_, int d_head_ = 128>
struct TileConfig {
    static constexpr int B_r = B_r_;
    static constexpr int B_c = B_c_;
    static constexpr int n_warps = n_warps_;
    static constexpr int d_head = d_head_;

    static constexpr int n_threads = n_warps * WARP_SIZE; //128 (블록당 스레드)

    // d_head 방향 fragment 개수. QK^T 에서는 K 차원, PV 에서는 N 차원.
    static constexpr int d_frags = d_head / FRAG; //16

    // Q/O : 워프당 행 수와 fragment 수 (로드 범위 == 계산 범위)
    static constexpr int qo_rows_per_warp = B_r / n_warps;  //16 (워프 하나가 담당하는 Q 행)
    static constexpr int qo_frags = qo_rows_per_warp / FRAG; //2

    // K/V : 계산은 블록 전체
    static constexpr int kv_frags = B_c / FRAG; //8
    // K/V : 로드는 워프별 자기 몫
    static constexpr int kv_rows_per_warp = B_c / n_warps; //16

    // Q | K | V.  O 는 Q 영역을 재사용한다 (Q 는 RF 로 올라간 뒤 없어짐).
    static constexpr int smem_bytes = (B_r + 2 * B_c) * d_head * 2; //48KB



    // 컴파일 시점에 조건을 검사해서, 조건이 거짓이면 아예 컴파일 자체를 실패시키는 C++ 문법
    static_assert(B_r % n_warps == 0, "B_r 은 n_warps 로 나누어떨어져야 한다");
    static_assert(B_c % n_warps == 0, "B_c 는 n_warps 로 나누어떨어져야 한다");
    static_assert(qo_rows_per_warp % (2 * FRAG) == 0,
                  "ldmatrix.x4 가 한 번에 16행을 읽으므로 워프당 Q 행 수는 "
                  "16의 배수여야 한다");
    static_assert(kv_frags % 2 == 0, "kv fragment 수는 짝수여야 한다");
    static_assert(d_frags % 2 == 0, "d_head fragment 수는 짝수여야 한다");
};

} // namespace mk
