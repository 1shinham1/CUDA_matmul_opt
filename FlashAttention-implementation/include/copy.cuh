#pragma once

#include "common.h"
#include "ptx.cuh"
#include "swizzle.cuh"

// ─────────────────────────────────────────────────────────────────────────────
//  메모리 계층 간 복사 4종
//
//    gmem -> smem :  copy_rows_gmem_to_smem   (Q, K, V 로드)
//    smem -> gmem :  copy_rows_smem_to_gmem   (O 저장)
//    smem -> RF   :  load_fragments           (Q, K)  / load_fragments_trans (V)
//    RF   -> smem :  store_fragments          (O)
//
//  네 함수 모두 같은 swizzle_chunk<SWIZZLE>() 를 쓴다. 하나라도 빠지면
//  데이터 위치가 어긋나 결과가 틀린다.
// ─────────────────────────────────────────────────────────────────────────────

namespace mk {

// ── gmem <-> smem ───────────────────────────────────────────────────────────
//
// 워프 하나가 (ROWS, COLS) 블록을 옮긴다.
// 한 번의 반복에서 32 lane x 16B = 512 B = 2행 x 256 B 를 처리한다.
//   lane 0..15 -> 행 r  의 청크 0..15   (256 B 연속)
//   lane 16..31-> 행 r+1의 청크 0..15
// => gmem 쪽은 128 B 트랜잭션 4개로 완전히 coalesce 된다.

template <int ROWS, int COLS, bool SWIZZLE, bool ASYNC, typename value_t>
FA_DEVICE void copy_rows_gmem_to_smem(const value_t *gmem, value_t *smem,
                                      int64_t gmem_row_stride, int lane) {
    constexpr int CHUNKS_PER_ROW = COLS / B16_PER_16B;
    constexpr int ROWS_PER_ITER = WARP_SIZE / CHUNKS_PER_ROW;
    static_assert(CHUNKS_PER_ROW <= WARP_SIZE, "행 하나가 워프보다 넓다");
    static_assert(ROWS % ROWS_PER_ITER == 0, "ROWS 가 나누어떨어지지 않는다");

    const int lane_row = lane / CHUNKS_PER_ROW;
    const int chunk = lane % CHUNKS_PER_ROW;

    FA_UNROLL
    for (int r = 0; r < ROWS; r += ROWS_PER_ITER) {
        const int row = r + lane_row;
        const int schunk = swizzle_chunk<SWIZZLE>(row, chunk);

        const value_t *src = gmem + row * gmem_row_stride + chunk * B16_PER_16B;
        value_t *dst = smem + row * COLS + schunk * B16_PER_16B;

        if constexpr (ASYNC) {
            cp_async_16B(dst, src);
        } else {
            copy_16B(dst, src);
        }
    }
}

template <int ROWS, int COLS, bool SWIZZLE, typename value_t>
FA_DEVICE void copy_rows_smem_to_gmem(value_t *gmem, const value_t *smem,
                                      int64_t gmem_row_stride, int lane) {
    constexpr int CHUNKS_PER_ROW = COLS / B16_PER_16B;
    constexpr int ROWS_PER_ITER = WARP_SIZE / CHUNKS_PER_ROW;

    const int lane_row = lane / CHUNKS_PER_ROW;
    const int chunk = lane % CHUNKS_PER_ROW;

    FA_UNROLL
    for (int r = 0; r < ROWS; r += ROWS_PER_ITER) {
        const int row = r + lane_row;
        const int schunk = swizzle_chunk<SWIZZLE>(row, chunk);

        copy_16B(gmem + row * gmem_row_stride + chunk * B16_PER_16B,
                 smem + row * COLS + schunk * B16_PER_16B);
    }
}

// ── smem -> RF (비전치) : Q, K ──────────────────────────────────────────────
//
// regs[m][k] 는 (행 fragment m, 열 fragment k) 를 담는다.
// ldmatrix.x4 한 번이 (16, 16) = 2x2 fragment 를 옮기므로 두 루프 모두 2씩 뛴다.
//
// k_frag_offset : smem 의 열 fragment 시작 위치.
//   블록 전체를 한 번에 올릴 땐 0, 조각내어 올릴 땐(kernel 4~) 현재 k 위치.

template <int COLS, bool SWIZZLE, typename value_t, int M_FRAGS, int K_FRAGS>
FA_DEVICE void load_fragments(uint32_t (&regs)[M_FRAGS][K_FRAGS],
                              const value_t *smem, int lane,
                              int k_frag_offset = 0) {
    static_assert(M_FRAGS % 2 == 0 && K_FRAGS % 2 == 0, "2의 배수여야 한다");

    const int lane_row = lane % 16;   // 행렬 0/1 의 행 (lane 0..15 / 16..31)
    const int lane_chunk = lane / 16; // 행렬 2/3 은 열 fragment +1

    FA_UNROLL
    for (int m = 0; m < M_FRAGS; m += 2) {
        const int row = m * FRAG + lane_row;
        FA_UNROLL
        for (int k = 0; k < K_FRAGS; k += 2) {
            const int chunk = k_frag_offset + k + lane_chunk;
            const int schunk = swizzle_chunk<SWIZZLE>(row, chunk);

            ldmatrix_x4(&smem[row * COLS + schunk * B16_PER_16B], regs[m][k],
                        regs[m + 1][k], regs[m][k + 1], regs[m + 1][k + 1]);
        }
    }
}

// ── smem -> RF (전치) : V ───────────────────────────────────────────────────
//
// smem 에는 V 가 (B_c, d_head) 로 들어있는데, PV 행렬곱은 V 를 B 피연산자
// (n = d_head, k = B_c) 로 요구한다. ldmatrix.trans 가 로드하면서 전치해준다.
//   => smem 에 V^T 사본을 따로 만들 필요가 없다.
//
// regs[n][k] : n = d_head fragment (smem 의 열), k = B_c fragment (smem 의 행)

template <int COLS, bool SWIZZLE, typename value_t, int N_FRAGS, int K_FRAGS>
FA_DEVICE void load_fragments_trans(uint32_t (&regs)[N_FRAGS][K_FRAGS],
                                    const value_t *smem, int lane,
                                    int k_frag_offset = 0) {
    static_assert(N_FRAGS % 2 == 0 && K_FRAGS % 2 == 0, "2의 배수여야 한다");

    const int lane_row = lane % 16;
    const int lane_chunk = lane / 16;

    FA_UNROLL
    for (int k = 0; k < K_FRAGS; k += 2) {
        const int row = (k_frag_offset + k) * FRAG + lane_row;
        FA_UNROLL
        for (int n = 0; n < N_FRAGS; n += 2) {
            const int chunk = n + lane_chunk;
            const int schunk = swizzle_chunk<SWIZZLE>(row, chunk);

            ldmatrix_x4_trans(&smem[row * COLS + schunk * B16_PER_16B],
                              regs[n][k], regs[n][k + 1], regs[n + 1][k],
                              regs[n + 1][k + 1]);
        }
    }
}

// ── RF -> smem : O ──────────────────────────────────────────────────────────
//
// mma 누산기 레이아웃(lane t -> 행 t/4, 열 (t%4)*2)을 그대로 4B 씩 내려쓴다.
// 이걸 gmem 에 직접 쓰면 워프당 흩어진 4B 스토어 8개가 되지만,
// smem 을 한 번 경유하면 copy_rows_smem_to_gmem 이 128 B 정렬 라인으로 내보낸다.

template <int COLS, bool SWIZZLE, typename value_t, int M_FRAGS, int N_FRAGS>
FA_DEVICE void store_fragments(const uint32_t (&regs)[M_FRAGS][N_FRAGS],
                               value_t *smem, int lane) {
    const int lane_row = lane / 4;
    const int lane_col = (lane % 4) * 2;

    FA_UNROLL
    for (int m = 0; m < M_FRAGS; ++m) {
        const int row = m * FRAG + lane_row;
        FA_UNROLL
        for (int n = 0; n < N_FRAGS; ++n) {
            const int schunk = swizzle_chunk<SWIZZLE>(row, n);
            *reinterpret_cast<uint32_t *>(
                &smem[row * COLS + schunk * B16_PER_16B + lane_col]) =
                regs[m][n];
        }
    }
}

} // namespace mk
