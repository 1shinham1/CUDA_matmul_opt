#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "common.h"
#include "ptx.cuh"

namespace mk {

// ─────────────────────────────────────────────────────────────────────────────
//  워프 단위 행렬곱 타일
//
//    C[M][N*2] += A[M][*] * B[N][*]^T     (K_FRAGS 개의 8-열 fragment 만큼)
//
//  A, B 의 k 시작 위치는 a_k0 / b_k0 로 준다. 둘 다 완전히 펼쳐진(unroll)
//  루프의 인덱스로만 넘겨야 한다. 런타임 값이 들어가면 레지스터 배열에
//  동적 인덱싱이 생겨 local memory 로 스필된다.
//  (빌드 시 -Xptxas=-warn-spills 로 확인 가능)
//
//  용례:
//    S = Q K^T   : warp_mma<d_frags >(rQ, rK, rS)     M=qo_frags, N=kv_frags
//    O += P V    : warp_mma<kv_frags>(rP, rV, rO)     M=qo_frags, N=d_frags
// ─────────────────────────────────────────────────────────────────────────────

template <int K_FRAGS, typename value_t, int M_FRAGS, int A_K, int N_FRAGS,
          int B_K>
FA_DEVICE void warp_mma(const uint32_t (&A)[M_FRAGS][A_K],
                        const uint32_t (&B)[N_FRAGS][B_K],
                        float (&C)[M_FRAGS][N_FRAGS * 2], int a_k0 = 0,
                        int b_k0 = 0) {
    static_assert(K_FRAGS % MMA_K_FRAGS == 0, "K 는 16의 배수여야 한다");
    static_assert(M_FRAGS % MMA_M_FRAGS == 0, "M 은 16의 배수여야 한다");

    FA_UNROLL
    for (int k = 0; k < K_FRAGS; k += MMA_K_FRAGS) {
        FA_UNROLL
        for (int m = 0; m < M_FRAGS; m += MMA_M_FRAGS) {
            FA_UNROLL
            for (int n = 0; n < N_FRAGS; n += MMA_N_FRAGS) {
                mma_m16n8k16<value_t>(
                    C[m][2 * n], C[m][2 * n + 1], C[m + 1][2 * n],
                    C[m + 1][2 * n + 1], A[m][a_k0 + k], A[m + 1][a_k0 + k],
                    A[m][a_k0 + k + 1], A[m + 1][a_k0 + k + 1],
                    B[n][b_k0 + k], B[n][b_k0 + k + 1]);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  f32 누산기 -> b16 mma 입력  (S -> P,  O -> O_b16)
//
//  ptx.cuh 의 레이아웃 표에서 봤듯이 mma 의 D(f32) 와 A(b16) 는 (행, 열) 매핑이
//  완전히 같다. 폭만 4B -> 2B 로 줄어든다. 그래서 float2 -> half2 캐스팅
//  하나로 끝난다: 레인 간 셔플 0회, smem 왕복 0회.
//
//  wmma 에는 accumulator fragment 를 matrix_a fragment 로 바꾸는 API 자체가
//  없어서, 이 자리에서 반드시 smem 을 왕복해야 한다. Flash Attention 에서
//  이건 KV 블록마다 발생하므로 치명적이다.
// ─────────────────────────────────────────────────────────────────────────────

template <typename value_t, int M_FRAGS, int N_FRAGS>
FA_DEVICE void convert_f32_to_b16(const float (&src)[M_FRAGS][N_FRAGS * 2],
                                  uint32_t (&dst)[M_FRAGS][N_FRAGS]) {
    using v2_t = std::conditional_t<std::is_same_v<value_t, half>, half2,
                                    nv_bfloat162>;

    const float2(&s)[M_FRAGS][N_FRAGS] =
        reinterpret_cast<const float2(&)[M_FRAGS][N_FRAGS]>(src);
    v2_t(&d)[M_FRAGS][N_FRAGS] =
        reinterpret_cast<v2_t(&)[M_FRAGS][N_FRAGS]>(dst);

    FA_UNROLL
    for (int m = 0; m < M_FRAGS; ++m) {
        FA_UNROLL
        for (int n = 0; n < N_FRAGS; ++n) {
            if constexpr (std::is_same_v<value_t, half>) {
                d[m][n] = __float22half2_rn(s[m][n]);
            } else {
                d[m][n] = __float22bfloat162_rn(s[m][n]);
            }
        }
    }
}

template <int M, int N>
FA_DEVICE void zero_accum(float (&C)[M][N]) {
    FA_UNROLL
    for (int m = 0; m < M; ++m) {
        FA_UNROLL
        for (int n = 0; n < N; ++n) {
            C[m][n] = 0.0f;
        }
    }
}

} // namespace mk
