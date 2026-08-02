#pragma once

#include "common.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Online softmax (Flash Attention 2)
//
//  mma 의 f32 누산기에서 "한 row"은 연속된 4개 lane 에 흩어져 있다 (lane t -> 행
//  t/4). 우리가 그 매핑을 알기 때문에 행 리덕션이 xor 셔플 2번으로 끝난다.
//  wmma::fragment 는 이 매핑이 명세상 정의되어 있지 않아서, 누산기를 smem 에
//  내렸다가 다시 올려야 한다 — KV 블록마다.
//
//  두 가지 softmax 경로:
//
//    OPT = false (커널 1~3 이 쓰는 기본 경로)
//        S *= scale            <- 별도 FMUL 패스
//        S  = exp(S - m)       <- expf 는 내부적으로 *log2(e) 후 EX2
//
//    OPT = true  (exp2 융합 경로)
//        scale 에 log2(e) 를 미리 곱해둔다.
//        S  = exp2(S*scale - m*scale)
//                 ^^^^^^^^^^^^^^^^^^ FFMA 하나 + MUFU.EX2 하나로 융합된다.
//        => 별도 스케일링 패스가 사라진다. (SASS: FMUL 194 -> 132, FFMA 2 -> 34)
//        이때 m 은 "스케일되지 않은" 원본 S 의 최댓값을 담는다.
// ─────────────────────────────────────────────────────────────────────────────

namespace mk {

// OPT = false 경로에서만 필요하다. S 를 1/sqrt(d) 로 스케일한다.
template <int Q, int KV>
FA_DEVICE void scale_accum(float (&S)[Q][KV], float scale) {
    FA_UNROLL
    for (int q = 0; q < Q; ++q) {
        FA_UNROLL
        for (int k = 0; k < KV; ++k) {
            S[q][k] *= scale;
        }
    }
}

// m_new[q] = max(m[q], rowmax(S[q][:]))
// 스레드 내 최댓값 -> 4-lane 그룹 리덕션 (거리 2, 1).
template <int Q, int KV>
FA_DEVICE void row_max(const float (&S)[Q][KV], const float (&m)[Q],
                       float (&m_new)[Q]) {
    FA_UNROLL
    for (int q = 0; q < Q; ++q) {
        float v = m[q];
        FA_UNROLL
        for (int k = 0; k < KV; ++k) {
            v = fmaxf(v, S[q][k]);
        }
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, 2));
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, 1));
        m_new[q] = v;
    }
}

// 최댓값이 갱신된 만큼 지금까지의 l 과 O 를 되돌린다.
//   c = exp(m_old - m_new);   l *= c;   O *= c;   m = m_new
template <bool OPT, int Q, int D>
FA_DEVICE void rescale_l_and_O(const float (&m_new)[Q], float (&m)[Q],
                               float (&l)[Q], float (&O)[Q][D], float scale) {
    FA_UNROLL
    for (int q = 0; q < Q; ++q) {
        float c;
        if constexpr (OPT) {
            c = exp2f((m[q] - m_new[q]) * scale);
        } else {
            c = expf(m[q] - m_new[q]);
        }
        m[q] = m_new[q];
        l[q] *= c;

        FA_UNROLL
        for (int d = 0; d < D; ++d) {
            O[q][d] *= c;
        }
    }
}

// S = exp(S - m)   (OPT 면 exp2 로 스케일을 융합)
template <bool OPT, int Q, int KV>
FA_DEVICE void exp_accum(float (&S)[Q][KV], const float (&m)[Q], float scale) {
    FA_UNROLL
    for (int q = 0; q < Q; ++q) {
        const float m_scaled = OPT ? m[q] * scale : m[q];
        FA_UNROLL
        for (int k = 0; k < KV; ++k) {
            if constexpr (OPT) {
                S[q][k] = exp2f(S[q][k] * scale - m_scaled);
            } else {
                S[q][k] = expf(S[q][k] - m_scaled);
            }
        }
    }
}

// l += rowsum(S).  lane 간 합산은 하지 않는다 — 마지막에 한 번만 모으면 된다.
template <int Q, int KV>
FA_DEVICE void accumulate_l(const float (&S)[Q][KV], float (&l)[Q]) {
    FA_UNROLL
    for (int q = 0; q < Q; ++q) {
        FA_UNROLL
        for (int k = 0; k < KV; ++k) {
            l[q] += S[q][k];
        }
    }
}

// 커널 끝에서 딱 한 번: l 을 4-lane 그룹에서 모으고 O 를 나눈다.
template <int Q, int D>
FA_DEVICE void normalize_O(float (&O)[Q][D], float (&l)[Q]) {
    FA_UNROLL
    for (int q = 0; q < Q; ++q) {
        l[q] += __shfl_xor_sync(FULL_MASK, l[q], 2);
        l[q] += __shfl_xor_sync(FULL_MASK, l[q], 1);
    }

    FA_UNROLL
    for (int q = 0; q < Q; ++q) {
        const float inv_l = 1.0f / l[q];
        FA_UNROLL
        for (int d = 0; d < D; ++d) {
            O[q][d] *= inv_l;
        }
    }
}

} // namespace mk
