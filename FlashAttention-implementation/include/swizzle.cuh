#pragma once

#include "common.h"

// ─────────────────────────────────────────────────────────────────────────────
//  XOR 스위즐 (kernel 2 부터 켜짐)
//
//  smem 한 행 = d_head(128) x b16 = 256 B = 64 word.
//  뱅크 = (byte_addr / 4) % 32 이므로 행 stride 64 word 는 32 로 나누어떨어진다.
//  => 스위즐이 없으면 "모든 행의 같은 열"이 전부 같은 뱅크에 떨어진다.
//
//  ldmatrix.x4 는 lane 0..15 가 행 0..15 를 같은 열에서 읽는다.
//    스위즐 X : 8-way 뱅크 충돌
//    스위즐 O : 행 r 의 16B 청크가 열 (r^c) 로 이동 -> 8개 청크가 32뱅크에 1회씩
//
//  주의: 이 함수는 gmem->smem, smem->RF, RF->smem 세 경로에 "똑같이" 적용되어야
//        데이터 위치가 일관된다. (copy.cuh 의 세 함수 모두가 이걸 호출한다)
//
//  chunk = 16B 단위 열 인덱스 (= b16 원소 8개 = fragment 한 행)
//  row % 8 과 XOR 하므로 청크 8개(=128 B) 그룹 안에서만 자리가 섞인다.
//  -> 16B 정렬이 유지되고, gmem 접근의 coalescing 도 그대로다.
// ─────────────────────────────────────────────────────────────────────────────

namespace mk {

template <bool SWIZZLE>
FA_DEVICE int swizzle_chunk(int row, int chunk) {
    if constexpr (SWIZZLE) {
        return (row % B16_PER_16B) ^ chunk;
    } else {
        return chunk;
    }
}

} // namespace mk
