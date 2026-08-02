#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "common.h"

// ─────────────────────────────────────────────────────────────────────────────
//  인라인 PTX 래퍼
//
//  이 파일이 "PTX 레벨"의 전부다. 나머지 코드는 전부 평범한 C++ 이다.
//  wmma 로는 표현할 수 없는 것들이 여기 모여 있다:
//    - cp.async        : gmem -> smem 비동기 복사 (레지스터를 경유하지 않음)
//    - ldmatrix        : 워프 협동 smem 로드. fragment 레이아웃을 우리가 안다.
//    - mma.m16n8k16    : 네이티브 텐서코어 명령. 누산기 레이아웃을 우리가 안다.
// ─────────────────────────────────────────────────────────────────────────────

namespace mk {

// ── cp.async ────────────────────────────────────────────────────────────────
//cp.async .cg  .shared .global  .L2::128B    [dst], [src], 16;
//   │      │      │       │         └ 프리페치 힌트
//   │      │      └───────┴─ 목적지 공간 ← 소스 공간
//   │      └ 캐시 정책 (.ca=L1+L2, .cg=L2만)
//   └ 오퍼레이션                              └ 바이트 수 (4/8/16)

// 16B 를 gmem 에서 smem 으로 비동기 복사한다.
// .cg 는 L1 을 우회한다 (어차피 재사용하지 않으므로 L1 을 오염시킬 이유가 없다).
FA_DEVICE void cp_async_16B(void *smem, const void *gmem) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" ::"r"(
                     smem_addr),
                 "l"(gmem));
}

// 지금까지 issue한 cp.async 들을 하나의 그룹으로 묶는다.
FA_DEVICE void cp_async_commit() { asm volatile("cp.async.commit_group;"); }

// 가장 최근 N개 그룹만 남기고 나머지가 끝날 때까지 기다린다.
//   wait<0> : 전부 대기
//   wait<1> : 가장 최근 1개 그룹은 계속 날려둔 채로 나머지만 대기
// 주의: 이건 "호출한 스레드"의 복사만 보장한다. 워프/블록 전체가 읽으려면
//       뒤에 __syncwarp() 또는 __syncthreads() 가 반드시 따라와야 한다.
template <int N>
FA_DEVICE void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// 동기 버전 (kernel 들이 async 를 쓰지 않을 때 비교용)
FA_DEVICE void copy_16B(void *dst, const void *src) {
    *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
}

// ── ldmatrix ────────────────────────────────────────────────────────────────
//ldmatrix .sync .aligned .x4 .trans .m8n8 .shared .b16   {d0..d3}, [addr];
//            │       │     │    │      │      │      └ 원소 타입
//            │       │     │    │      │      └ 주소 공간
//            │       │     │    │      └ 행렬 모양
//            │       │     │    └ 8×8 전치 (옵션)
//            │       │     └ 행렬 몇 개 (.x1/.x2/.x4)
//            │       └ 워프 전체가 참여해야 함(정렬 요구)
//            └ 워프 동기 명령

// smem 의 (16, 16) b16 청크를 워프 레지스터로 옮긴다 = (8,8) fragment 4개. (16x16x4B = 512B)
//
// 주소 규약: lane L 이 하나의 행 주소를 제공한다.
//   lane  0.. 7 -> 행렬 0 의 8개 행     -> d0
//   lane  8..15 -> 행렬 1 의 8개 행     -> d1
//   lane 16..23 -> 행렬 2 의 8개 행     -> d2
//   lane 24..31 -> 행렬 3 의 8개 행     -> d3
FA_DEVICE void ldmatrix_x4(const void *smem, uint32_t &d0, uint32_t &d1,
                           uint32_t &d2, uint32_t &d3) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, "
                 "[%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(smem_addr));
}

// 위와 같지만 각 (8,8) 행렬을 traspose해서 register에 담는다.
// V 를 (B_c, d_head) 로 저장해두고 (d_head, B_c) 로 읽을 때 쓴다.
// -> smem 에 V^T 를 따로 만들 필요가 없다.
FA_DEVICE void ldmatrix_x4_trans(const void *smem, uint32_t &d0, uint32_t &d1,
                                 uint32_t &d2, uint32_t &d3) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, "
                 "%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(smem_addr));
}

// ── mma ─────────────────────────────────────────────────────────────────────
//mma .sync .aligned .m16n8k16 .row .col .f32 .f16 .f16 .f32   D, A, B, C;
//                       │       │    │    │    │    │    └ C 타입
//                       │       │    │    │    └────┴ A, B 타입
//                       │       │    │    └ D 타입
//                       │       └────┴ A는 row-major, B는 col-major
//                       └ M×N×K

// D = A * B^T + D,   A: (16,16) b16,  B: (8,16) b16,  D: (16,8) f32
//
// 레지스터 레이아웃 (lane t 기준) — 이 표가 이 프로젝트 전체의 근간이다:
//
//   A  a0 : 행 t/4    , 열 (t%4)*2 + {0,1}        (즉 (행fragment 0, 열fragment 0))
//      a1 : 행 t/4 + 8, 열 (t%4)*2 + {0,1}        (       1        ,        0      )
//      a2 : 행 t/4    , 열 (t%4)*2 + {0,1} + 8    (       0        ,        1      )
//      a3 : 행 t/4 + 8, 열 (t%4)*2 + {0,1} + 8    (       1        ,        1      )
//              A
//                            열 0-7      열 8-15
//                    행 0-7 [   a0   ]  [   a2   ]
//                    행 8-15[   a1   ]  [   a3   ]
//
//   D  d0,d1 : 행 t/4    , 열 (t%4)*2 + {0,1}
//      d2,d3 : 행 t/4 + 8, 열 (t%4)*2 + {0,1}
//
//   >>> D(f32) 와 A(b16) 의 (행, 열) 매핑이 완전히 동일하다. <<<
//   그래서 S 누산기를 P 입력으로 바꿀 때 float2 -> half2 캐스팅 한 번이면 된다.
//   레인 간 셔플도, smem 왕복도 필요 없다. (softmax.cuh 의 convert_f32_to_b16)
//   wmma 로는 이 경로가 아예 표현되지 않는다.
template <typename value_t>
FA_DEVICE void mma_m16n8k16(float &d0, float &d1, float &d2, float &d3,
                            uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                            uint32_t b0, uint32_t b1) {
    static_assert(std::is_same_v<value_t, half> ||
                      std::is_same_v<value_t, nv_bfloat16>,
                  "value_t must be half or nv_bfloat16");

    if constexpr (std::is_same_v<value_t, half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                     "{%0, %1, %2, %3};"
                     : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                     "{%0, %1, %2, %3};"
                     : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
}

} // namespace mk
