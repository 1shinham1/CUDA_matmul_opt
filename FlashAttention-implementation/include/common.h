#pragma once

#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
//  공통 매크로 / 상수
// ─────────────────────────────────────────────────────────────────────────────

#define FA_DEVICE __forceinline__ __device__
//GPU는 함수 호출(call/return, 스택 사용)의 오버헤드가 CPU보다 훨씬 부담스럽다. 
//특히 스레드마다 레지스터를 세밀하게 관리하는 커널에서는, 
//함수 호출이 그대로 남아 있으면 레지스터 할당이 꼬이거나 불필요한 오버헤드가 생길 수 있다. 
//그래서 자주 호출되는 작은 헬퍼 함수(load_fragments, warp_mma 같은 것들)는 무조건 인라인시켜서,
//컴파일된 결과가 마치 그 코드를 손으로 다 풀어쓴 것과 똑같이 나오게 강제하는 것이다.
#define FA_UNROLL _Pragma("unroll")

namespace mk {

//constexp란 프로그램이 실행되기도 전에, 컴파일러가 코드를 기계어로 바꾸는 그 순간 이미 값이 확정되어 있어야 한다
constexpr int WARP_SIZE = 32;
constexpr uint32_t FULL_MASK = 0xffffffffu;

// 이 코드 전체에서 "fragment" 는 (8, 8) b16 타일을 뜻한다.
// ldmatrix 가 한 번에 옮기는 최소 단위이자, mma 피연산자를 세는 단위다.
constexpr int FRAG = 8;

// 16B(uint4) 벡터 접근 단위. b16 원소 8개 = fragment 한 행 = 16B.
// gmem<->smem 복사와 ldmatrix 주소 계산이 모두 이 단위로 이뤄진다.
constexpr int B16_PER_16B = 8;

// mma.sync.aligned.m16n8k16 의 타일 모양 (fragment 개수 단위)
constexpr int MMA_M_FRAGS = 2; // 16 rows
constexpr int MMA_N_FRAGS = 1; //  8 cols
constexpr int MMA_K_FRAGS = 2; // 16 cols

// log2(e). optimized softmax (OPT_SOFTMAX = true) 에서 스케일에 미리 곱해둔다.
constexpr float LOG2E = 1.4426950408889634f;

} // namespace mk
