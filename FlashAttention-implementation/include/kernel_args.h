#pragma once

#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
//  커널 인자
//
//  torch 에 의존하지 않는다. 그래서 kernel_N.cu 를
//    - 파이썬 확장 (launch.cuh 경유)
//    - 독립 실행 프로파일링 바이너리 (profile/harness.cu 경유)
//  양쪽에서 재사용할 수 있다.
//
//  모든 텐서가 (batch, seq, heads, d_head) row-major 이고 Q/K/V/O 의 stride 가
//  동일하다고 가정한다. 커널이 실제로 쓰는 값만 담는다.
// ─────────────────────────────────────────────────────────────────────────────

namespace mk {

struct KernelArgs {
    const void *__restrict__ Q;
    const void *__restrict__ K;
    const void *__restrict__ V;
    void *__restrict__ O;

    int64_t batch_stride;
    int64_t seq_stride;
    int64_t head_stride;

    int n_kv_blocks;
};

using ForwardKernelFn = void (*)(const KernelArgs);

} // namespace mk
