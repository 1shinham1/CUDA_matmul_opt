#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <optional>
#include <tuple>

#include "kernel_args.h"
#include "tile_config.cuh"

namespace mk {

// ─────────────────────────────────────────────────────────────────────────────
//  공통 호스트 런처
//
//  kernel_N.cu 는 이 함수에 자기 커널 포인터 두 개(fp16/bf16)만 넘기면 된다.
//  반환값: (출력 텐서, 커널 실행시간 ms — benchmark=false 면 0)
// ─────────────────────────────────────────────────────────────────────────────
template <typename Cfg>
inline std::tuple<torch::Tensor, float>
launch_flash_forward(ForwardKernelFn kernel_fp16, ForwardKernelFn kernel_bf16,
                     const torch::Tensor &Q, const torch::Tensor &K,
                     const torch::Tensor &V, std::optional<torch::Tensor> out,
                     bool benchmark) {
    // ── 입력 검증 ────────────────────────────────────────────────────────────
    for (const auto &t : {Q, K, V}) {
        TORCH_CHECK(t.device().is_cuda(), "입력은 CUDA 텐서여야 합니다");
        TORCH_CHECK(t.is_contiguous(), "입력은 contiguous 여야 합니다");
        TORCH_CHECK(t.dim() == 4, "입력은 (batch, seq, heads, d_head) 4차원");
    }
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "Q, K, V 는 같은 shape 여야 합니다");

    const at::cuda::CUDAGuard guard{Q.device()};

    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, Q.device().index()));
    const int cc = prop.major * 10 + prop.minor;
    TORCH_CHECK(cc >= 80, "SM_80 이상이 필요합니다 (현재 SM_", prop.major, ".",
                prop.minor, ")");

    const auto dtype = Q.scalar_type();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "fp16 / bf16 만 지원합니다");
    TORCH_CHECK(K.scalar_type() == dtype && V.scalar_type() == dtype,
                "Q, K, V 의 dtype 이 같아야 합니다");

    const int64_t batch = Q.size(0);
    const int64_t seq_len = Q.size(1);
    const int64_t n_heads = Q.size(2);
    const int64_t d_head = Q.size(3);

    TORCH_CHECK(d_head == Cfg::d_head, "이 커널은 d_head=", Cfg::d_head,
                " 전용입니다 (입력: ", d_head, ")");
    TORCH_CHECK(seq_len % Cfg::B_r == 0, "seq_len 은 B_r(", Cfg::B_r,
                ")의 배수여야 합니다");
    TORCH_CHECK(seq_len % Cfg::B_c == 0, "seq_len 은 B_c(", Cfg::B_c,
                ")의 배수여야 합니다");

    // ── 출력 텐서 ────────────────────────────────────────────────────────────
    torch::Tensor O;
    if (out.has_value()) {
        O = out.value();
        TORCH_CHECK(O.scalar_type() == dtype, "출력 dtype 이 다릅니다");
        TORCH_CHECK(O.sizes() == Q.sizes(), "출력 shape 이 다릅니다");
        TORCH_CHECK(O.is_contiguous(), "출력은 contiguous 여야 합니다");
    } else {
        O = torch::empty_like(Q);
    }

    // ── 커널 인자 / 런치 구성 ────────────────────────────────────────────────
    const KernelArgs args{Q.data_ptr(),
                          K.data_ptr(),
                          V.data_ptr(),
                          O.data_ptr(),
                          Q.stride(0),
                          Q.stride(1),
                          Q.stride(2),
                          static_cast<int>(seq_len / Cfg::B_c)};

    const dim3 grid(static_cast<unsigned>(seq_len / Cfg::B_r),
                    static_cast<unsigned>(n_heads),
                    static_cast<unsigned>(batch));
    const dim3 block(Cfg::n_threads);

    ForwardKernelFn kernel =
        (dtype == torch::kFloat16) ? kernel_fp16 : kernel_bf16;

    // 동적 smem 은 opt-in 없이 48 KB 까지만 쓸 수 있다.
    // 기본 설정(B_r=B_c=64)은 정확히 48 KB 라 이 분기는 죽어 있다.
    if constexpr (Cfg::smem_bytes > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void *>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::smem_bytes));
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    float ms = 0.0f;
    cudaEvent_t start{}, stop{};
    if (benchmark) {
        C10_CUDA_CHECK(cudaEventCreate(&start));
        C10_CUDA_CHECK(cudaEventCreate(&stop));
        C10_CUDA_CHECK(cudaEventRecord(start, stream));
    }

    kernel<<<grid, block, Cfg::smem_bytes, stream>>>(args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (benchmark) {
        C10_CUDA_CHECK(cudaEventRecord(stop, stream));
        C10_CUDA_CHECK(cudaEventSynchronize(stop));
        C10_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        C10_CUDA_CHECK(cudaEventDestroy(start));
        C10_CUDA_CHECK(cudaEventDestroy(stop));
    }

    return {O, ms};
}

} // namespace mk
