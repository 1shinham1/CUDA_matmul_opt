#pragma once

#include <torch/extension.h>

#include <optional>
#include <tuple>

// kernel_1.cu ~ kernel_7.cu 가 각각 하나씩 정의하고, bindings.cpp 가 노출한다.
// 반환값: (출력 텐서, 커널 실행시간 ms)

#define MK_DECLARE_FORWARD(n)                                                  \
    std::tuple<torch::Tensor, float> forward_##n(                              \
        const torch::Tensor &Q, const torch::Tensor &K,                        \
        const torch::Tensor &V, std::optional<torch::Tensor> out,              \
        bool benchmark)

MK_DECLARE_FORWARD(1);
MK_DECLARE_FORWARD(2);
MK_DECLARE_FORWARD(3);
MK_DECLARE_FORWARD(4);
MK_DECLARE_FORWARD(5);
MK_DECLARE_FORWARD(6);
MK_DECLARE_FORWARD(7);

/*
python:  my_flash.forward_1(q, k, v)
              │
              │  bindings.cpp 가 연결
              ▼
C++:     forward_1(...)                    ← kernel_1.cu 하단, 네임스페이스 밖
              │
              ▼
         launch_flash_forward<kernel_1::Cfg>(&kernel_1::flash_forward<half>, ...)
              │
              ▼
GPU:     kernel_1::flash_forward<<<grid, block, 48KB>>>(args)    ← 진짜 커널
*/