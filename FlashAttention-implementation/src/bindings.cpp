#include <torch/extension.h>

#include "entry.h"

// kernel_1.cu ~ kernel_3.cu 를 하나의 파이썬 모듈로 노출한다.
//
//   >>> import my_flash
//   >>> o, ms = my_flash.forward_3(q, k, v, None, True)

//PYBIND11_MODULE(my_flash, m) = "my_flash 모듈 만들 것"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Flash Attention 2 forward, kernel iterations 1-3";

// 주의: 줄 끝의 \ 는 "다음 줄과 이어붙이기" 표시라 반드시 그 줄의 마지막
//       글자여야 한다. 그래서 주석은 // 가 아니라 /* */ 로 \ 앞에 둔다.
#define MK_BIND(n, desc)                                                       \
    m.def("forward_" #n, &forward_##n,                                         \
          /* 인자 이름 붙이기 (파이썬에서 키워드 인자로 쓸 수 있게) */                      \
          py::arg("q"), py::arg("k"), py::arg("v"),                            \
          /* 기본값 있는 인자 (파이썬에서 생략 가능) */                                 \
          py::arg("o") = std::nullopt,                                         \
          py::arg("benchmark") = false,                                        \
          /* 설명글*/                                                            \
          "kernel " #n ": " desc)

    MK_BIND(1, "Base Implementation");
    MK_BIND(2, "Swizzling");
    MK_BIND(3, "Eagerly Loading K & V Blocks");

#undef MK_BIND
}
