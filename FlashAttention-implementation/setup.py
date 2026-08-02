"""my_kernel 독립 빌드 스크립트.

    cd my_kernel
    pip install --no-build-isolation -e .

빌드 대상 아키텍처는 기본적으로 현재 GPU 를 따라간다.
환경변수로 덮어쓸 수 있다:

    MK_ARCH="80;89" pip install --no-build-isolation -e .
"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).parent.resolve()


def target_arches() -> list[str]:
    env = os.environ.get("MK_ARCH", "").strip()
    if env:
        return [a.strip().replace(".", "") for a in env.split(";") if a.strip()]

    import torch

    if not torch.cuda.is_available():
        return ["80"]  # Ampere 기본값
    major, minor = torch.cuda.get_device_capability(0)
    return [f"{major}{minor}"]


def nvcc_args() -> list[str]:
    args = [
        "-std=c++20",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        '-Xcudafe="--diag_suppress=3189"',  # c++20 + pytorch 헤더 경고
        # 진단: 레지스터 사용량과 스필을 빌드 로그에 찍는다.
        # 이 프로젝트에서 스필은 곧 버그다 (레지스터 배열 동적 인덱싱).
        "--resource-usage",
        "-Xptxas=-warn-spills",
        "-Xptxas=-warn-lmem-usage",
        "--generate-line-info",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-Xcompiler=-fdiagnostics-color=always",
    ]
    for arch in target_arches():
        args += ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]
    return args

#my_flash.cpython-310-x86_64-linux-gnu.so 만듦
#import my_flash 할 때 사용
setup(
    name="my_flash",
    version="0.1.0",
    description="Flash Attention 2 forward — kernel iterations 1-3",
    ext_modules=[
        CUDAExtension(
            name="my_flash",
            sources=[
                "src/bindings.cpp",
                "src/kernel_1.cu",
                "src/kernel_2.cu",
                "src/kernel_3.cu",
            ],
            include_dirs=[str(HERE / "include"), str(HERE / "src")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fdiagnostics-color=always"],
                "nvcc": nvcc_args(),
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
