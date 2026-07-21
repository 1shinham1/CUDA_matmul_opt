# FlashAttention implementation

FlashAttention의 알고리즘과 CUDA 최적화 효과를 단계별로 비교하는 프로젝트다.
공통 CUDA 헤더는 `include/`, 실행 가능한 벤치마크 소스는 `src/`, 정확성
테스트는 `tests/`에 둔다. 비교용 외부 구현과 비교 스크립트는 `official/`과
`comparison/`으로 격리한다.

## Directory layout

```text
FlashAttention-implementation/
├── README.md
├── Makefile
├── run.sh
├── comparison/                    # 자체 WMMA와 official FA1 비교
├── official/                      # 비교용 FlashAttention v1.0.9
├── bin/                           # 컴파일 산출물
├── include/
│   ├── common.cuh                 # CPU reference, allocator, timer
│   ├── naive_kernels.cuh          # unfused naive attention
│   ├── flash_kernels.cuh          # CUDA Core/FMA FlashAttention
│   └── flash_kernels_tc.cuh       # Tensor Core/WMMA FlashAttention
├── tests/
│   ├── test_naive.cu
│   ├── test_flash.cu
│   └── test_flash_tc.cu
├── src/
│   ├── benchmark_algorithm.cu
│   ├── benchmark_normalization.cu
│   ├── benchmark_tensor_core.cu
│   └── benchmark_paper_grid.cu
├── scripts/
│   └── benchmark.sh
├── results/
└── docs/
    ├── IMPLEMENTATION.md
    └── OPTIMIZATION_PLAN.md
```

## Implementations

| 단계 | 파일 | 목적 |
|---|---|---|
| CPU reference | `include/common.cuh` | fp64 중심의 정답 계산 |
| Naive CUDA | `include/naive_kernels.cuh` | `N×N` 중간 행렬을 저장하는 baseline |
| Flash FMA | `include/flash_kernels.cuh` | 타일링, online softmax, backward 재계산 |
| Flash WMMA | `include/flash_kernels_tc.cuh` | 두 행렬곱을 Tensor Core로 실행 |

Q/K/V/O와 gradient 저장은 fp16이고 softmax 및 주요 누적은 fp32다.
WMMA forward는 현재 `HEAD_DIM=64` 전용이다.

## Build and test

기본 GPU target은 RTX 4090용 `sm_89`다. `nvcc`가 PATH에 있어야 한다.

```bash
cd FlashAttention-implementation
make
./run.sh

# 다른 GPU architecture
make ARCH=sm_80
```

빌드 산출물은 `bin/`에만 생성된다. 이전 target 이름도 호환된다.

```bash
make test_flash_tc
make benchmark_tc
```

## Benchmark

```bash
make run
# 또는
./scripts/benchmark.sh
```

최신 CSV는 `results/results_*.csv`, 실행별 snapshot은
`results/runs/<timestamp>/`에 생성된다. 자세한 구현과 기존 측정 결과는
[`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md)를 참고한다.
