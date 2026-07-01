# GEMM Optimization — CUDA

4096×4096 FP32 행렬 곱셈(GEMM)을 단계별로 최적화하며 cuBLAS 성능에 근접하는 과정을 담은 프로젝트입니다.  
CUDA Core(FP32)와 Tensor Core(TF32, WMMA API) 두 가지 경로로 최적화합니다.

---

## 환경 요구사항

| 항목 | 버전 |
|------|------|
| CUDA Toolkit | 13.3 |
| GPU | RTX 4090 (sm_89) |
| nvcc | CUDA Toolkit에 포함 |

---

## 빌드 & 실행

```bash
# 실행 권한 부여 (최초 1회)
chmod +x run.sh scripts/benchmark.sh

# 전체 빌드
make all

# 전체 벤치마크 실행 (결과를 results/benchmark_<timestamp>.csv 에 저장)
make run

# 특정 커널만 실행
./run.sh 5      # 05_gemm_vectorization
./run.sh 12     # 12_gemm_tc_warptiling

# GPU 스펙 확인
make info

# NCU 프로파일 생성 → results/profiles/*.ncu-rep
make profile

# 빌드 산출물 삭제
make clean
```

---

## 최적화 단계

### CUDA Core (FP32)

| # | 파일 | 핵심 기법 | 주요 포인트 |
|---|------|-----------|-------------|
| 01 | `01_gemm_naive.cu` | 기본 구현 | thread 1개 = C 원소 1개, coalescing 없음 |
| 02 | `02_gemm_coalesced.cu` | Memory Coalescing | 1D threadIdx → warp이 연속 메모리 접근 |
| 03 | `03_gemm_shared_memory.cu` | Shared Memory Tiling | GMEM → SMEM 타일 로드로 반복 접근 제거 |
| 04 | `04_gemm_microtiling.cu` | Register Tiling | thread 1개가 TR×TC 서브타일 담당, register 재사용 |
| 05 | `05_gemm_vectorization.cu` | float4 벡터화 | 128-bit 로드/스토어, As 전치 저장으로 연속 읽기 |
| 06 | `06_gemm_param_tune.cu` | 파라미터 튜닝 | BM=BN=128, BK=32 → SMEM 32KB, thread 256개 |
| 07 | `07_gemm_warptiling.cu` | Warp Tiling | warp을 4×2로 명시 분할, bank conflict 감소 |
| 08 | `08_gemm_doublebuffer.cu` | Double Buffering | `__pipeline_memcpy_async`로 DMA와 연산 오버랩 |
| 09 | `09_gemm_cublas.cu` | cuBLAS FP32 | 비교 기준선 (100회 평균) |

### Tensor Core (TF32, WMMA API)

| # | 파일 | 핵심 기법 | 주요 포인트 |
|---|------|-----------|-------------|
| 10 | `10_gemm_tc_naive.cu` | 기본 WMMA | warp 1개 = 16×16 타일 1개, GMEM 직접 접근 |
| 11 | `11_gemm_tc_shared_memory.cu` | Shared Memory | GMEM → SMEM 후 fragment 로드 |
| 12 | `12_gemm_tc_warptiling.cu` | Fragment Tiling | warp 1개가 2×2 fragment 처리, arithmetic intensity 향상 |
| 13 | `13_gemm_tc_doublebuffer.cu` | Double Buffering | 버퍼 2개로 로드와 연산 오버랩 |
| 14 | `14_gemm_tc_vectorization.cu` | Vectorization | 128-bit 로드/스토어 |

각 Tensor Core 파일은 실행 시 cuBLAS TF32와 정확도(relative error)도 함께 출력합니다.

---

## 결과

`make run` 실행 후 `results/benchmark_<timestamp>.csv` 에 저장됩니다.

```
kernel,time_ms,tflops,cublas_pct
01_Naive,128.3,2.13,12.3
...
09_cuBLAS,15.8,17.32,100.0
10_TC_Naive,30.1,9.11,45.2
...
TC_cuBLAS_TF32,12.4,22.08,100.0
```

NCU 프로파일은 `make profile` 후 `results/profiles/` 에 `.ncu-rep` 형태로 저장됩니다.

```bash
ncu-ui results/profiles/01_naive.ncu-rep
```

---

## 주요 GPU 스펙 (RTX 4090, sm_89)

| 항목 | 값 |
|------|----|
| SM 수 | 128 |
| SM당 최대 SMEM | 48 KB |
| 블록당 최대 스레드 | 1024 |
| SM당 최대 레지스터 | 65536 |
| Warp 크기 | 32 |
| L2 캐시 | 72 MB |
| 메모리 버스 | 384-bit |

---

## 프로젝트 구조

```
gemm-optimization/
├── README.md
├── Makefile
├── run.sh
├── src/
│   ├── 01_gemm_naive.cu
│   ├── 02_gemm_coalesced.cu
│   ├── 03_gemm_shared_memory.cu
│   ├── 04_gemm_microtiling.cu
│   ├── 05_gemm_vectorization.cu
│   ├── 06_gemm_param_tune.cu
│   ├── 07_gemm_warptiling.cu
│   ├── 08_gemm_doublebuffer.cu
│   ├── 09_gemm_cublas.cu
│   ├── 10_gemm_tc_naive.cu
│   ├── 11_gemm_tc_shared_memory.cu
│   ├── 12_gemm_tc_warptiling.cu
│   ├── 13_gemm_tc_doublebuffer.cu
│   ├── 14_gemm_tc_vectorization.cu
│   └── utils_device_info.cu
├── include/
│   ├── gemm.h       ← CUDA Core 공통 (M/K/N 상수)
│   └── gemm_tc.h    ← Tensor Core 공통 (WMMA 상수, CUDA_CHECK, 초기화/검증)
├── scripts/
│   └── benchmark.sh
└── results/
    ├── benchmark_<timestamp>.csv   ← make run 후 생성
    └── profiles/                   ← make profile 후 생성
```