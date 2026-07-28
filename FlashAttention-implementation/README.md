# FlashAttention — Tensor Core baseline에서 다시 시작

FlashAttention **forward pass**를 처음부터 다시 최적화해나가는 프로젝트다.
`CUDA_matmul_opt`(GEMM 최적화 프로젝트)와 같은 패턴을 따른다 —
`src/NN_*.cu` 파일 하나가 최적화 기법 하나이고, 커널 정의 + fp64 CPU
레퍼런스 검증 + 벤치마크가 전부 그 파일 안에 자기완결적으로 들어있다.

backward는 다루지 않는다.

---

## 환경 요구사항

| 항목 | 값 |
|---|---|
| CUDA Toolkit | 13.x (`nvcc`가 PATH에 있어야 함) |
| GPU | RTX 4090 (`sm_89`) — 다른 아키텍처는 `make ARCH=sm_80` 등으로 지정 |
| OpenMP | fp64 CPU 레퍼런스 병렬화용 (`-Xcompiler -fopenmp`) |

official과 비교하려면(`comparison/`) PyTorch + `flash_attn`(FlashAttention
v1.0.9)이 설치된 Python 환경이 추가로 필요하다.

---

## 빌드 & 실행

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cuda_env   # nvcc가 이 환경에 있음
cd FlashAttention-implementation
make all              # bin/01_flash_tc_naive 빌드
make run              # 전체 빌드 + 벤치마크 실행 -> results/results_*.csv
./run.sh              # make run과 동일
./run.sh 1            # 01_flash_tc_naive 하나만 실행 (correctness + benchmark)
./run.sh 1 causal     # causal 모드로 실행
make profile          # NCU 프로파일 -> results/profiles/*.ncu-rep
make clean            # bin/, results/profiles/ 삭제
```

실행할 때마다:
1. **정확성 검증** — 작은 크기(`seq_len≈128~130`, causal/non-causal)에서
   fp64 CPU 레퍼런스(`include/flash.h`의 `ref_forward`, OpenMP 병렬화 +
   `bin/.cpu_ref_*.bin` 캐싱)와 상대오차를 비교해 OK/WARNING을 출력한다.
2. **성능 벤치마크** — `seq_len = 128..32768` sweep, `BH=32`(batch×heads),
   `head_dim=64` 고정. 결과를 `results/results_01_tc_naive_*.csv`에 직접
   쓴다. 같은 프로세스 안에서 `include/flash.h`의 tiled(FMA) 커널도 같은
   `seq_len`에 대해 함께 돌려 "vs FMA" speedup을 리포트한다 — official은
   PyTorch 확장이라 C++ 프로세스 안에서 직접 호출할 수 없으므로, GEMM에서
   cuBLAS가 맡는 "프로세스 내 빠른 기준" 역할을 tiled 커널이 대신한다.

---

## 현재 baseline: `01_flash_tc_naive.cu`

Q/K/V를 shared memory 타일로 순회하는 tiling + online softmax(Algorithm 2)
구조 위에서, Q@Kᵀ·P@V 두 매트멀을 `nvcuda::wmma`로 텐서 코어에 올린 첫
버전이다. KV 타일=16(서브타일 1개), 워프 1개=쿼리 16행 담당, 더블버퍼링
없음(K/V를 매번 동기적으로 로드→sync→연산).

**이미 적용된 것**: 텐서 코어(WMMA), shared memory 타일링, online softmax.
**아직 적용 안 된 것 — 다음 최적화 후보**:

| 기법 | 설명 |
|---|---|
| Warp-tiling | KV 타일을 16→64로 넓혀 sync 지점당 텐서 코어 작업량을 늘림 |
| Double buffering | K/V를 2-stage 버퍼로 두어 로드와 연산을 오버랩 |
| Shared-memory swizzling | XOR 스와즐로 bank conflict 회피 |
| 벡터화 로드 | `uint4` 등으로 global→shared 로드를 128비트 단위로 묶음 |
| 레지스터 블로킹 | V를 레지스터에 상주시켜 shared memory 재읽기를 줄임 |
| K/V 버퍼 공유 | K와 V가 같은 shared memory 공간을 시분할로 재사용 (smem footprint 절감) |

official(`official/`, FlashAttention v1.0.9)은 이 여섯 가지를 전부
갖추고 있고(더블버퍼링은 Q 타일 대상, K/V는 `SHARE_SMEM_FOR_K_AND_V`로
공유), 그래서 이 baseline과 6~10배 격차가 난다 — 격차 상세는
[`comparison/`](comparison/README.md) 참고.

---

## 결과

`make run` 실행 후 `results/results_*.csv`에 저장된다. 아래는 RTX 4090,
`BH=32, head_dim=64, non-causal` 기준 실측치 (`make run`으로 재측정 가능).

### Correctness

`seq_len∈{128,130}`, causal on/off — fp64 CPU 레퍼런스와 상대오차
`<3e-2`로 일치(`OK`), tiled(FMA) 커널과도 상대오차 `<3e-2`로 교차 검증됨.

### baseline vs. tiled(FMA) 커널

| seq_len | tiled (ms) | tc_naive (ms) | speedup vs FMA |
|---:|---:|---:|---:|
| 1024 | 2.07 | 0.59 | 3.49× |
| 4096 | 28.84 | 8.32 | 3.48× |
| 8192 | 115.82 | 40.41 | 2.91× |
| 32768 | 1818.02 | 587.68 | 3.10× |

### official과의 격차

| seq_len | official (ms) | tc_naive (ms) | tc_naive vs official |
|---:|---:|---:|---:|
| 1024 | 0.08 | 0.59 | 7.71× |
| 4096 | 1.07 | 8.31 | 7.80× |
| 8192 | 6.15 | 40.42 | 6.57× |
| 32768 | 104.36 | 586.42 | 5.62× |

official과의 직접 비교는 [`comparison/`](comparison/README.md)에서
`compare.py`로 재측정할 수 있다.

---

## 프로젝트 구조

```
FlashAttention-implementation/
├── README.md
├── Makefile
├── run.sh
├── include/
│   ├── flash.h        # 공용: 매크로, CPU fp64 레퍼런스+캐싱+검증,
│   │                   # DeviceAllocTracker, tiled(FMA) 커널
│   └── flash_tc.h      # Tensor Core 공용: WMMA 상수, align16 (flash.h 포함)
├── src/
│   └── 01_flash_tc_naive.cu   # 현재 baseline
├── scripts/
│   └── benchmark.sh
├── comparison/          # baseline vs. official FA1 head-to-head
├── official/            # 비교용 FlashAttention v1.0.9 (vendor, 무변경)
└── results/
```

`bin/`은 빌드 산출물(gitignored)이라 목록에서 뺐다.
