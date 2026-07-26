# results

`src/01_flash_tc_naive.cu`가 자기 벤치마크를 직접 CSV로 쓴다 (GEMM의
stdout-grep 방식보다 견고 -- 파싱 실수로 숫자가 밀릴 일이 없다).

## 파일

| 파일 | 생성 주체 | 컬럼 |
|---|---|---|
| `results_01_tc_naive_{noncausal,causal}.csv` | `01_flash_tc_naive` | `seq_len,time_ms,fma_ms,speedup_vs_fma,gflops` |
| `compare.csv` | `comparison/compare.py` | 01 vs. official, 고정 그리드 |

`fma_ms`/`speedup_vs_fma`는 같은 프로세스 안에서 같은 `seq_len`에 대해
`include/flash.h`의 tiled(FMA) 커널도 함께 돌려 잰 시간이다 -- GEMM이
cuBLAS를 프로세스 내 기준으로 삼는 것과 같은 역할(official은 PyTorch
확장이라 C++ 프로세스 안에서 직접 호출할 수 없으므로).

## 생성 방법

```bash
make run
# 또는
./run.sh          # 전체
./run.sh 1        # 01_flash_tc_naive 하나만
```

실행별 스냅샷은 `results/runs/<timestamp>/`에 남는다.

## NCU 프로파일

`make profile` 실행 후 `results/profiles/`에 `.ncu-rep` 파일이 생성된다.

```
results/profiles/
└── 01_tc_naive.ncu-rep
```
