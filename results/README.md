# Results

## 벤치마크 결과

`make run` 실행 후 `benchmark_<timestamp>.csv` 로 저장됩니다.

```
kernel,time_ms,tflops,cublas_pct
01_Naive,128.3,2.13,12.3
...
09_cuBLAS,15.8,17.32,100.0
10_TC_Naive,30.1,9.11,45.2
...
TC_cuBLAS_TF32,12.4,22.08,100.0
```

## NCU 프로파일

`make profile` 실행 후 `profiles/` 에 `.ncu-rep` 파일이 생성됩니다.

```
results/
└── profiles/
    ├── 01_naive.ncu-rep
    ├── 02_coalesced.ncu-rep
    ├── 03_shared_memory.ncu-rep
    ├── 04_microtiling.ncu-rep
    ├── 05_vectorization.ncu-rep
    ├── 06_param_tune.ncu-rep
    ├── 07_warptiling.ncu-rep
    ├── 08_doublebuffer.ncu-rep
    ├── 09_cublas.ncu-rep
    ├── 10_tc_naive.ncu-rep
    ├── 11_tc_shared_mem.ncu-rep
    ├── 12_tc_warptiling.ncu-rep
    ├── 13_tc_doublebuffer.ncu-rep
    └── 14_tc_vectorization.ncu-rep
```