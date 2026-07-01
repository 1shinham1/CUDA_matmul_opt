# Results

`make run` 실행 후 `results/benchmark_<timestamp>.csv` 에 저장됩니다.

```
kernel,time_ms,tflops,cublas_pct
01_Naive,128.3,2.13,12.3
02_Coalesced,89.1,3.07,17.7
...
09_cuBLAS,15.8,17.32,100.0
```

`make profile` 실행 후 `profiles/` 디렉토리에 `.ncu-rep` 파일이 생성됩니다.

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
    └── 09_cublas.ncu-rep
```
