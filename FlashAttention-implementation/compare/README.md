# compare — FA_with_cuda vs. FA_official, head-to-head

Everything in [`FA_with_cuda/README.md`](../FA_with_cuda/README.md) compares
`FA_with_cuda`'s own kernels against each other (naive vs. flash, FMA vs.
WMMA). This directory is the one place that benchmarks
`FA_with_cuda`'s WMMA forward kernel directly against
[`FA_official/`](../FA_official/) — the paper's real, CUTLASS-based
FlashAttention-1 kernel — at matching configs, forward pass, non-causal,
fp16.

## Results (RTX 4090)

### 1. Fixed grid (`compare.py` → `compare.csv`): batch=4, heads=8, head_dim=64

| seq_len | official (ms) | FA_with_cuda (ms) | FA_with_cuda / official |
|---:|---:|---:|---:|
| 128 | 0.015 | 0.036 | 2.46× |
| 256 | 0.014 | 0.063 | 4.39× |
| 512 | 0.024 | 0.205 | 8.57× |
| 1024 | 0.077 | 0.701 | 9.12× |
| 2048 | 0.275 | 2.722 | 9.89× |
| 4096 | 1.065 | 10.683 | 10.03× |
| 8192 | 6.155 | 59.426 | 9.66× |
| 16384 | 25.658 | 197.246 | 7.69× |
| 32768 | 104.348 | 711.564 | 6.82× |

### 2. Paper's own grid (`compare_paper_grid.py` → `compare_paper_grid.csv`)

hidden_dim=2048, `heads = hidden_dim / head_dim`, `batch = 16384 / seq_len` —
so total parallel work (batch×heads) shrinks as seq_len grows, unlike the
fixed grid above. `FA_with_cuda`'s WMMA kernel only supports HEAD_DIM=64
(`static_assert` in `flash_kernels_tc.cuh`), so head_dim=128 rows have no
`FA_with_cuda` number.

| head_dim | heads | seq_len | batch | official (ms) | FA_with_cuda (ms) | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 32 | 512 | 32 | 0.658 | 8.434 | 12.82× |
| 64 | 32 | 1024 | 16 | 1.511 | 15.168 | 10.04× |
| 64 | 32 | 2048 | 8 | 3.089 | 29.768 | 9.64× |
| 64 | 32 | 4096 | 4 | 6.456 | 59.072 | 9.15× |
| 64 | 32 | 8192 | 2 | 12.777 | 117.991 | 9.24× |
| 64 | 32 | 16384 | 1 | 25.875 | 196.475 | 7.59× |
| 128 | 16 | * | * | measured | N/A | — head_dim=128 not implemented |

## Reading the numbers

Both grids land in the same ballpark: `FA_with_cuda` is **roughly 2.5–13×
slower** than the real FA1 kernel, worst in the low-thousands seq_len range
and best at the extremes (smallest seq_len, where fixed per-launch overhead
dominates both equally; largest seq_len, where `FA_with_cuda`'s relative
disadvantage narrows). The paper grid's ratios sit slightly higher than the
fixed grid's at matching seq_len (e.g. 9.15× vs 10.03× at seq_len=4096 — close,
but the paper grid's shrinking batch as seq_len grows gives `FA_official`'s
more mature occupancy/scheduling relatively more room to show its advantage).

This is the same kernel documented in
[`FA_with_cuda/README.md`](../FA_with_cuda/README.md#3-tensor-cores-benchmark_tc-the-big-lever)
and analyzed for remaining bottlenecks in
[`FA_with_cuda/OPTIMIZATION_PLAN.md`](../FA_with_cuda/OPTIMIZATION_PLAN.md) —
the gap to `FA_official` here is the same class of missing optimizations
(S/P round-trip through shared memory forced by the opaque `nvcuda::wmma`
API vs. `FA_official`'s raw `mma.sync`, no warp-tiling across the key
dimension, half-idle softmax lanes, no L2-aware rasterization), not a
different problem.

## How to run

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cuda_env   # has flash_attn (FA_official) installed
cd compare
python compare.py              # fixed batch=4/heads=8/head_dim=64 grid -> compare.csv
python compare_paper_grid.py   # paper's own grid, head_dim in {64,128}  -> compare_paper_grid.csv
```

Both scripts reuse `FA_with_cuda`'s already-generated
`results_tc_noncausal.csv` / `results_tc_paper_grid.csv` (run
`./benchmark_tc` in `FA_with_cuda/` first if those don't exist yet) and only
benchmark `FA_official` fresh, via its own `flash_attn_unpadded_qkvpacked_func`
call path.
