# FlashAttention — plain C/CUDA version

Plain C++/CUDA, no PyTorch, no Triton — hand-written `__global__` kernels
compiled with `nvcc`, tested against a self-contained double-precision CPU
reference. Naive forward → naive backward → FlashAttention forward →
FlashAttention backward → benchmark comparison, then two follow-up
experiments (a literal-Algorithm-1/2 kernel, and a tensor-core kernel).

**Precision**: `Q, K, V, O, dQ, dK, dV, dO` are stored as `__half` (fp16),
matching the paper's actual training setup (Apex/PyTorch AMP). Everything
else — softmax, the online-softmax running statistics, the `S/P/dP/dS`
scratch buffers in the naive kernels — stays fp32, the same "store fp16,
compute fp32" split PyTorch AMP uses.

## Files

| File | Contents |
|---|---|
| `include/common.cuh` | Error-checking macros, a byte-tracking device allocator (`DeviceAllocTracker`), a CUDA-event timer, fp16↔fp32 conversion helpers (`to_half`/`to_float`/`snap_to_half`), and the double-precision CPU reference (`ref_forward`/`ref_backward`) used as ground truth for every test. |
| `include/naive_kernels.cuh` | Standard attention (Algorithm 0 forward / Algorithm 3 backward) as **unfused** kernels: `naive_scores_kernel` (S=QKᵀ), `naive_softmax_kernel`, `naive_output_kernel` (O=PV), then `naive_dv/dp/ds/dq/dk_kernel` for the backward. Every intermediate (S, P, dP, dS) is its own fp32 `N×N` buffer in global memory — no shared memory, no tiling. |
| `include/flash_kernels.cuh` | FlashAttention forward (`flash_fwd_kernel`) and backward (`flash_bwd_dkdv_kernel` + `flash_bwd_dq_kernel`) as fused, tiled kernels with explicit `__shared__` memory tiles and the online-softmax recurrence (plain scalar FMA loops, no tensor cores). Also contains `flash_fwd_kernel_strict`, which normalizes every K/V tile instead of deferring to the end (see "Literal Algorithm 1/2" below). |
| `include/flash_kernels_tc.cuh` | The tensor-core forward (`flash_fwd_kernel_tc`) and backward (`flash_bwd_dkdv_kernel_tc` + `flash_bwd_dq_kernel_tc`) kernels, using CUDA's `nvcuda::wmma` API — see "Tensor cores" below. |
| `tests/*.cu` | Correctness tests: GPU output vs. the fp64 CPU reference, across shapes/causal/non-causal/non-block-aligned N. |
| `src/*.cu` | Algorithm, normalization, Tensor Core, and paper-grid benchmarks. |
| `Makefile` | `make all` builds everything into `bin/` with `nvcc -O3 -arch=sm_89`. |

## Build & run

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cuda_env   # has nvcc
cd FlashAttention-implementation
make all
./run.sh                    # correctness, all vs. fp64 CPU reference
make run                    # all benchmark variants -> results/results_*.csv
```

## Results (RTX 4090, batch·heads=32, head_dim=64)

### Correctness

All three test binaries pass every case (causal/non-causal, non-block-aligned
seq_len, head_dim 32/64/128) against the fp64 CPU reference. Typical fp16
storage rounding error, no algorithmic error:

```
BH=2 seq_len=  130 d= 64 causal=1 | O 0.000938 dQ 0.000479 dK 0.000672 dV 0.000959  OK          (naive)
BH=2 seq_len=  130 d= 64 causal=1 | O 0.000938 dQ 0.000529 dK 0.000672 dV 0.000959  OK          (flash FMA)
BH=2 seq_len=  130 d= 64 causal=1 | fwd: TC vs CPU-ref 0.000938 | TC vs FMA-kernel 0.000977  OK  (flash WMMA fwd)
  bwd: dQ ref=0.000738 fma=0.000977 | dK ref=0.000672 fma=0.000977 | dV ref=0.001202 fma=0.001953  OK
```

### 1. naive vs. flash (no tensor cores): O(N²) vs. O(N)

| N | naive fwd (ms) | flash fwd (ms) | speedup | naive mem (MB) | flash mem (MB) | mem ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 71.61 | 28.85 | 2.48× | 4224 | 128.5 | 32.9× |
| 8192 | 290.39 | 116.28 | 2.50× | 16640 | 257 | 64.8× |
| 16384 | OOM | 452.57 | — | — | 514 | — |
| 32768 | OOM | 1805.55 | — | — | 1028 | — |

Runtime speedup here (~2.5×) reflects the pure IO-awareness effect in
isolation, since neither kernel uses tensor cores yet — see the "Tensor
cores" experiment below for what happens once they're added.

fp16 vs. fp32 storage (holding the algorithm fixed) has almost no effect on
*speed* here — 28.03ms (fp32) vs. 28.85ms (fp16) forward at N=4096 — but
**exactly halves memory** (256.5MB → 128.5MB), since flash's memory is
dominated by the O(N) `Q,K,V,O` buffers, which are now 2 bytes/element
instead of 4. Storage format alone doesn't buy speed without tensor cores;
it only buys memory. That's the setup for the next two experiments.

### 2. Literal Algorithm 1/2 vs. deferred normalization (`benchmark_norm`)

The paper's Algorithm 1/2 pseudocode normalizes and writes `O_i` to HBM at
*every* K/V tile (line 12/15), not just once at the end — that "defer to the
end" optimization is what FlashAttention-2 calls out as its own contribution
(fewer non-matmul FLOPs). `flash_fwd_kernel_strict` implements the literal
per-tile version; `flash_fwd_kernel` defers. Same final result (verified),
timed separately:

| N | deferred (ms) | strict/literal (ms) | slowdown |
|---:|---:|---:|---:|
| 512 | 0.52 | 0.78 | 1.49× |
| 4096 | 27.99 | 35.48 | 1.27× |
| 32768 | 1766.26 | 2141.34 | 1.21× |

Deferring normalization alone is worth ~20–50%. Small next to tensor cores
(below), but not free — a real, measured slice of FlashAttention-2's
contribution, isolated from everything else.

### 3. Tensor cores (`benchmark_tc`): the big lever

`flash_fwd_kernel_tc` / `flash_bwd_dkdv_kernel_tc` / `flash_bwd_dq_kernel_tc`
replace the FMA-loop matmuls with `nvcuda::wmma` (CUDA's built-in tensor-core
API — see "Tensor cores" below), keeping the same tiling / online-softmax /
recomputation structure otherwise. Both forward *and* backward are covered.

**Forward only:**

| N | naive (ms) | flash FMA (ms) | flash WMMA (ms) | WMMA vs. naive | WMMA vs. FMA |
|---:|---:|---:|---:|---:|---:|
| 1024 | 4.82 | 2.06 | 0.55 | 8.79× | 3.76× |
| 4096 | 71.66 | 28.81 | 8.19 | 8.75× | 3.52× |
| 8192 | 290.19 | 114.90 | 41.12 | 7.06× | 2.79× |
| 32768 | OOM | 1806.26 | 577.89 | — | 3.13× |

**Forward + backward (training-realistic):**

| N | naive (ms) | flash FMA (ms) | flash WMMA (ms) | WMMA vs. naive | WMMA vs. FMA |
|---:|---:|---:|---:|---:|---:|
| 1024 | 11.27 | 6.44 | 2.45 | 4.60× | 2.63× |
| 4096 | 178.28 | 101.81 | 44.33 | 4.02× | 2.30× |
| 8192 | OOM | 394.51 | 182.04 | — | 2.17× |
| 32768 | OOM | 6293.30 | 2520.02 | — | 2.50× |

Causal (skips upper-triangular tiles, compounds with tensor cores in
forward; the backward WMMA kernels *don't* implement the causal-tile-skip
optimization the FMA kernels have — see caveat below — so the causal
WMMA-vs-FMA ratio for fwd+bwd is *smaller* than non-causal, the one place
in this whole project where a "more optimized" version is intentionally
still behind):

| N | naive fwd+bwd (ms) | flash FMA fwd+bwd (ms) | flash WMMA fwd+bwd (ms) | WMMA vs. naive | WMMA vs. FMA |
|---:|---:|---:|---:|---:|---:|
| 4096 | 151.58 | 57.55 | 31.00 | 4.89× | 1.86× |
| 8192 | OOM | 212.98 | 120.51 | — | 1.77× |
| 32768 | OOM | 3240.41 | 1707.20 | — | 1.90× |

Switching on tensor cores alone (same algorithm, same tiling, same fp16
storage — only the matmuls change) is worth **~1.8–3.9×** on top of the
FMA-loop flash kernels, forward and backward included. Combined with the
algorithmic win over naive, the WMMA kernels are **4–11× faster than
naive**, versus the FMA kernels' ~2.5×. The remaining gap to a production
kernel comes from what a real compiler/library does that
this hand-written WMMA code doesn't (software pipelining/double-buffering of
shared-memory loads against compute, better occupancy tuning, possibly warp
specialization, and — specifically for backward — the causal-skip
optimization noted above that the WMMA kernels here are missing).

## Tensor cores: what changed and why it works

`flash_fwd_kernel_tc` uses **`nvcuda::wmma`** (`<mma.h>`, included with the
CUDA toolkit), so no additional matrix-multiplication library is required.
It is a documented, higher-level C++ interface to Tensor Core tile operations.

**Why the kernel had to be restructured.** `flash_fwd_kernel` (FMA version)
uses "1 CUDA thread = 1 query row." WMMA can't work that way: a `wmma`
instruction is a *warp-collective* operation — all 32 threads of a warp
jointly compute one 16×16 tile via tensor cores; there's no such thing as
one thread doing a WMMA matmul alone. So `flash_fwd_kernel_tc` uses "1 warp
= 16 query rows, 4 warps = 64 rows/block" instead. The two matmuls
(`Q·Kᵀ` accumulated over `head_dim` in chunks of 16, `P·V` accumulated over
16 key positions per tile) run on all 32 threads of a warp via
`wmma::load_matrix_sync` / `mma_sync` / `store_matrix_sync`. Softmax itself
(row max/sum, `exp`, the online-softmax rescale) isn't a matmul, so it gets
no benefit from tensor cores; that part runs on only 16 of each warp's 32
lanes (one lane per row this warp owns), reading/writing a small
per-warp `__shared__` scratch tile that the WMMA calls write to/read from —
this scratch relay through shared memory is required because WMMA
accumulator fragments are scattered across a warp's registers in a
hardware-defined layout, not addressable "row `i`, lane `i`."

**The `K` transpose is free.** `Q·Kᵀ` needs `K` transposed, but `K` is
stored row-major as `[seq_len][head_dim]`. Loading that same memory with
`wmma::load_matrix_sync(..., wmma::col_major)` reinterprets it as `Kᵀ`
directly — no physical transpose, just a different fragment layout
declaration over identical bytes. (`P·V` needs no such trick: `P` and `V`
are already in the right orientation for `wmma::row_major`.)

**Backward has twice the matmuls, and reuses the same free-transpose trick
twice more.** Forward needs 2 matmuls per tile (`Q·Kᵀ`, `P·V`); backward
needs 4: `S=Q·Kᵀ` and `dP=dO·Vᵀ` (recomputed, same shape/trick as forward's
`Q·Kᵀ`), then `dV=Pᵀ·dO` and `dK=dSᵀ·Q`. The latter two need `P` and `dS`
*transposed* — both are already sitting in `__shared__` as `[Q-row][KV-col]`
row-major (the natural layout they're produced in), so exactly the same
`wmma::col_major`-reinterpretation trick used for `K` in forward gives `Pᵀ`
and `dSᵀ` for free again, no physical transpose anywhere in this codebase.
`dQ=dS·K` needs no transpose at all (both operands already in the right
orientation). Two kernels, split the same way as the FMA backward (one warp
= 16 K/V rows for `dK,dV`; one warp = 16 Q rows for `dQ`, mirroring
forward) — same reasoning as before: each output row is written by exactly
one warp, so no atomics.

**One caveat**: the FMA backward kernels skip K/V-tiles that are entirely
above the causal diagonal (`flash_bwd_dkdv_kernel`'s `m_lo` check); the WMMA
backward kernels here don't implement that skip (they mask element-by-element
instead, always doing the full matmul work). That's why the causal
WMMA-vs-FMA advantage for backward is smaller than forward's — a concrete,
measured example of "the tensor-core matmul is faster per-tile, but skipping
whole tiles you don't need is a separate optimization that has to be
re-earned in the new kernel structure."

## Implementation notes

- **Naive kernels are deliberately unfused**, matching Algorithm 0/3: separate
  kernels for S, softmax, O (forward) and dV/dP/dS/dQ/dK (backward), each
  reading/writing its own `N×N` (or `N×d`) buffer in global memory. This is
  the baseline the paper argues against, not an attempt at a fast reference.
- **Flash backward** is split into two kernels:
  `flash_bwd_dkdv_kernel` (one thread per K/V row, loops over
  Q tiles) and `flash_bwd_dq_kernel` (one thread per Q row, loops over K/V
  tiles, mirrors forward) — each output tensor is written by exactly one
  thread, so no atomics anywhere. Recomputes `S_ij`/`P_ij` from `Q,K,V` and
  the saved logsumexp `L`, exactly like Appendix B.2/B.4. `flash_kernels_tc.cuh`
  has WMMA versions of both (`_tc` suffix), same two-kernel split, "one warp
  = 16 rows" instead of "one thread = 1 row" (see "Tensor cores" above).
- **`HEAD_DIM` and tile widths are C++ template parameters** (not runtime
  values) so per-row accumulators (`acc[HEAD_DIM]`, `scores[BLOCK_N]`) can
  live in registers. Shared memory is allocated dynamically (`extern
  __shared__` + `cudaFuncSetAttribute(...,
  cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`) since several configs
  (`head_dim=128`, the WMMA kernel's multi-region scratch layout) exceed the
  48KB static-`__shared__` limit.
- **OOM handling in the benchmarks** is a pre-flight `cudaMemGetInfo` check
  (with a 20% safety margin) rather than try/catch-style recovery; once
  naive is infeasible at some N it's marked infeasible for all larger N.

## Takeaways

- The **algorithm** (tiling + online softmax + recomputation) gives a
  ~17–65× memory reduction and pushes the OOM boundary out (naive OOMs by
  N=16384, flash keeps scaling to N=32768) — that part is "free" once you've
  translated the paper correctly, independent of the hardware-utilization
  work below.
- The **wall-clock speedup is not free** and decomposes into separable,
  measurable pieces: deferred normalization (~1.2–1.5×, FlashAttention-2's
  contribution), and tensor cores (~1.8–3.9×, by far the largest lever here,
  forward and backward both). Stacking algorithm + deferred-norm +
  tensor-cores takes forward+backward from ~1.75× faster than naive (FMA) to
  ~4–5× faster than naive (WMMA) — most of the remaining gap to a production
  kernel is compiler/library-level scheduling (pipelining,
  occupancy, warp specialization) that hand-tuned CUTLASS does and this
  hand-written WMMA code doesn't attempt, plus the specific missing
  causal-tile-skip in the backward WMMA kernels noted above.
- **Forward and backward needed separate tensor-core kernels**, and backward
  needed twice as many matmuls (4 vs. 2) — but every transpose either kernel
  needed (`Kᵀ` in forward; `Pᵀ`, `dSᵀ` in backward) turned out to be free via
  `wmma::col_major` reinterpretation of already-row-major `__shared__` data,
  with zero physical transposes anywhere in the codebase. That pattern
  generalizing cleanly across both passes was the main thing that made
  backward tractable to add after forward already worked.
- This is a fairly direct, measured illustration of the paper's own Section 5
  point ("Compiling to CUDA"): writing a *correct* IO-aware kernel by hand is
  very achievable (both the FMA and WMMA versions here passed every
  correctness test on the first real run); writing a *fast* one requires
  progressively more of what compiler-level scheduling and libraries like
  CUTLASS automate — tiling and softmax fusion get you the algorithm,
  tensor cores get you most of the remaining hardware throughput, and the
  last mile is scheduling (plus optimizations like causal-tile-skipping that
  have to be re-earned in each new kernel structure) that's genuinely hard
  to hand-write correctly and completely.
