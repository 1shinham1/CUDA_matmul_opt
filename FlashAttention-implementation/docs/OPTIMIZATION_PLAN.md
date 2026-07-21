# FA_with_cuda WMMA forward kernel: warp-tiling + double-buffering

**Outcome**: implemented largely as planned (warp-tiling to BLOCK_N=64,
STAGES=2 double-buffered K/V), with one deviation from §5 below —
`cuda::pipeline`/`memcpy_async` was tried but dropped in favor of plain
`__syncthreads()`-based double buffering, because it kept tripping
`racecheck`/`synccheck` hazards when combined with this kernel's
lane-divergent softmax section (cp.async fallback vs. WMMA read conflicts,
mbarrier arrival-count mismatches). See the top-of-file comment in
`flash_kernels_tc.cuh` for the full reasoning; the shared-memory layout,
warp-tiling, and verification steps below are otherwise unchanged from what
shipped.

## Context

`flash_fwd_kernel_tc` (the tensor-core/WMMA FlashAttention forward kernel in
`include/flash_kernels_tc.cuh`) had too little tensor-core work between
synchronization points. The goal of this pass was to improve that kernel's
hardware utilization without changing its public launcher interface.

Root cause (established via code reading + a live `-Xptxas -v` build): the
kernel processes only **one 16×16 WMMA tile of keys per outer-loop
iteration** (`BLOCK_N=16`), so only 8 `mma_sync` calls happen between two
`__syncthreads()` + three `__syncwarp()` calls — sync/dispatch overhead
dominates actual tensor-core work. Separately, K/V tile loads and compute are
fully serial (load → sync → compute → sync), so copy latency is never hidden.

This pass fixes exactly those two things — **warp tiling** (widen the KV
chunk from 16 to 64 columns, so 4× more `mma_sync` work happens per sync
point) and **double buffering** (overlap next-tile K/V load with
current-tile compute via `cuda::pipeline`) — in the **forward kernel only**
(`flash_fwd_kernel_tc` + its launcher `flash_forward_launch_tc`), for
**HEAD_DIM=64 only** (the config the benchmarks actually use). Backward
kernels and other head_dims are explicitly out of scope for this pass — a
follow-up decision once this is measured.

Both optimizations, and the `cuda::pipeline`/`cuda::memcpy_async` mechanism
specifically, were validated ahead of time: a standalone repro (WMMA +
`cuda::pipeline`, including a `seq_len` boundary/tail case) was compiled with
this exact toolchain (CUDA 13.1, `nvcc -arch=sm_89 -std=c++17`) and run on
this RTX 4090, passing `compute-sanitizer --tool memcheck|racecheck|synccheck`
with 0 errors and bit-exact expected output. Device limits were queried live:
`sharedMemPerBlockOptin=101376` bytes, `sharedMemPerMultiprocessor=102400`.

**Honest expectation** (from the design agent's analysis, not just optimism):
this should land the forward kernel at roughly **~27-38% of Triton** (a
~2.5-3.5× wall-clock speedup over today's TC forward kernel) — a real,
worthwhile jump, but not 80%. The remaining gap after this pass is the
16-of-32-lane-idle softmax section (explicitly deferred), the S/P
round-trip through shared memory forced by the opaque `nvcuda::wmma` API
(vs. raw PTX/CUTLASS), and no warp-tiling across the key dimension (all 4
warps in a block still redundantly share one K/V tile, splitting only query
rows). Occupancy will also roughly halve (2 resident blocks/SM → 1, per the
new 81,920-byte shared-memory footprint vs. today's 34,816) — a real,
partially-offsetting cost that should be measured, not assumed away.

## Plan

All changes are confined to `flash_fwd_kernel_tc` and `flash_forward_launch_tc`
in **`flash_kernels_tc.cuh`**. Public signature
`flash_forward_launch_tc<HEAD_DIM>(Q,K,V,O,L,BH,seq_len,causal)` stays
unchanged — `tests/test_flash_tc.cu` and `src/benchmark_tensor_core.cu`
keep compiling as-is.
Add `static_assert(HEAD_DIM == 64, ...)` (or similar) so instantiating with
another head_dim fails loudly at compile time instead of overflowing shared
memory silently at runtime, since this pass doesn't implement a fallback for
other sizes.

### 1. Shared memory layout (HEAD_DIM=64, BLOCK_N=64, 2 pipeline stages)

| Region | Shape | Bytes |
|---|---|---|
| `Qs` | `[BLOCK_M=64][64]` half | 8,192 (unchanged) |
| `Ks` | `[2 stages][BLOCK_N=64][64]` half | 16,384 |
| `Vs` | same | 16,384 |
| `Stile` (per-warp scratch) | `[16][64]` float | 16,384 |
| `Ptile` (per-warp scratch) | `[16][64]` half | 8,192 |
| `Otile` (per-warp scratch) | `[16][64]` float | 16,384 (unchanged — accumulate in registers before one store) |
| **Total** | | **81,920 B**, vs. 101,376 cap (19,456 B margin) |

Only `Ks`/`Vs` need the ×2 stage dimension — `Qs` loads once at kernel entry
(never overwritten mid-loop) and the per-warp scratch tiles are fully
produced-and-consumed within one iteration by the same warp.

### 2. S = Q·Kᵀ — widen from 1 to 4 sub-tiles per iteration

Loop `n` over 4 KV sub-tiles (each still `HD_TILES` `mma_sync` calls over the
head_dim reduction, unchanged trip count per sub-tile); `b_frag` load gets a
`+ n*16*HEAD_DIM` row offset into the current stage's `Ks` buffer. The
existing "read K as `col_major` to get Kᵀ for free" trick is untouched — it's
local to each 16-row window. Store each sub-tile's result into
`my_Stile + n*16` (row-major, ld=64). `mma_sync` count per iteration:
`4 × HD_TILES` (was `1 × HD_TILES`).

### 3. Online softmax — same recurrence, wider inner loop

One combined pass over all 64 columns (not 4 chained 16-wide updates — that
would defeat the "more work per sync" goal and change rounding order).
`scores[]` grows from `float[16]` to `float[64]`; still only `lane_id < 16`
active (explicitly deferred inefficiency — flag in code comment).
Mathematically identical online-softmax recurrence; only summation order
changes, well within the existing `tol=3e-2` in `test_flash_tc.cu`.

### 4. P·V — reduce over 4 sub-chunks per head-dim slice

For each of `HD_TILES` head-dim output slices, loop `sc` over 4 KV
sub-chunks, loading `p_frag` from `my_Ptile + sc*16` (ld=64) and `v_frag`
from the current stage's `Vs + sc*16*HEAD_DIM + nn*16`, accumulating into one
`o_frag` across all 4 `sc` before a single `store_matrix_sync` per `nn` (so
`Otile` doesn't need to grow). `mma_sync` count: `HD_TILES × 4` (was
`HD_TILES × 1`).

### 5. `cuda::pipeline` double buffering (2 stages)

```
#include <cuda/pipeline>
#include <cooperative_groups.h>
```
`cuda::pipeline_shared_state<cuda::thread_scope_block, 2>` in shared memory,
`auto pipeline = cuda::make_pipeline(cg::this_thread_block(), &pstate)`.

Per-tile load helper (uniform across all threads, called by the whole
block):
- Normal (non-boundary) tile: `producer_acquire()` → two
  `cuda::memcpy_async(...)` calls (K and V, full `BLOCK_N*HEAD_DIM*sizeof(half)`
  bytes) → `producer_commit()`.
- Boundary tile (`kv0 + BLOCK_N > seq_len`, i.e. the tail when `seq_len %
  BLOCK_N != 0`): `producer_acquire()` → `memcpy_async` only the `n_valid =
  seq_len - kv0` valid rows → **plain stores** (not `memcpy_async`) to
  zero-fill the remaining rows, followed by an explicit `__syncthreads()`
  before `producer_commit()`. The plain zero-fill stores aren't tracked by
  `memcpy_async`'s arrival-count mechanism, so don't rely on
  `consumer_wait()` alone for their visibility — this exact pattern
  (mixed memcpy_async + plain store + explicit sync) was validated in the
  pre-check repro under `racecheck`/`synccheck`.

Main loop: prologue prefetches tile 0 before the loop starts; each iteration
issues the *next* tile's load before `consumer_wait()` on the *current*
tile (so the wait overlaps with the just-issued copy), does the S/softmax/PV
compute (§2-4) against the current stage buffer, then `consumer_release()`.
The loop's own `if (t+1 < n_tiles)` guard is the natural epilogue — no
separate drain needed. `Qs`'s one-shot load at kernel entry stays a plain
cooperative copy (not worth pipelining a single load).

### 6. Launcher (`flash_forward_launch_tc`)

Update the dynamic shared-memory size calculation to match the new layout in
§1 (still via the existing `cudaFuncSetAttribute(...,
cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` pattern — same mechanism,
new byte count).

## Verification

1. `make test_flash_tc && bin/test_flash_wmma` — all 7 cases
   must print `OK` (fp64 CPU reference **and** FMA-kernel cross-check,
   `tol=3e-2`), including both `seq_len=130` (non-multiple-of-64 boundary)
   cases and both causal cases. Note this test also exercises the (untouched)
   backward TC kernels, which depend on `O_tc`/`L_tc` from the forward
   kernel, so a full run (not a forward-only subset) is required.
2. `compute-sanitizer --tool memcheck bin/test_flash_wmma` — 0 errors.
3. `compute-sanitizer --tool racecheck bin/test_flash_wmma` — 0 errors (the
   critical check for double-buffering hazards: wrong
   `producer_acquire`/`consumer_release` sequencing would show up here as a
   WAR/WAW hazard on the `Ks`/`Vs` stage buffers).
4. `compute-sanitizer --tool synccheck bin/test_flash_wmma` — 0 errors (new
   check, relevant because of the mixed pipeline + explicit `__syncthreads()`
   in the boundary-tile path).
5. `nvcc -Xptxas -v` on the rebuilt kernel — confirm no unexpected register
   spills (baseline today: 123 regs/thread, 0 spills).
6. `make benchmark_tc && bin/benchmark_tensor_core && bin/benchmark_tensor_core causal` —
   regenerate `results_tc_noncausal.csv` / `results_tc_causal.csv`.

### Critical files
- `include/flash_kernels_tc.cuh` (all changes)
- `tests/test_flash_tc.cu`, `src/benchmark_tensor_core.cu` (run unmodified, for verification)
