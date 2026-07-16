# FlashAttention CUDA Learning Project — Session Handoff

Read this first when starting a new session in this directory — it's a one-time
context transfer from a previous conversation that ran at the old location
(`/home/shshin/Flashattention`, before the project was moved here). Delete
this file once you've absorbed it; it's not meant to be permanent project docs.

## Goal

The user is hand-writing CUDA FlashAttention kernels (`FA_with_cuda`) to learn
tensor-core GEMM/attention optimization, and wants to progressively close the
performance gap against the real, production FlashAttention-1 implementation
(`FA_official`) — specifically **not** against Triton, which was deliberately
ruled out as a comparison target (different compilation model, so its
optimizations don't transfer to hand-written CUDA the way FA_official's do).

## Directory layout (here, at `/home/shshin/MATMUL/FlashAttention-implementation/`)

- **`FA_official/`** — vendored FlashAttention **v1.0.9** (FA1 — not FA2/FA3,
  which is what `pip install flash-attn` installs today). Trimmed down to only:
  `flash_attn/__init__.py`, `flash_attn/flash_attn_interface.py` (the varlen
  API actually used), `csrc/flash_attn/fmha_api.cpp` + `csrc/flash_attn/src/`
  (the real FMHA CUDA kernel — this is the study material), the compiled
  `flash_attn_cuda.cpython-310-x86_64-linux-gnu.so`, `benchmark.py` (paper-grid
  sweep), and `README.md`/`LICENSE`/`AUTHORS`/`BUILD.md`/`setup.py`. Unrelated
  bundled content (layer_norm, fused_dense, xentropy, rotary, ft_attention,
  full GPT/BERT/LLaMA model code, an alternate Triton implementation,
  block-sparse variant, CI config) was deliberately deleted as clutter
  unrelated to studying the FMHA kernel or to `import flash_attn` working.

- **`FA_with_cuda/`** — the user's own hand-written kernels: naive (no tensor
  cores, O(N²) memory) → FMA-based flash attention (tiled, no tensor cores) →
  WMMA tensor-core flash attention (`flash_kernels_tc.cuh`, warp-tiled +
  double-buffered per `OPTIMIZATION_PLAN.md`, **HEAD_DIM=64 only** via
  `static_assert`). Also contains `gemm_tc/` — a standalone TF32 tensor-core
  GEMM microbenchmark (separate from the FlashAttention kernels) exploring
  shared-memory bank-conflict swizzling.

- **`compare/`** — scripts benchmarking `FA_with_cuda` against `FA_official`
  at matching configs. Triton was stripped out of these per the user's
  decision (see Goal above) — only `FA_with_cuda` vs `FA_official` now.

## Key environment facts (all already handled, but relevant if anything breaks)

- GPU: RTX 4090 (sm_89, Ada). System-wide CUDA is 13.1 (`/usr/local/cuda`),
  but a conda env **`cuda_env`** (torch 2.11+cu128, Python 3.10) is used for
  `FA_official`'s Python side. It needed a *matching* CUDA 12.8 toolkit
  installed *inside* `cuda_env` (`conda install -n cuda_env -c nvidia
  cuda-toolkit=12.8.1 -y`) because `torch.utils.cpp_extension` hard-fails on a
  CUDA major-version mismatch between `nvcc` and the torch build.
- `FA_official` is Ada-incompatible upstream (only targets sm_75/80/90) — its
  `setup.py` was patched to add `-gencode arch=compute_89,code=sm_89`.
- cutlass (FA1's build dependency, pinned at commit
  `319a389f42b776fae5701afcb943fc03be5b5c25`, confirmed via the GitHub API for
  tag v1.0.9) was vendored manually, then deleted after building to save 88MB.
  Full rebuild steps (including this) are documented in `FA_official/BUILD.md`
  — read that before trying to rebuild the `.so`.
- The compiled `.so` (198MB) is gitignored, as are compiled `FA_with_cuda`
  binaries are *not* gitignored by design — the user wanted to handle that
  exclusion by hand rather than have it automated.

## Benchmark results so far

(batch=4, heads=8, head_dim=64, fp16, forward pass, non-causal, unless noted)

- **`gemm_tc/tc_swizzle.cu`** (pure GEMM, vs cuBLAS): 83.2% @ 2048³, 75.8% @
  4096³, 55.1% @ 8192³ — efficiency degrades with size. `ncu` profiling showed
  occupancy is flat (~16.6%, register-limited to 2 blocks/SM — *not* the cause
  of the size-dependent decline) while L2 hit rate drops (92%→86.7%) and DRAM
  traffic rises as size grows. Diagnosed cause: no L2-aware threadblock
  rasterization/swizzle — the `dim_grid(M/128, N/128)` launch order isn't
  grouped for L2 reuse the way cuBLAS/CUTLASS do it. **Not yet fixed.**
- **`FA_with_cuda` TC kernel vs `FA_official`** (fixed BH=32 grid, seq_len
  128→32768): `FA_with_cuda` is **2.3x–10x slower**, worst around seq_len=4096
  (see `compare/compare.csv`).
- Same comparison on the *paper's own grid* (head_dim 64/128, hidden_dim=2048,
  batch=16384/seq_len — so total batch×heads is much larger at small
  seq_len): gap widens to **13x–17x** (see `compare/compare_paper_grid.csv`).
  `FA_with_cuda`'s head_dim=128 rows are blank/N-A — the kernel only supports
  HEAD_DIM=64.

## Diagnosed remaining bottlenecks in `FA_with_cuda`'s TC kernel

Already done (per `OPTIMIZATION_PLAN.md`): warp-tiling + double-buffering
(widened KV chunk, `cuda::pipeline`-based async load) for the forward kernel,
HEAD_DIM=64 only.

Still deferred / candidate next steps, roughly in likely-impact order:
1. **S/P round-trips through shared memory** because `nvcuda::wmma` is an
   opaque API — `FA_official` avoids this via raw `mma.sync` (CUTLASS 2.x
   primitives, see `FA_official/csrc/flash_attn/src/fmha/gemm.h`).
2. **No warp-tiling across the key/KV dimension** — all 4 warps in a block
   redundantly share one K/V tile, only splitting query rows.
3. **Softmax section only uses 16 of 32 lanes** per warp (half idle).
4. **No L2-aware block rasterization** (same class of issue found in
   `gemm_tc`, likely applies here too, untested).

## Suggested reading order through `FA_official`'s kernel

(for pulling techniques into `FA_with_cuda`)

1. `flash_attn/flash_attn_interface.py` — API shape, quick skim
2. `csrc/flash_attn/fmha_api.cpp` — dispatch logic (head_dim/dtype/causal →
   which pre-compiled kernel instantiation gets called)
3. `csrc/flash_attn/src/fmha.h`, `fmha_kernel.h`, `fmha/kernel_traits.h`,
   `fmha_fwd_launch_template.h` — tile-size constants, grid/block dims
4. `csrc/flash_attn/src/fmha_fprop_kernel_1xN.h` — **the main event**, the
   forward kernel body. Compare directly against `flash_fwd_kernel_tc` in
   `FA_with_cuda/flash_kernels_tc.cuh`.
5. `csrc/flash_attn/src/fmha/{gmem_tile,smem_tile,gemm,softmax,mask}.h` —
   building blocks. `smem_tile.h` has the real bank-conflict swizzle logic,
   directly comparable to `gemm_tc/tc_swizzle.cu`'s `swizzle_a`/`swizzle_b`.
6. `csrc/flash_attn/src/fmha_dgrad_kernel_1xN_loop.h` — backward (Algorithm
   4), read after forward is understood.

Note: FA1 handles head_dim=32/64/128 via separate, individually-validated
`FMHA_kernel_traits<...>` instantiations per head_dim (compare
`fmha_fwd_hdim128.cu` vs `fmha_fwd_hdim64.cu`) — this is the *same* templating
mechanism `FA_with_cuda` already uses (`template<int HEAD_DIM>`), just not yet
extended past HEAD_DIM=64.

## Outstanding / unresolved items

- **Git push is blocked.** A commit (`a0ad3f7`, "Add FlashAttention CUDA/Triton
  learning project...") was made at the *old* location
  (`/home/shshin/Flashattention`, which still has a bare `.git` there — no
  working files, just history — with `origin` pointing at
  `https://github.com/1shinham1/FlashAttention-implementation.git`) but was
  never pushed. Push failed: no GitHub auth configured anywhere in this
  environment (no credential helper, no SSH key, no `gh` CLI, no token env
  var). The user hasn't decided how to authenticate yet.
- **This directory (`FlashAttention-implementation/`) has no git history of
  its own** — the move here deliberately left `.git` behind at the old
  location (user's explicit choice) rather than moving it, so this is
  currently a plain, ungitted directory sitting inside the separate `MATMUL`
  git repo (`/home/shshin/MATMUL`, which also contains `CUDA_matmul_opt/`).
  It has *not* been `git add`ed there yet.
- **11 CSV result files** (`FA_with_cuda/results_*.csv`, `compare/*.csv`)
  were deliberately left uncommitted at the user's request (wanted to review
  before adding to git) — they moved here along with everything else but
  aren't tracked in any git history yet.
- The orphaned `/home/shshin/Flashattention/.git` was intentionally left in
  place, not deleted — the user may want to reconcile/reconnect it with this
  new location, or abandon it, still undecided.

## User preferences learned this session

- Wants terse, direct answers grounded in actual measured data, not
  speculation — always run the numbers rather than estimate.
- Prefers hand-written CUDA (not Triton) as the comparison/learning target,
  since techniques transfer directly to what's being built.
- Cares about git hygiene: build artifacts (`.so`, compiled binaries)
  gitignored; CSV benchmark results tracked/committed (not gitignored) since
  they're small, diffable, and meaningful as a progress record; editor config
  (`.vscode/`) gitignored.
- Wants LICENSE/AUTHORS preserved when vendoring FA1 (BSD-3-Clause requires
  retaining the copyright notice on redistribution).
