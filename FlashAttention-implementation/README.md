# FlashAttention-implementation

A reproduction of FlashAttention (Dao, Fu, Ermon, Rudra, Ré — 2022) in plain
C++/CUDA, benchmarked against both naive attention and the paper's own
reference implementation on an RTX 4090:

- **[FA_with_cuda/](FA_with_cuda/)** — hand-written `__global__` kernels
  compiled with `nvcc`, no PyTorch, no Triton, verified against a
  from-scratch double-precision CPU reference (every kernel, FMA and WMMA,
  forward and backward, passed on the first real GPU run). Q/K/V/O(/dQ/dK/dV/dO)
  are fp16 (matching the paper), everything else fp32. Three layered
  comparisons, each isolating one effect: naive vs. flash (algorithm alone,
  ~2.5×), deferred-normalization vs. literal-Algorithm-1/2
  (FlashAttention-2's contribution, ~1.2–1.5×), and FMA-loop vs.
  `nvcuda::wmma` tensor cores — forward *and* backward, ~1.8–3.9× —
  referenced against [FA_official/](FA_official/), the paper's own
  CUTLASS-based repo. Stacked together, naive→WMMA is 4–11× faster than
  naive. See that directory's README for the full breakdown and what it
  shows about the paper's Section 5 point on why writing fast IO-aware
  kernels by hand is hard.
- **[FA_official/](FA_official/)** — the paper's own reference implementation
  (cloned for comparison), used to check how the real CUDA kernels use
  tensor cores (CUTLASS warp-level MMA / raw PTX `mma.sync`).
- **[compare/](compare/)** — head-to-head benchmarks of `FA_with_cuda`'s WMMA
  kernel directly against `FA_official`, at matching configs:
  **~2.5–13× slower** than the real FA1 kernel, worst around seq_len
  2048–4096.

Each subdirectory with its own results has a README with the full
results/tables and a "how to run" section.
