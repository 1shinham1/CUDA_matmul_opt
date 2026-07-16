"""Compare FA_with_cuda vs the official FlashAttention v1.0.9 (FA_official)
on the paper's own benchmark grid (head_dim in {64, 128}, hidden_dim=2048 ->
heads = hidden_dim/head_dim, seq_len in {512..16384}, batch = 16384/seq_len),
forward pass, non-causal.

FA_with_cuda's WMMA kernel only supports head_dim=64 (static_assert in
flash_kernels_tc.cuh -- see OPTIMIZATION_PLAN.md), so its head_dim=128 rows
are left blank (N/A) rather than skipping the comparison entirely.

Usage:
    python compare_paper_grid.py
"""

import os

import torch
import pandas as pd
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

HERE = os.path.dirname(__file__)
CUDA_CSV = os.path.join(HERE, "..", "FA_with_cuda", "results_tc_paper_grid.csv")

dtype = torch.float16
hidden_dim = 2048
head_dims = [64, 128]
seq_lens = [512, 1024, 2048, 4096, 8192, 16384]


def bench_official(batch, heads, seq_len, head_dim, iters=20, warmup=5):
    total = batch * seq_len
    qkv = torch.randn(total, 3, heads, head_dim, device="cuda", dtype=dtype)
    cu_seqlens = torch.arange(0, (batch + 1) * seq_len, step=seq_len, device="cuda", dtype=torch.int32)

    def call():
        return flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, seq_len, dropout_p=0.0, causal=False)

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()
    del qkv, cu_seqlens
    torch.cuda.empty_cache()
    return start.elapsed_time(end) / iters


if __name__ == "__main__":
    cuda_df = pd.read_csv(CUDA_CSV)  # head_dim=64 only

    rows = []
    for head_dim in head_dims:
        heads = hidden_dim // head_dim
        for seq_len in seq_lens:
            batch = 16384 // seq_len
            print(f"head_dim={head_dim} seq_len={seq_len} ...", flush=True)

            official_ms = bench_official(batch, heads, seq_len, head_dim)

            if head_dim == 64:
                match = cuda_df[cuda_df["seq_len"] == seq_len]
                cuda_ms = float(match["tc_fwd_ms"].iloc[0]) if len(match) else None
            else:
                cuda_ms = None  # FA_with_cuda: head_dim=128 not implemented

            rows.append({
                "head_dim": head_dim, "heads": heads, "seq_len": seq_len, "batch": batch,
                "official_ms": official_ms,
                "fa_with_cuda_ms": cuda_ms,
            })

    df = pd.DataFrame(rows)
    print("\n=== forward pass, ms (lower is faster; FA_with_cuda blank = head_dim=128 not implemented) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    df.to_csv(os.path.join(HERE, "compare_paper_grid.csv"), index=False)
