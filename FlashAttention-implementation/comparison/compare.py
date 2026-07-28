"""Compare the project's kernel stages (forward only, non-causal) against
official FlashAttention v1.0.9 at matching config (batch=4, heads=8,
head_dim=64, fp16):
  - 00_attention_naive: standard attention -- no fusion, no tiling, no online
                        softmax; materializes the N*N score matrix in HBM
  - 01_flash_tc_naive : first WMMA version (KV tile=16, single buffer,
                        scalar sync loads) -- this project's starting point
  - 02_flash_tc_async : 01 + cp.async 16B vectorized GMEM->SMEM loads
                        (KV tile still 16, still single-buffered)
  - FA_official       : official FlashAttention v1.0.9, via this folder's own
                        flash_attn_unpadded_qkvpacked_func

Reuses each stage's already-generated CSV (no need to rebuild/rerun them),
and runs a matching-config benchmark here (seq_len=4096) for the official
kernel so the numbers you see are produced by this script's own code path.

Usage:
    python compare.py
"""

import os
import sys

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")

# official/ ships flash_attn as a local package (not pip-installed in this
# env), so it's only importable if official/ is on sys.path -- add it
# explicitly rather than relying on the caller's cwd/PYTHONPATH.
sys.path.insert(0, os.path.join(HERE, "..", "official"))

import torch
import pandas as pd
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

# stage name -> (csv filename, column label used in the merged table)
STAGES = [
    ("00_naive", "results_00_attention_naive_noncausal.csv"),
    ("01_tc_naive", "results_01_tc_naive_noncausal.csv"),
    ("02_tc_async", "results_02_tc_async_noncausal.csv"),
]

dtype = torch.float16
batch, heads, head_dim = 4, 8, 64  # BH = batch * heads = 32, matches DEFAULT_BH/DEFAULT_HEAD_DIM
seq_lens = [4096]


def bench(seq_len, iters=20, warmup=5):
    total = batch * seq_len
    qkv = torch.randn(total, 3, heads, head_dim, device="cuda", dtype=dtype)
    cu_seqlens = torch.arange(0, (batch + 1) * seq_len, step=seq_len, device="cuda", dtype=torch.int32)

    def call():
        return flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, seq_len, dropout_p=0.0, causal=False)

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()

    del qkv, cu_seqlens
    torch.cuda.empty_cache()
    return start.elapsed_time(end) / iters


def load_stage(name, filename):
    path = os.path.join(RESULTS_DIR, filename)
    df = pd.read_csv(path)[["seq_len", "time_ms"]].rename(columns={"time_ms": f"{name}_ms"})
    # -1 marks "skipped, would not fit in GPU memory" (see include/flash.h fits_in_gpu)
    df[f"{name}_ms"] = df[f"{name}_ms"].where(df[f"{name}_ms"] > 0)
    return df


if __name__ == "__main__":
    print(f"Config: batch={batch}, heads={heads}, head_dim={head_dim}, dtype={dtype}\n")

    official_rows = []
    for n in seq_lens:
        print(f"seq_len={n} ...", flush=True)
        official_rows.append({"seq_len": n, "official_ms": bench(n)})
    df = pd.DataFrame(official_rows)

    for name, filename in STAGES:
        df = df.merge(load_stage(name, filename), on="seq_len", how="left")
        df[f"{name}_vs_official_x"] = df[f"{name}_ms"] / df["official_ms"]

    print("\n=== forward pass, ms (lower is faster) -- baseline vs. official ===\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    df.to_csv(os.path.join(RESULTS_DIR, "compare.csv"), index=False)
