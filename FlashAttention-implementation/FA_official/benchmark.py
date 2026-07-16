"""Speed test for FlashAttention v1.0.9 (FA_official), sweeping the exact
configuration used in the paper's own benchmark grid:
  - head_dim in {64, 128}, hidden_dim = 2048 (heads = hidden_dim / head_dim)
  - seq_len in {512, 1024, 2048, 4096, 8192, 16384}
  - batch_size = 16384 / seq_len (so batch * seq_len is constant)

FA1 has no plain flash_attn_func(q, k, v, causal=...) like FA2 -- it's
varlen-only, so qkv is packed as one tensor with cu_seqlens marking batch
boundaries (here every sequence has the same length).

Usage:
    python benchmark.py
"""

import torch
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

dtype = torch.float16
hidden_dim = 2048
head_dims = [64, 128]
seq_lens = [512, 1024, 2048, 4096, 8192, 16384]


def bench(batch, heads, seq_len, head_dim, iters=30, warmup=5):
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


if __name__ == "__main__":
    print(f"{'head_dim':>8} {'heads':>6} {'seq_len':>8} {'batch':>6} {'ms/iter':>10}")
    rows = []
    for head_dim in head_dims:
        heads = hidden_dim // head_dim
        for seq_len in seq_lens:
            batch = 16384 // seq_len
            ms = bench(batch, heads, seq_len, head_dim)
            rows.append((head_dim, heads, seq_len, batch, ms))
            print(f"{head_dim:>8} {heads:>6} {seq_len:>8} {batch:>6} {ms:>10.4f}")
