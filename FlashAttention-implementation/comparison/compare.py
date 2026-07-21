"""Compare your two FlashAttention implementations at the exact same config
(batch=4, heads=8, head_dim=64, fp16, forward pass, non-causal):
  - FA_with_cuda   : hand-written WMMA tensor-core CUDA kernel
  - FA_official    : official FlashAttention v1.0.9, via this folder's own
                     flash_attn_unpadded_qkvpacked_func call

Reuses the already-generated CSV for FA_with_cuda (no need to rebuild/rerun
that), and runs a matching-config sweep here for the official kernel so the
numbers you see are produced by this script's own code path.

Usage:
    python compare.py
"""

import os

import torch
import pandas as pd
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func  #official API 가져오

#csv 경로 만들기
HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
CUDA_CSV = os.path.join(RESULTS_DIR, "results_tc_noncausal.csv")

# 비교 조건 설정
dtype = torch.float16
batch, heads, head_dim = 4, 8, 64 #BH = batch × heads = 32
seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def bench(seq_len, iters=20, warmup=5):
    total = batch * seq_len
    #QKV 생성, shape [batch × seq_len, 3, heads, head_dim], 3은 1개의 tensor로 묶었다는 뜻
    qkv = torch.randn(total, 3, heads, head_dim, device="cuda", dtype=dtype)

    #batch=4, seq_len=128이면:
    #cu_seqlens = [0, 128, 256, 384, 512]
    #batch 0: qkv[0:128]
    #batch 1: qkv[128:256]
    #batch 2: qkv[256:384]
    #batch 3: qkv[384:512]
    cu_seqlens = torch.arange(0, (batch + 1) * seq_len, step=seq_len, device="cuda", dtype=torch.int32)

    #official API 호출 정의
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
    print(f"Config: batch={batch}, heads={heads}, head_dim={head_dim}, dtype={dtype}\n")

    #모든 seq_len에서 official 실행, sequence length마다 앞에서 본 bench(n)을 실행
    official_rows = []
    for n in seq_lens:
        print(f"seq_len={n} ...", flush=True)
        official_rows.append({"seq_len": n, "official_ms": bench(n)})
    official = pd.DataFrame(official_rows)


    #내 CUDA CSV 읽기
    # [sequence length, 내 Tensor Core forward 시간]
    cuda = pd.read_csv(CUDA_CSV)[["seq_len", "tc_fwd_ms"]].rename(columns={"tc_fwd_ms": "fa_with_cuda_ms"})

    df = official.merge(cuda, on="seq_len")
    df["cuda_vs_official_x"] = df["fa_with_cuda_ms"] / df["official_ms"]

    print("\n=== forward pass, ms (lower is faster) ===")
    #화면 출력
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    #최종 CSV 저장
    df.to_csv(os.path.join(RESULTS_DIR, "compare.csv"), index=False)
