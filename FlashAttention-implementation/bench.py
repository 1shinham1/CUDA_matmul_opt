#!/usr/bin/env python3
"""kernel 1~3 성능 측정.

    python bench.py                    # seq_len 512~8192
    python bench.py --seq-len 4096
    python bench.py --dtype bf16
    python bench.py --seq-len 4096 --iters 200   # 반복 늘려 노이즈 줄이기
    python bench.py --dtype bf16 --seq-len 4096 --iters 200 
기준선은 PyTorch 의 scaled_dot_product_attention (FLASH 백엔드) 이다.
"""

import argparse
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
#my_flash.cpython-310-x86_64-linux-gnu.so 호출
import my_flash

KERNELS = [
    (1, "Base Implementation"),
    (2, "Swizzling"),
    (3, "Eager K & V Loads"),
]

# seq_len 이 커져도 총 작업량이 비슷하도록 batch 를 줄인다.
BATCH_FOR_SEQ = {512: 32, 1024: 16, 2048: 8, 4096: 4, 8192: 2, 16384: 1}
N_HEADS = 16
D_HEAD = 128


def attn_flops(batch, seq_len, n_heads, d_head):
    """non-causal forward: QK^T 와 PV 두 번의 행렬곱."""
    return 4 * batch * n_heads * seq_len * seq_len * d_head


def time_fn(fn, n_warmup=10, n_iters=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / n_iters


def bench_seq_len(seq_len, dtype, n_iters):
    batch = BATCH_FOR_SEQ.get(seq_len, 1)
    shape = (batch, seq_len, N_HEADS, D_HEAD)

    torch.manual_seed(0)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    o = torch.empty_like(q)

    flops = attn_flops(batch, seq_len, N_HEADS, D_HEAD)



    # 기준선: torch SDPA (flash backend). (B, S, H, D) -> (B, H, S, D)
    qt, kt, vt = (t.transpose(1, 2).contiguous() for t in (q, k, v))

    def ref():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(qt, kt, vt, is_causal=False)

    ref_ms = time_fn(ref, n_iters=n_iters)
    ref_tflops = flops / (ref_ms * 1e-3) / 1e12

    print(f"\n  seq_len = {seq_len:<6d} batch = {batch:<3d} heads = {N_HEADS}"
          f"  d_head = {D_HEAD}  dtype = {str(dtype).replace('torch.', '')}")
    print(f"  {'커널':<26s} {'ms':>9s} {'TFLOPs':>9s} {'vs SDPA':>9s}")
    print("  " + "-" * 56)

    for n, name in KERNELS:
        fn = getattr(my_flash, f"forward_{n}")
        ms = time_fn(lambda: fn(q, k, v, o, False), n_iters=n_iters)
        tflops = flops / (ms * 1e-3) / 1e12
        print(f"  {n}. {name:<23s} {ms:9.3f} {tflops:9.1f} "
              f"{100 * ref_ms / ms:8.1f}%")

    print("  " + "-" * 56)
    print(f"  {'0. torch SDPA (flash)':<26s} {ref_ms:9.3f} {ref_tflops:9.1f} "
          f"{100.0:8.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    major, minor = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  (SM {major}.{minor}, {props.multi_processor_count} SMs)")

    seq_lens = [args.seq_len] if args.seq_len else [512, 1024, 2048, 4096, 8192]
    for s in seq_lens:
        bench_seq_len(s, dtype, args.iters)


if __name__ == "__main__":
    main()
