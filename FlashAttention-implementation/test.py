#!/usr/bin/env python3
"""kernel 1~3 정확도 검증.

    python test.py

판정 기준은 공식 flash-attention 저장소와 같다:

    |kernel - ref_fp32|_max  <=  2 * |torch_b16 - ref_fp32|_max

즉 "같은 dtype 으로 순진하게 계산한 torch 보다 2배 이상 나쁘지 않으면 통과".
b16 누적 오차가 절대 임계값보다 훨씬 크기 때문에 이런 상대 기준을 쓴다.
"""

import math
import sys

import torch
#my_flash.cpython-310-x86_64-linux-gnu.so 호출
import my_flash 

KERNELS = [
    (1, "Base Implementation"),
    (2, "Swizzling"),
    (3, "Eagerly Loading K & V Blocks"),
]


def attention_ref(q, k, v, upcast: bool):
    """(B, S, H, D) 입력에 대한 non-causal attention 참조 구현."""
    dtype = torch.float32 if upcast else q.dtype

    # (B, S, H, D) -> (B, H, S, D)
    q_ = q.transpose(1, 2).to(dtype)
    k_ = k.transpose(1, 2).to(dtype)
    v_ = v.transpose(1, 2).to(dtype)

    scale = 1.0 / math.sqrt(q.shape[-1])
    s = (q_ @ k_.transpose(-1, -2)) * scale
    p = torch.softmax(s.float(), dim=-1).to(dtype)
    o = p @ v_

    return o.transpose(1, 2).contiguous()


def run_case(dtype, batch, seq_len, n_heads, d_head, device="cuda"):
    torch.manual_seed(0)
    shape = (batch, seq_len, n_heads, d_head)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    ref = attention_ref(q, k, v, upcast=True).float()
    pt = attention_ref(q, k, v, upcast=False).float()
    baseline = (pt - ref).abs().max().item()
    tol = max(2.0 * baseline, 1e-4)

    tag = f"{str(dtype).replace('torch.', ''):8s} b={batch} s={seq_len:<6d} h={n_heads} d={d_head}"
    print(f"\n  {tag}   (torch b16 오차 {baseline:.3e}, 허용 {tol:.3e})")

    outs = {}
    ok = True
    
    #같은 저정밀도 dtype으로 순진하게 돌린 PyTorch보다 2배 이상 나쁘지 않으면 통과
    for n, name in KERNELS:
        fn = getattr(my_flash, f"forward_{n}")
        out, _ = fn(q, k, v, None, False)
        torch.cuda.synchronize()

        err = (out.float() - ref).abs().max().item()
        finite = torch.isfinite(out).all().item()
        passed = finite and err <= tol
        ok &= passed

        outs[n] = out
        status = "PASS" if passed else "FAIL"
        extra = "" if finite else "  <-- NaN/Inf 발생"
        print(f"    kernel {n}  {status}   max err {err:.3e}"
              f"   ({err / baseline if baseline else 0:.2f}x){extra}   {name}")

    # 커널끼리도 서로 거의 같아야 한다 (셋 다 같은 수식이므로 오차가 거의 동일)
    ref_out = outs[1].float()
    for n, _ in KERNELS[1:]:
        d = (outs[n].float() - ref_out).abs().max().item()
        if d > tol:
            print(f"    [경고] kernel {n} 이 kernel 1 과 {d:.3e} 만큼 다릅니다")
            ok = False

    return ok


def main():
    if not torch.cuda.is_available():
        print("CUDA GPU 가 필요합니다")
        return 1

    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  (SM {major}.{minor})")
    if major < 8:
        print("SM_80 이상이 필요합니다")
        return 1

    d_head = 128
    cases = [
        # dtype, batch, seq_len, n_heads
        (torch.float16, 2, 256, 4),
        (torch.float16, 1, 512, 8),
        (torch.float16, 2, 1024, 4),
        (torch.float16, 1, 2048, 8),
        (torch.bfloat16, 2, 256, 4),
        (torch.bfloat16, 1, 1024, 8),
        (torch.bfloat16, 1, 2048, 4),
    ]

    all_ok = True
    for dtype, batch, seq_len, n_heads in cases:
        all_ok &= run_case(dtype, batch, seq_len, n_heads, d_head)

    print("\n" + ("=" * 72))
    print("전체 통과" if all_ok else "실패한 케이스가 있습니다")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
