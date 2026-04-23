"""Benchmark Triton INT4 fused kernel vs pure-PyTorch dequant+matmul.

Measures end-to-end forward+backward latency of a Linear4bit layer at
LLaMA / Ministral QLoRA shapes with and without MUD_PUPPY_INT4_TRITON.
Target: 3-5x speedup on the 7900 XTX for typical QLoRA linear shapes
(attn q/k/v/o projections, MLP gate/up/down).

Usage:
    python3 benchmarks/bench_int4_triton.py
"""

from __future__ import annotations

import os
import time

import torch

from mud_puppy.bnb_rocm import Linear4bit


SHAPES = [
    # (in_features, out_features, batch*seq, label)
    (4096, 4096,  256, "llama-7b attn proj (B*S=256)"),
    (4096, 11008, 256, "llama-7b MLP up_proj (B*S=256)"),
    (5120, 5120,  512, "ministral-8b attn proj"),
    (5120, 14336, 512, "ministral-8b MLP up_proj"),
    (4096, 4096, 1024, "llama-7b attn proj (B*S=1024)"),
]

WARMUP = 5
ITERS = 30


def _bench(q: Linear4bit, x: torch.Tensor, use_triton: bool) -> dict:
    if use_triton:
        os.environ["MUD_PUPPY_INT4_TRITON"] = "1"
    else:
        os.environ.pop("MUD_PUPPY_INT4_TRITON", None)

    # Warmup
    for _ in range(WARMUP):
        x_ = x.detach().clone().requires_grad_(True)
        out = q(x_)
        out.sum().backward()
    torch.cuda.synchronize()

    # Forward only
    t0 = time.perf_counter()
    for _ in range(ITERS):
        with torch.no_grad():
            _ = q(x)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) / ITERS * 1000

    # Forward + backward
    t0 = time.perf_counter()
    for _ in range(ITERS):
        x_ = x.detach().clone().requires_grad_(True)
        out = q(x_)
        out.sum().backward()
    torch.cuda.synchronize()
    fwd_bwd_ms = (time.perf_counter() - t0) / ITERS * 1000

    return {"fwd_ms": fwd_ms, "fwd_bwd_ms": fwd_bwd_ms}


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA/HIP not available. Aborting.")
        return

    print(f"{'Shape':<45} {'fwd (pt/tr)':>18} {'fwd+bwd (pt/tr)':>22} {'fwd×':>6} {'bwd×':>6}")
    print("-" * 110)

    for K, N, M, label in SHAPES:
        torch.manual_seed(0)
        lin = torch.nn.Linear(K, N, bias=False)
        q = Linear4bit(lin, dtype=torch.bfloat16).cuda()
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        pt = _bench(q, x, use_triton=False)
        tr = _bench(q, x, use_triton=True)

        fwd_speedup = pt["fwd_ms"] / tr["fwd_ms"]
        fwdbwd_speedup = pt["fwd_bwd_ms"] / tr["fwd_bwd_ms"]
        tag = f"K={K} N={N} M={M}"
        print(
            f"{label:<45} "
            f"{pt['fwd_ms']:>6.2f}/{tr['fwd_ms']:<6.2f}ms   "
            f"{pt['fwd_bwd_ms']:>6.2f}/{tr['fwd_bwd_ms']:<6.2f}ms   "
            f"{fwd_speedup:>5.2f}x {fwdbwd_speedup:>5.2f}x"
        )

    print()
    print("Note: pt = pure-PyTorch dequant+linear; tr = Triton fused kernel.")


if __name__ == "__main__":
    main()
