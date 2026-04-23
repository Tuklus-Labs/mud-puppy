"""End-to-end QLoRA step benchmark with Triton INT4 kernel on vs off.

Simulates one training step on a transformer block stack shaped like
Ministral-3-14B (hidden=5120, intermediate=14336, 40 layers) and measures
wall-clock step time with and without MUD_PUPPY_INT4_TRITON=1. This is
the headline number that matters to training throughput.

Usage:
    python3 benchmarks/bench_qlora_step.py
    python3 benchmarks/bench_qlora_step.py --layers 40 --batch 4 --seq 256
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from mud_puppy.bnb_rocm import Linear4bit


class QLoRABlock(nn.Module):
    """One Ministral-style transformer block with Linear4bit base + LoRA.

    We only care about the linear layers (which dominate step time); the
    attention kernel itself is unchanged from the pytorch path, so we
    skip softmax/rope and just do 4 attn-projection linears + 3 MLP
    linears. That's representative of what the Triton kernel accelerates.
    """

    def __init__(self, hidden: int, intermediate: int, lora_r: int = 16):
        super().__init__()
        # Attn projections
        self.q = Linear4bit(nn.Linear(hidden, hidden, bias=False), dtype=torch.bfloat16)
        self.k = Linear4bit(nn.Linear(hidden, hidden, bias=False), dtype=torch.bfloat16)
        self.v = Linear4bit(nn.Linear(hidden, hidden, bias=False), dtype=torch.bfloat16)
        self.o = Linear4bit(nn.Linear(hidden, hidden, bias=False), dtype=torch.bfloat16)
        # MLP
        self.gate = Linear4bit(nn.Linear(hidden, intermediate, bias=False), dtype=torch.bfloat16)
        self.up = Linear4bit(nn.Linear(hidden, intermediate, bias=False), dtype=torch.bfloat16)
        self.down = Linear4bit(nn.Linear(intermediate, hidden, bias=False), dtype=torch.bfloat16)

        # LoRA A+B on every linear (trainable, bf16)
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.randn(lora_r, hidden,       dtype=torch.bfloat16) * 0.02),  # q
            nn.Parameter(torch.randn(lora_r, hidden,       dtype=torch.bfloat16) * 0.02),  # k
            nn.Parameter(torch.randn(lora_r, hidden,       dtype=torch.bfloat16) * 0.02),  # v
            nn.Parameter(torch.randn(lora_r, hidden,       dtype=torch.bfloat16) * 0.02),  # o
            nn.Parameter(torch.randn(lora_r, hidden,       dtype=torch.bfloat16) * 0.02),  # gate
            nn.Parameter(torch.randn(lora_r, hidden,       dtype=torch.bfloat16) * 0.02),  # up
            nn.Parameter(torch.randn(lora_r, intermediate, dtype=torch.bfloat16) * 0.02),  # down
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden,       lora_r, dtype=torch.bfloat16)),
            nn.Parameter(torch.zeros(hidden,       lora_r, dtype=torch.bfloat16)),
            nn.Parameter(torch.zeros(hidden,       lora_r, dtype=torch.bfloat16)),
            nn.Parameter(torch.zeros(hidden,       lora_r, dtype=torch.bfloat16)),
            nn.Parameter(torch.zeros(intermediate, lora_r, dtype=torch.bfloat16)),
            nn.Parameter(torch.zeros(intermediate, lora_r, dtype=torch.bfloat16)),
            nn.Parameter(torch.zeros(hidden,       lora_r, dtype=torch.bfloat16)),
        ])

    def _linear_with_lora(self, layer, x, a, b):
        base = layer(x)
        # LoRA: x @ A.T @ B.T  (rank r, so shape goes through [*, r])
        lora = (x @ a.T) @ b.T
        return base + lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attn-ish: q, k, v projections combined into one sum (stand-in for attn output)
        q_out = self._linear_with_lora(self.q, x, self.lora_A[0], self.lora_B[0])
        k_out = self._linear_with_lora(self.k, x, self.lora_A[1], self.lora_B[1])
        v_out = self._linear_with_lora(self.v, x, self.lora_A[2], self.lora_B[2])
        # Synthetic attention: just average qkv (skip softmax; not what we benchmark)
        attn = (q_out + k_out + v_out) / 3.0
        o_out = self._linear_with_lora(self.o, attn, self.lora_A[3], self.lora_B[3])
        x = x + o_out
        # MLP (SwiGLU-ish)
        g = self._linear_with_lora(self.gate, x, self.lora_A[4], self.lora_B[4])
        u = self._linear_with_lora(self.up,   x, self.lora_A[5], self.lora_B[5])
        h = torch.nn.functional.silu(g) * u
        d = self._linear_with_lora(self.down, h, self.lora_A[6], self.lora_B[6])
        return x + d


def bench(n_layers: int, batch: int, seq: int, hidden: int, intermediate: int,
          use_triton: bool, warmup: int, iters: int) -> float:
    if use_triton:
        os.environ["MUD_PUPPY_INT4_TRITON"] = "1"
    else:
        os.environ.pop("MUD_PUPPY_INT4_TRITON", None)

    blocks = nn.ModuleList([
        QLoRABlock(hidden, intermediate) for _ in range(n_layers)
    ]).cuda()

    opt = torch.optim.AdamW(
        [p for p in blocks.parameters() if p.requires_grad],
        lr=1e-4,
    )

    def step():
        x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.bfloat16,
                        requires_grad=False)
        for blk in blocks:
            x = blk(x)
        loss = x.float().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=8,
                   help="Number of transformer blocks (default 8; full Ministral = 40)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--intermediate", type=int, default=14336)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA/HIP device. Abort.")
        return

    print(f"config: {args.layers} layers, batch={args.batch}, seq={args.seq}, "
          f"hidden={args.hidden}, intermediate={args.intermediate}")
    print(f"warmup={args.warmup}, iters={args.iters}")
    print()

    pt = bench(args.layers, args.batch, args.seq, args.hidden, args.intermediate,
               use_triton=False, warmup=args.warmup, iters=args.iters)
    # Fresh allocator to not mix memory footprints across runs.
    torch.cuda.empty_cache()
    tr = bench(args.layers, args.batch, args.seq, args.hidden, args.intermediate,
               use_triton=True, warmup=args.warmup, iters=args.iters)

    print(f"pytorch path:   {pt*1000:>8.1f} ms/step")
    print(f"triton path:    {tr*1000:>8.1f} ms/step")
    print(f"speedup:        {pt/tr:>8.2f}x")
    if args.layers < 40:
        extrap = pt / tr
        print(f"(extrapolated to full 40-layer Ministral: same {extrap:.2f}x ratio)")


if __name__ == "__main__":
    main()
