#!/usr/bin/env python3
"""mud-puppy v0.4 throughput benchmark.

Measures tokens/sec for the three training modes introduced in v0.4:
  - baseline   : standard full-param training
  - packing    : sequence packing (PackedCollator)
  - streaming  : per-layer GPU streaming (LayerStreamer)

Usage::

    # Quick smoke test (tiny model, synthetic data)
    python bench/throughput.py --model gpt2 --seq-len 256 --steps 20

    # Full benchmark with all three modes
    python bench/throughput.py --model meta-llama/Llama-3-8B \\
        --seq-len 2048 --steps 50 --modes baseline packing streaming

Output format (stdout, one JSON blob per mode)::

    {"mode": "baseline", "tokens_per_sec": 12340.5, "steps": 50, ...}

Environment
-----------
ROCm/CUDA: auto-detected via torch.cuda.is_available().
Set ``HIP_VISIBLE_DEVICES`` or ``CUDA_VISIBLE_DEVICES`` to pin a device.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("bench.throughput")


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: Any,
) -> Dict[str, Any]:
    """Return a minimal training batch on *device*."""
    import torch

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _make_packed_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: Any,
) -> Dict[str, Any]:
    """Return a packed-format batch with block-diagonal attention mask."""
    import torch
    from mud_puppy.packing import PackedCollator

    # Create short variable-length examples to pack
    rng = torch.Generator()
    rng.manual_seed(42)
    examples = []
    for _ in range(batch_size * 4):  # overprovision; collator bins-packs
        length = torch.randint(32, seq_len // 2, (1,), generator=rng).item()
        ids = torch.randint(0, vocab_size, (length,))
        examples.append({
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones(length, dtype=torch.long),
        })

    collator = PackedCollator(max_seq_length=seq_len)
    batch = collator(examples)
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _benchmark_mode(
    mode: str,
    model: Any,
    batch_fn,
    steps: int,
    warmup: int,
    batch_size: int,
    seq_len: int,
    device: Any,
) -> Dict[str, Any]:
    """Run *steps* forward+backward passes and return timing stats."""
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    def _one_step() -> float:
        batch = batch_fn(batch_size, seq_len, model.config.vocab_size, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    # Warmup (not timed)
    for _ in range(warmup):
        try:
            _one_step()
        except Exception as exc:
            log.warning("warmup step failed: %s", exc)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    losses = []
    for _ in range(steps):
        losses.append(_one_step())

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    total_tokens = steps * batch_size * seq_len
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "mode": mode,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "mean_loss": round(sum(losses) / len(losses), 4) if losses else 0.0,
    }


# ---------------------------------------------------------------------------
# Mode setup helpers
# ---------------------------------------------------------------------------

def _load_model(model_name: str, device: Any, streaming: bool, prefetch_layers: int):
    """Load the model and optionally attach LayerStreamer."""
    from transformers import AutoModelForCausalLM

    log.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    if streaming:
        from mud_puppy.stream import LayerStreamer
        LayerStreamer.wrap(model, prefetch_layers=prefetch_layers)
        log.info("LayerStreamer attached (prefetch_layers=%d)", prefetch_layers)
    else:
        model = model.to(device)

    model.train()
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="mud-puppy v0.4 throughput benchmark")
    p.add_argument("--model", default="gpt2", help="model name or path (default: gpt2)")
    p.add_argument("--seq-len", type=int, default=256, dest="seq_len",
                   help="sequence length per sample (default: 256)")
    p.add_argument("--batch-size", type=int, default=1, dest="batch_size",
                   help="batch size (default: 1)")
    p.add_argument("--steps", type=int, default=20,
                   help="timed steps per mode (default: 20)")
    p.add_argument("--warmup", type=int, default=3,
                   help="warmup steps (default: 3)")
    p.add_argument("--prefetch-layers", type=int, default=2, dest="prefetch_layers",
                   help="LayerStreamer prefetch depth for streaming mode (default: 2)")
    p.add_argument("--modes", nargs="+",
                   choices=["baseline", "packing", "streaming"],
                   default=["baseline", "packing"],
                   help="modes to benchmark (default: baseline packing)")
    p.add_argument("--device", default="cuda" if True else "cpu",
                   help="device (default: cuda if available, else cpu)")
    p.add_argument("--output", default="-",
                   help="output file path, or '-' for stdout (default: -)")
    p.add_argument("--verbose", action="store_true",
                   help="enable debug logging")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    import torch

    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Resolve device
    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available; falling back to CPU")
        device_str = "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)
    log.info("Benchmark device: %s", device)

    results = []

    for mode in args.modes:
        log.info("--- Mode: %s ---", mode)

        streaming = mode == "streaming"
        model = _load_model(args.model, device, streaming=streaming,
                            prefetch_layers=args.prefetch_layers)

        if mode == "packing":
            batch_fn = _make_packed_batch
        else:
            batch_fn = _make_batch

        try:
            result = _benchmark_mode(
                mode=mode,
                model=model,
                batch_fn=batch_fn,
                steps=args.steps,
                warmup=args.warmup,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=device,
            )
        except Exception as exc:
            log.error("Mode %s failed: %s", mode, exc)
            result = {"mode": mode, "error": str(exc)}
        finally:
            # Release model memory between modes
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append(result)

    # Output
    output_lines = [json.dumps(r) for r in results]
    if args.output == "-":
        for line in output_lines:
            print(line)
    else:
        with open(args.output, "w") as f:
            for line in output_lines:
                f.write(line + "\n")
        log.info("Results written to %s", args.output)

    # Print a summary table to stderr
    print("\n--- Summary ---", file=sys.stderr)
    header = f"{'Mode':<12} {'tok/s':>10} {'elapsed':>10} {'mean_loss':>10}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)
    for r in results:
        if "error" in r:
            print(f"{r['mode']:<12} {'ERROR':>10}", file=sys.stderr)
        else:
            print(
                f"{r['mode']:<12} {r['tokens_per_sec']:>10.1f}"
                f" {r['elapsed_sec']:>9.1f}s"
                f" {r['mean_loss']:>10.4f}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
