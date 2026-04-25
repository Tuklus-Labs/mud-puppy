"""Phase-5 trial: QLoRA on gpt-oss-20b with the new MXFP4 machinery.

What this does:
    1. Loads the converted HF gpt-oss-20b in bf16 with device_map='auto'.
       Attention + small tensors go to GPU; MoE experts overflow to CPU.
    2. Optionally wraps every nn.Linear on GPU with MXFP4Linear (frees
       ~50% of the GPU-resident weight memory). Experts stay bf16 on
       CPU -- wrapping them requires a dedicated MXFP4Experts module
       (Phase 6 work).
    3. Attaches LoRA to attention q/v projections via PEFT. LoRA lives
       on GPU in bf16; base is frozen.
    4. Trains for a handful of steps on opus46_final.jsonl and prints
       the loss curve.

This is the "does anything work on gpt-oss?" smoke test. Expected
failure modes, in order of likelihood:
    - OOM at training start (expert streams exceed working set).
      Fix: smaller seq, smaller batch, or wait for Phase 6 MoE quant.
    - Glacial step time (experts stream bf16 over PCIe every forward).
      Fix: same as above.
    - Loss NaN within a few steps (numerics issue through MXFP4 +
      cross-device).  Fix: bf16 -> fp32 loss; drop MXFP4Linear on this
      path; investigate.
    - Crash from a transformers/PEFT integration edge case on MoE.
      Fix: iterate.

Usage:
    python scripts/trial_gpt_oss_qlora.py --steps 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/aegis/Models/gpt-oss-20b-hf")
    p.add_argument("--data", default="/home/aegis/Projects/mud-puppy/training_data_sets/opus46_final.jsonl")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--mxfp4-attn", action="store_true",
                   help="Wrap attention Linears with MXFP4Linear (experimental)")
    p.add_argument("--swap-experts", action="store_true", default=True,
                   help="Swap MoE experts with MXFP4Experts (Phase 6). Default on.")
    p.add_argument("--no-swap-experts", dest="swap_experts", action="store_false",
                   help="Disable expert swap; fall back to device_map=auto CPU offload.")
    p.add_argument("--gpu-mem-gib", type=float, default=18.0,
                   help="Max GPU memory budget for model+activations")
    return p.parse_args()


def load_and_quantize(path: str, swap_experts: bool, gpu_gib: float):
    """Load the model on CPU, swap MoE experts to MXFP4, then move to GPU.

    Phase 6 made MXFP4Experts available, so we don't need device_map='auto'
    with CPU offload anymore -- the full quantized model fits on a single
    GPU. The load path:

        1. from_pretrained on CPU in bf16 (fits in 192 GB RAM)
        2. swap_gpt_oss_experts_to_mxfp4(model)     [~3x expert compression]
        3. model.to("cuda")                         [now ~14 GB, fits in 24 GB]

    Setting ``swap_experts=False`` is supported for debugging/comparison
    and falls back to the old ``device_map='auto'`` CPU-offload path.
    """
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)

    if swap_experts:
        print(f"[load] bf16 on CPU (full model, pre-swap) ...", flush=True)
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
        )
        print(f"[load] {time.time()-t0:.1f}s, type={type(model).__name__}", flush=True)
        pre_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"[load] pre-swap params: {pre_bytes/1e9:.2f} GB", flush=True)

        from mud_puppy.mxfp4_experts import swap_gpt_oss_experts_to_mxfp4
        print("[swap] MXFP4Experts for MoE layers ...", flush=True)
        t0 = time.time()
        n = swap_gpt_oss_experts_to_mxfp4(model)
        gc.collect()
        post_params = sum(p.numel() * p.element_size() for p in model.parameters())
        post_bufs = sum(b.numel() * b.element_size() for _, b in model.named_buffers())
        print(
            f"[swap] {n} layers in {time.time()-t0:.1f}s; "
            f"post params={post_params/1e9:.2f} GB + buffers={post_bufs/1e9:.2f} GB",
            flush=True,
        )

        print("[gpu] moving to cuda ...", flush=True)
        t0 = time.time()
        model = model.cuda()
        torch.cuda.synchronize()
        print(
            f"[gpu] {time.time()-t0:.1f}s, VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB",
            flush=True,
        )
    else:
        # Fallback: old CPU-offload path via accelerate device_map
        max_memory = {0: f"{int(gpu_gib)}GiB", "cpu": "150GiB"}
        print(f"[load] no swap, device_map=auto max_memory={max_memory}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="auto",
            max_memory=max_memory, low_cpu_mem_usage=True,
        )
        devices = {}
        for n, p in model.named_parameters():
            dev = str(p.device)
            devices[dev] = devices.get(dev, 0) + p.numel() * p.element_size()
        for d, b in sorted(devices.items()):
            print(f"       {d}: {b/1e9:.2f} GB")

    return model, tok


def apply_mxfp4_to_attention(model: nn.Module) -> int:
    """Wrap every nn.Linear whose name matches self_attn.* with MXFP4Linear.

    Skips MoE tensors (they're raw Parameters, not Linear modules) and
    router/embed/head (too small or too numerically sensitive).
    """
    from mud_puppy.mxfp4_kernels import MXFP4Linear, _set_module
    swapped = 0
    for name, module in list(model.named_modules()):
        if "self_attn" not in name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        # Only GPU-resident modules -- wrapping a CPU module would force
        # the dequantized weight through PCIe on every call.
        try:
            dev = next(module.parameters()).device
        except StopIteration:
            continue
        if dev.type != "cuda":
            continue
        new_mod = MXFP4Linear(module, dtype=torch.bfloat16).to(dev)
        _set_module(model, name, new_mod)
        swapped += 1
    return swapped


def attach_lora(model: nn.Module, r: int):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=r,
        lora_alpha=r * 2,
        target_modules=["q_proj", "v_proj"],  # attention-only for safety
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
          f"({trainable/total*100:.4f}%)")
    return model


def iter_batches(data_path: str, tok, batch: int, seq_len: int):
    with open(data_path) as f:
        lines = f.readlines()
    print(f"[data] {len(lines)} lines in {data_path}")
    while True:
        for i in range(0, len(lines), batch):
            texts = []
            for line in lines[i:i + batch]:
                rec = json.loads(line)
                if "messages" in rec:
                    # Apply chat template if available, else concatenate.
                    try:
                        text = tok.apply_chat_template(
                            rec["messages"], tokenize=False, add_generation_prompt=False
                        )
                    except Exception:
                        text = "\n".join(m["content"] for m in rec["messages"])
                elif "text" in rec:
                    text = rec["text"]
                else:
                    text = str(rec)
                texts.append(text)
            batch_enc = tok(
                texts, return_tensors="pt",
                truncation=True, max_length=seq_len, padding=True,
            )
            yield batch_enc


def main():
    args = parse_args()
    os.environ.setdefault("MUD_PUPPY_MXFP4_TRITON", "1")

    model, tok = load_and_quantize(args.model, args.swap_experts, args.gpu_mem_gib)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.mxfp4_attn:
        n = apply_mxfp4_to_attention(model)
        print(f"[mxfp4] wrapped {n} attention Linears")
        torch.cuda.empty_cache()
        gc.collect()

    model = attach_lora(model, args.lora_r)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr)

    data_iter = iter_batches(args.data, tok, args.batch, args.seq_len)
    print(f"[train] starting for {args.steps} steps, batch={args.batch}, seq_len={args.seq_len}")
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        step_start = time.time()
        enc = next(data_iter)
        # Move to embedding device (first GPU in device_map).
        input_ids = enc["input_ids"].to("cuda")
        attn_mask = enc["attention_mask"].to("cuda")
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        step_ms = (time.time() - step_start) * 1000
        vram = torch.cuda.memory_allocated() / 1e9
        vram_max = torch.cuda.max_memory_allocated() / 1e9
        print(f"[step {step+1:3d}/{args.steps}] loss={loss.item():.4f}  "
              f"dt={step_ms:7.0f}ms  vram={vram:5.2f}GB (peak {vram_max:5.2f})")
        losses.append(loss.item())
        if not torch.isfinite(loss):
            print("[train] loss is non-finite, aborting")
            break

    print(f"[train] done. total wall: {time.time()-t0:.1f}s")
    print(f"[train] loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
