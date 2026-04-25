"""Coder SFT driver: gpt-oss-20b warm-start on the four commit JSONLs.

This is the SFT stage for Stream B of the MI300X plan. Reshapes the
Agent D coder JSONL trajectories (one of
``training_data_sets/coder/{llamacpp,redis,sqlite,leveldb}-commits.jsonl``)
into a messages format and trains gpt-oss-20b with LoRA on the
attention-projection modules. The MoE experts stay bf16; only
attention is wrapped in MXFP4 for memory headroom.

Why a hand-rolled driver and not ``mud-puppy sft``?

* mud-puppy CLI does not give us direct handles to attach the MXFP4
  attention wrapper or to point LoRA at specific modules per Gary's
  MI300X plan.
* We also want to emit charon milestones at the start, at each save,
  and at completion, which the CLI does not wire.

The data pipeline mirrors ``scripts/trial_gpt_oss_qlora.py`` (the
reference for the gpt-oss MXFP4 path). Coder trajectories are rendered
into a two-message conversation:

    system: "You are a coding assistant. Apply the requested change."
    user:   <prompt>  +  serialized files_before  +  list of test files
    assistant: <target.diff>

Loss is computed only on the assistant portion via the standard
completions-only SFTTrainer behavior (assistant mask derived from the
chat template).

Usage:
    python scripts/coder_sft.py \\
        --data-glob 'training_data_sets/coder/*-commits.jsonl' \\
        --output outputs/coder-sft \\
        --epochs 1

Charon milestones are emitted at launch, each checkpoint save, and
completion. Rate-limit and OOM errors surface as charon failures so the
pod-side monitor can react.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

log = logging.getLogger("coder_sft")

DEFAULT_MODEL = "/home/aegis/Models/gpt-oss-20b-hf"
DEFAULT_DATA_GLOB = "/home/aegis/Projects/mud-puppy/training_data_sets/coder/*-commits.jsonl"
DEFAULT_OUTPUT = "outputs/coder-sft"

SYSTEM_PROMPT = (
    "You are a coding assistant. You will be given a task description, "
    "the current contents of relevant files, and (optionally) the names "
    "of test files. Produce a unified-diff patch that addresses the task. "
    "Respond only with the patch; no prose, no code fences."
)


# ---------------------------------------------------------------------------
# Charon helpers (best-effort; do not fail the run if charon is down)
# ---------------------------------------------------------------------------

def _is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def charon(kind: str, msg: str) -> None:
    if not _is_main_process():
        return
    try:
        subprocess.run(
            ["charon-emit", kind, "coder-sft", msg],
            timeout=5, capture_output=True,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data reshape: coder JSONL -> messages JSONL
# ---------------------------------------------------------------------------

def _render_files_before(files_before: List[Dict]) -> str:
    """Render the ``files_before`` list into a prompt-appendable string.

    Each entry is ``{path, content, truncated}``. Very long files are
    already truncated in data prep with a marker.
    """
    blocks: List[str] = []
    for rec in files_before or []:
        path = rec.get("path", "<unknown>")
        content = rec.get("content", "")
        trunc = " (truncated)" if rec.get("truncated") else ""
        blocks.append(f"### File: {path}{trunc}\n```\n{content}\n```")
    return "\n\n".join(blocks)


def coder_row_to_messages(row: Dict) -> Dict:
    """Turn one coder trajectory row into a messages-format row."""
    ctx = row.get("context", {}) or {}
    files_before = ctx.get("files_before", []) or []
    test_files = ctx.get("test_files", []) or []

    user_parts = [row.get("prompt", "")]
    rendered_files = _render_files_before(files_before)
    if rendered_files:
        user_parts.append("Relevant files:\n\n" + rendered_files)
    if test_files:
        user_parts.append("Test files to consider:\n" + "\n".join(f"- {f}" for f in test_files))

    user = "\n\n".join(p for p in user_parts if p)
    diff = (row.get("target") or {}).get("diff", "")

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": diff},
        ],
        # Pass-through for debugging / evals
        "id": row.get("id"),
        "source": row.get("source"),
    }


def reshape_jsonls(globs: Iterable[str], out_path: Path, max_rows: int | None = None) -> int:
    """Write a messages JSONL from the union of input globs.

    Returns the number of rows written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w") as out:
        for pattern in globs:
            for path in sorted(glob.glob(pattern)):
                log.info("reshape: reading %s", path)
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        rendered = coder_row_to_messages(row)
                        out.write(json.dumps(rendered) + "\n")
                        count += 1
                        if max_rows and count >= max_rows:
                            log.info("reshape: hit max_rows=%d", max_rows)
                            return count
    return count


# ---------------------------------------------------------------------------
# Model + LoRA
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, mxfp4_attn: bool):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    log.info("loaded %s", type(model).__name__)

    if mxfp4_attn:
        from mud_puppy.mxfp4_kernels import MXFP4Linear, _set_module
        swapped = 0
        for name, module in list(model.named_modules()):
            if "self_attn" not in name:
                continue
            if not isinstance(module, nn.Linear):
                continue
            try:
                dev = next(module.parameters()).device
            except StopIteration:
                continue
            if dev.type != "cuda":
                continue
            new_mod = MXFP4Linear(module, dtype=torch.bfloat16).to(dev)
            _set_module(model, name, new_mod)
            swapped += 1
        log.info("mxfp4: wrapped %d attention Linears", swapped)
        torch.cuda.empty_cache()
        gc.collect()
    return model, tok


def attach_lora(model, r: int, alpha: int):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("lora trainable: %.2fM / %.2fM (%.4f%%)",
             trainable / 1e6, total / 1e6, trainable / total * 100)
    return model


# ---------------------------------------------------------------------------
# Charon-wired TrainerCallback
# ---------------------------------------------------------------------------

def make_charon_callback(output_dir: str):
    from transformers import TrainerCallback

    class CharonCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            charon(
                "milestone",
                f"checkpoint saved step={state.global_step} loss={state.log_history[-1].get('loss') if state.log_history else None} dir={output_dir}",
            )

        def on_train_end(self, args, state, control, **kwargs):
            charon("milestone", f"training complete, step={state.global_step}")

    return CharonCallback()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data-glob", default=DEFAULT_DATA_GLOB,
                   help="Glob pattern for coder JSONLs (quote the arg).")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per-device-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--mxfp4-attn", action="store_true", default=True)
    p.add_argument("--no-mxfp4-attn", dest="mxfp4_attn", action="store_false")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap on reshape for fast dry-runs")
    p.add_argument("--reshape-only", action="store_true",
                   help="Just write the messages JSONL and exit")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--fsdp", type=str, default="",
                   help='FSDP mode. "" for single-GPU, "full_shard" for MI300X x8.')
    p.add_argument("--fsdp-wrap-class", type=str, default="GptOssDecoderLayer",
                   help="Transformer layer class name for FSDP auto-wrap.")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    messages_path = out_dir / "coder-sft.messages.jsonl"

    charon("milestone",
           f"launch: model={args.model} epochs={args.epochs} "
           f"batch={args.per_device_batch}x{args.grad_accum} lr={args.lr} "
           f"lora=r{args.lora_r}/a{args.lora_alpha} mxfp4_attn={args.mxfp4_attn}")

    # --- Reshape -------------------------------------------------------
    globs = [g for g in args.data_glob.split(",") if g]
    t0 = time.time()
    n = reshape_jsonls(globs, messages_path, max_rows=args.max_rows)
    log.info("reshape: %d rows written to %s in %.1fs", n, messages_path, time.time() - t0)
    charon("milestone", f"reshape done, {n} rows at {messages_path}")

    if args.reshape_only:
        log.info("--reshape-only set, exiting after data prep")
        return

    # --- Model + LoRA --------------------------------------------------
    import torch  # noqa: F401  (ensure torch is importable before we load)
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = load_model_and_tokenizer(args.model, args.mxfp4_attn)
    model = attach_lora(model, args.lora_r, args.lora_alpha)
    model.train()

    ds = load_dataset("json", data_files=str(messages_path))["train"]
    log.info("dataset size: %d", len(ds))

    fsdp_kwargs = {}
    if args.fsdp:
        fsdp_kwargs["fsdp"] = args.fsdp
        if args.fsdp_wrap_class:
            fsdp_kwargs["fsdp_config"] = {
                "transformer_layer_cls_to_wrap": [args.fsdp_wrap_class],
                "activation_checkpointing": True,
            }
        charon("milestone", f"fsdp={args.fsdp} wrap_class={args.fsdp_wrap_class}")

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=3,
        bf16=True,
        max_length=args.max_seq_length,
        packing=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        **fsdp_kwargs,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tok,
        callbacks=[make_charon_callback(str(out_dir))],
    )

    charon("milestone", f"training start, steps_per_epoch~={len(ds) // (args.per_device_batch * args.grad_accum)}")
    try:
        trainer.train(resume_from_checkpoint=args.resume)
    except Exception as exc:
        charon("failure", f"training crashed: {type(exc).__name__}: {exc}")
        raise

    trainer.save_model(str(out_dir / "final"))
    tok.save_pretrained(str(out_dir / "final"))
    charon("milestone", f"final adapter saved to {out_dir / 'final'}")


if __name__ == "__main__":
    main()
