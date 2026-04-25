#!/usr/bin/env python3
"""Reviewer SFT warmup on Ministral-3-14B (full bf16 + LoRA adapters).

Target hardware: MI300X (192 GB HBM3). We have the memory to run bf16 weights,
bf16 activations, LoRA-only trainable params, no int4 quantization. Running
QLoRA on an MI300X is leaving quality on the table; use this driver instead.

Inputs:
    --data PATH     Path to codereviewer-sft-warmup.messages.jsonl (default
                    points at the checked-in training_data_sets file).
    --out PATH      Output dir for adapters + trainer state.
    --base PATH     Base model dir (default: Models/Ministral-3-14B-Reasoning).
    --max-samples N Optional cap for debugging.
    --dry-run       Load everything but do not call trainer.train().

Contract:
    - Uses mud_puppy.model_loader.load_model_graceful for Tier 0
      (mistral-native consolidated.safetensors).
    - Uses TRL SFTTrainer with SFTConfig.
    - Emits charon milestones at: start, data-loaded, model-loaded, per-save,
      and completion.

Notes on TRL 1.1.0:
    - SFTConfig uses `max_length` (not `max_seq_length`); both accept ints.
    - Messages-format JSONL is auto-detected; no dataset_text_field needed.
    - LoRA is wired via peft_config on the SFTTrainer constructor.

No em dashes in output. No quietly-destructive git ops. No auto-push.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Silence the noisy transformers side imports while still surfacing real
# problems via log.error downstream.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    REPO_ROOT / "training_data_sets" / "reviewer"
    / "codereviewer-sft-warmup.messages.jsonl"
)
DEFAULT_BASE = Path("/home/aegis/Models/Ministral-3-14B-Reasoning")
DEFAULT_OUT = REPO_ROOT / "outputs" / "reviewer-sft"


# --------------------------------------------------------------------------
# Charon milestones
# --------------------------------------------------------------------------

def _is_main_process() -> bool:
    """Only LOCAL_RANK 0 should log and emit milestones under FSDP/torchrun."""
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def milestone(msg: str) -> None:
    """Fire a charon milestone. Non-fatal if charon is not available."""
    if not _is_main_process():
        return
    try:
        subprocess.run(
            ["charon-emit", "milestone", f"Agent Rv1: reviewer_sft: {msg}"],
            check=False, timeout=5,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    print(f"[reviewer_sft] {msg}", flush=True)


# --------------------------------------------------------------------------
# Trainer-level save callback for mid-run milestones
# --------------------------------------------------------------------------

def _make_save_callback():
    from transformers import TrainerCallback

    class CharonSaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            milestone(
                f"checkpoint saved step={state.global_step} "
                f"epoch={state.epoch:.3f}"
            )

        def on_train_end(self, args, state, control, **kwargs):
            milestone(
                f"train_end step={state.global_step} "
                f"best_metric={state.best_metric}"
            )

    return CharonSaveCallback()


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Cap training rows (0 = all).")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="per_device_train_batch_size")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="gradient_accumulation_steps")
    ap.add_argument("--max-length", type=int, default=2048,
                    help="Max sequence length (MI300X: 2048).")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--fsdp",
        type=str,
        default="",
        help='FSDP mode when launched under torchrun. One of "", "full_shard", '
        '"shard_grad_op", "no_shard", "hybrid_shard". Empty disables FSDP '
        "(single-GPU path).",
    )
    ap.add_argument(
        "--fsdp-wrap-class",
        type=str,
        default="MistralDecoderLayer",
        help="Transformer layer class name for FSDP auto-wrap. "
        "Ministral-3 text backbone uses MistralDecoderLayer.",
    )
    args = ap.parse_args()

    milestone(f"start base={args.base} data={args.data} out={args.out}")

    # Validate inputs up front so charon sees the failure cleanly.
    if not args.data.exists():
        print(f"ERROR: data file missing: {args.data}", file=sys.stderr)
        return 2
    if not args.base.exists():
        print(f"ERROR: base model missing: {args.base}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    # --- Lazy imports (keep --help fast) --------------------------------
    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from mud_puppy.model_loader import load_model_graceful

    # --- Data -----------------------------------------------------------
    milestone("loading data")
    ds = load_dataset(
        "json", data_files=str(args.data), split="train"
    )
    if args.max_samples and len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))
    milestone(f"data_loaded rows={len(ds)}")

    # --- Tokenizer ------------------------------------------------------
    milestone("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(args.base))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model ----------------------------------------------------------
    milestone("loading model via mud_puppy.load_model_graceful (bf16)")
    result = load_model_graceful(
        str(args.base),
        dtype=torch.bfloat16,
        device_map=None,  # let accelerate / trainer place it
        low_cpu_mem_usage=True,
    )
    model = result.model
    milestone(
        f"model_loaded tier={result.tier} "
        f"submodule={result.extracted_submodule}"
    )
    # Make sure the model knows its pad token so loss masking is correct.
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Gradient checkpointing + input-grad enablement is standard for LoRA
    # + PEFT on causal LMs. Do it before wrapping.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # --- LoRA config ----------------------------------------------------
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # --- SFT config -----------------------------------------------------
    fsdp_kwargs = {}
    if args.fsdp:
        fsdp_kwargs["fsdp"] = args.fsdp
        if args.fsdp_wrap_class:
            fsdp_kwargs["fsdp_config"] = {
                "transformer_layer_cls_to_wrap": [args.fsdp_wrap_class],
                "activation_checkpointing": True,
            }
        milestone(f"fsdp={args.fsdp} wrap_class={args.fsdp_wrap_class}")

    sft_config = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        fp16=False,
        max_length=args.max_length,
        packing=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=4,
        report_to=["tensorboard"],
        seed=args.seed,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        **fsdp_kwargs,
    )

    # --- Trainer --------------------------------------------------------
    milestone("building SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=sft_config,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[_make_save_callback()],
    )

    if args.dry_run:
        milestone("dry-run: exiting before train()")
        return 0

    # --- Train ----------------------------------------------------------
    milestone("train_start")
    trainer.train()
    milestone("train_complete, saving final adapter")
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    milestone(f"complete out={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
