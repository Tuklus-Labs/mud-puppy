#!/usr/bin/env python3
"""Reviewer GRPO on top of the SFT-warmed Ministral-3-14B adapter.

Inputs:
    --data PATH            Path to reviewer-grpo.jsonl.
    --out PATH             Output dir for adapters + trainer state.
    --base PATH            Base model dir (bf16 weights).
    --sft-checkpoint PATH  Directory containing the SFT-trained adapter.
                           Loaded into the base model before GRPO starts.
    --max-samples N        Optional cap.
    --dry-run              Load everything but do not call trainer.train().

Reward:
    mud_puppy.rl_verifier.reviewer_verdict_reward

    Dataset columns that flow through to the reward via TRL's broadcast:
        - expected_verdict
        - expected_reason_keywords

    remove_unused_columns MUST be False so these columns survive the
    DataCollator pass and reach the reward function's **kwargs.

Hardware target: MI300X, 192 GB HBM3. Config:
    - num_generations=4 (group size)
    - per_device_train_batch_size=2
    - gradient_accumulation_steps=4
    - max_completion_length=256
    - learning_rate=5e-6
    - epochs=1
    - logging_steps=5, save_steps=100

No em dashes anywhere. No auto-push / commit / force operations.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    REPO_ROOT / "training_data_sets" / "reviewer" / "reviewer-grpo.jsonl"
)
DEFAULT_BASE = Path("/home/aegis/Models/Ministral-3-14B-Reasoning")
DEFAULT_OUT = REPO_ROOT / "outputs" / "reviewer-grpo"


# --------------------------------------------------------------------------
# Charon milestones
# --------------------------------------------------------------------------

def _is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def milestone(msg: str) -> None:
    if not _is_main_process():
        return
    try:
        subprocess.run(
            ["charon-emit", "milestone", f"Agent Rv1: reviewer_grpo: {msg}"],
            check=False, timeout=5,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    print(f"[reviewer_grpo] {msg}", flush=True)


# --------------------------------------------------------------------------
# Save callback
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
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument(
        "--sft-checkpoint", type=Path, required=False, default=None,
        help="Path to SFT-trained adapter directory. If omitted, GRPO "
             "starts from the raw base model.",
    )
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--fsdp",
        type=str,
        default="",
        help='FSDP mode. Empty for single-GPU. "full_shard" for MI300X x8.',
    )
    ap.add_argument(
        "--fsdp-wrap-class",
        type=str,
        default="MistralDecoderLayer",
    )
    args = ap.parse_args()

    milestone(f"start base={args.base} data={args.data} out={args.out} sft={args.sft_checkpoint}")

    if not args.data.exists():
        print(f"ERROR: data file missing: {args.data}", file=sys.stderr)
        return 2
    if not args.base.exists():
        print(f"ERROR: base model missing: {args.base}", file=sys.stderr)
        return 2
    if args.sft_checkpoint is not None and not args.sft_checkpoint.exists():
        print(
            f"ERROR: sft checkpoint missing: {args.sft_checkpoint}",
            file=sys.stderr,
        )
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    # --- Lazy imports --------------------------------------------------
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from mud_puppy.model_loader import load_model_graceful
    from mud_puppy.rl_verifier import reviewer_verdict_reward

    # --- Data ----------------------------------------------------------
    milestone("loading data")
    ds = load_dataset("json", data_files=str(args.data), split="train")
    if args.max_samples and len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))
    milestone(f"data_loaded rows={len(ds)}")

    # Sanity: confirm the reward contract columns are present.
    required = {"prompt", "expected_verdict", "expected_reason_keywords"}
    missing = required - set(ds.column_names)
    if missing:
        print(f"ERROR: dataset missing columns: {missing}", file=sys.stderr)
        return 3

    # --- Tokenizer -----------------------------------------------------
    milestone("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(args.base))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model ---------------------------------------------------------
    milestone("loading model via mud_puppy.load_model_graceful (bf16)")
    result = load_model_graceful(
        str(args.base),
        dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model = result.model
    milestone(f"model_loaded tier={result.tier}")
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load the SFT-warmed LoRA into the base model. GRPOTrainer will
    # continue training the same LoRA parameters (or merge + fresh LoRA
    # if peft_config is passed). Strategy: keep training the same LoRA
    # adapter so the SFT-learned review format is preserved.
    peft_config = None
    if args.sft_checkpoint is not None:
        milestone(f"loading SFT adapter from {args.sft_checkpoint}")
        model = PeftModel.from_pretrained(
            model, str(args.sft_checkpoint), is_trainable=True,
        )
        milestone("SFT adapter attached; GRPO will continue tuning it")
    else:
        milestone("no SFT checkpoint provided, attaching fresh LoRA")
        peft_config = LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # --- GRPO config ---------------------------------------------------
    fsdp_kwargs = {}
    if args.fsdp:
        fsdp_kwargs["fsdp"] = args.fsdp
        if args.fsdp_wrap_class:
            fsdp_kwargs["fsdp_config"] = {
                "transformer_layer_cls_to_wrap": [args.fsdp_wrap_class],
                "activation_checkpointing": True,
            }
        milestone(f"fsdp={args.fsdp} wrap_class={args.fsdp_wrap_class}")

    grpo_config = GRPOConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        bf16=True,
        fp16=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=4,
        report_to=["tensorboard"],
        seed=args.seed,
        # CRITICAL: keep expected_verdict and expected_reason_keywords in
        # the batch so the reward function receives them via **kwargs.
        remove_unused_columns=False,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        temperature=0.9,
        top_p=1.0,
        top_k=0,
        **fsdp_kwargs,
    )

    # --- Trainer -------------------------------------------------------
    milestone("building GRPOTrainer")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reviewer_verdict_reward],
        args=grpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[_make_save_callback()],
    )

    if args.dry_run:
        milestone("dry-run: exiting before train()")
        return 0

    milestone("train_start")
    trainer.train()
    milestone("train_complete, saving final adapter")
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    milestone(f"complete out={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
