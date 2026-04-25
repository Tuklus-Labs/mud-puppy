"""Coder GRPO driver: resume SFT-warmed coder, train with the live sandbox reward.

This is the GRPO stage for Stream B of the MI300X plan. Picks up from
the SFT checkpoint, generates ``num_generations=4`` candidate patches
per prompt, scores them with
``mud_puppy.rl_verifier.coder_compile_test_reward`` (which delegates to
the bwrap-jailed compile-and-test sandbox), and updates the policy.

Key dataset invariants (baked into this script):

* ``prompt`` column in chat-messages form, without the gold diff.
* ``repo`` shorthand column (one of llamacpp/redis/sqlite/leveldb).
* ``base_commit`` full SHA.
* ``test_command`` optional, ``command_tier`` optional.

``remove_unused_columns=False`` is set so TRL hands those dataset
columns to the reward function via kwargs.

Usage:
    python scripts/coder_grpo.py \\
        --sft-checkpoint outputs/coder-sft/final \\
        --data-glob 'training_data_sets/coder/*-commits.jsonl' \\
        --output outputs/coder-grpo \\
        --repo-cache-root /scratch/coder_repos

Charon milestones at launch, each save, and completion.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List

log = logging.getLogger("coder_grpo")

DEFAULT_MODEL = "/home/aegis/Models/gpt-oss-20b-hf"
DEFAULT_DATA_GLOB = "/home/aegis/Projects/mud-puppy/training_data_sets/coder/*-commits.jsonl"
DEFAULT_OUTPUT = "outputs/coder-grpo"
DEFAULT_CACHE_ROOT = "/scratch/coder_repos"

SYSTEM_PROMPT = (
    "You are a coding assistant. You will be given a task description, "
    "the current contents of relevant files, and (optionally) the names "
    "of test files. Produce a unified-diff patch that addresses the task. "
    "Respond only with the patch; no prose, no code fences."
)


def _is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def charon(kind: str, msg: str) -> None:
    if not _is_main_process():
        return
    try:
        subprocess.run(
            ["charon-emit", kind, "coder-grpo", msg],
            timeout=5, capture_output=True,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------

def _render_files_before(files_before: List[Dict]) -> str:
    blocks: List[str] = []
    for rec in files_before or []:
        path = rec.get("path", "<unknown>")
        content = rec.get("content", "")
        trunc = " (truncated)" if rec.get("truncated") else ""
        blocks.append(f"### File: {path}{trunc}\n```\n{content}\n```")
    return "\n\n".join(blocks)


def _repo_from_source(src: str) -> str:
    """Extract the shorthand repo key from the ``source`` field.

    Sources look like ``git:llamacpp`` / ``git:redis`` / etc.
    """
    if src and ":" in src:
        return src.split(":", 1)[1]
    return src


def coder_row_to_grpo(row: Dict, prompt_head_char_limit: int = 8000) -> Dict:
    """Turn one coder trajectory row into a GRPO-ready dict.

    We keep the prompt in messages form so TRL's GRPOTrainer can tokenize
    it with the chat template. We also include the columns the reward
    function will consume via kwargs.
    """
    ctx = row.get("context", {}) or {}
    files_before = ctx.get("files_before", []) or []
    test_files = ctx.get("test_files", []) or []

    user_parts = [row.get("prompt", "")]
    rendered = _render_files_before(files_before)
    if rendered:
        user_parts.append("Relevant files:\n\n" + rendered)
    if test_files:
        user_parts.append("Test files to consider:\n" + "\n".join(f"- {f}" for f in test_files))
    user = "\n\n".join(p for p in user_parts if p)
    # Hard cap on prompt size: gpt-oss-20b context is small and a single
    # sqlite vdbe.c turn can blow past 100K chars.
    if len(user) > prompt_head_char_limit:
        user = user[:prompt_head_char_limit] + "\n\n...[prompt truncated at char limit]"

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "repo": _repo_from_source(row.get("source", "")),
        "base_commit": (row.get("metadata") or {}).get("commit_sha", ""),
        "test_command": (row.get("metadata") or {}).get("test_command"),  # usually absent, sandbox picks default
        "command_tier": (row.get("metadata") or {}).get("command_tier", "quick"),
        "id": row.get("id"),
    }


def reshape_for_grpo(globs: Iterable[str], out_path: Path, max_rows: int | None = None) -> int:
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
                        rendered = coder_row_to_grpo(row)
                        if not rendered["base_commit"] or not rendered["repo"]:
                            continue
                        out.write(json.dumps(rendered) + "\n")
                        count += 1
                        if max_rows and count >= max_rows:
                            return count
    return count


# ---------------------------------------------------------------------------
# Reward wiring: wrap the rl_verifier path so cache_root gets injected
# ---------------------------------------------------------------------------

def make_reward_fn(cache_root: str, timeout_sec: int):
    """Return a closure that calls coder_compile_test_reward with cache_root.

    TRL passes dataset columns as kwargs; we need to also pin the sandbox
    cache directory per invocation.
    """
    from mud_puppy.coder_sandbox import run_sample as _run_sample
    from mud_puppy.rl_verifier import coder_compile_test_reward

    # Pre-warm the cache root so GRPO does not pay the clone cost
    # on the first generation group.
    os.environ.setdefault("MUD_PUPPY_CODER_CACHE", cache_root)

    def reward_fn(completions, **kwargs):
        # GRPOTrainer sometimes passes completions as list-of-messages
        # (when the prompt was message-form) and sometimes as list-of-str.
        # Normalize to str.
        flat_completions = []
        for c in completions:
            if isinstance(c, list):
                # TRL groups: list of {"role": ..., "content": ...}
                flat_completions.append(
                    "\n".join(m.get("content", "") for m in c if isinstance(m, dict))
                )
            else:
                flat_completions.append(str(c))
        return coder_compile_test_reward(
            completions=flat_completions,
            timeout_sec=timeout_sec,
            **kwargs,
        )

    reward_fn.__name__ = "coder_compile_test_reward"
    return reward_fn


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------

def load_policy(base_model: str, sft_checkpoint: str | None, mxfp4_attn: bool):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    log.info("base loaded: %s", type(model).__name__)

    if mxfp4_attn:
        from mud_puppy.mxfp4_kernels import MXFP4Linear, _set_module
        swapped = 0
        for name, module in list(model.named_modules()):
            if "self_attn" not in name or not isinstance(module, nn.Linear):
                continue
            try:
                dev = next(module.parameters()).device
            except StopIteration:
                continue
            if dev.type != "cuda":
                continue
            _set_module(model, name, MXFP4Linear(module, dtype=torch.bfloat16).to(dev))
            swapped += 1
        log.info("mxfp4: wrapped %d attention Linears", swapped)
        torch.cuda.empty_cache()
        gc.collect()

    if sft_checkpoint:
        from peft import PeftModel
        log.info("attaching SFT LoRA adapter from %s", sft_checkpoint)
        model = PeftModel.from_pretrained(model, sft_checkpoint, is_trainable=True)
    return model, tok


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

def make_charon_callback(output_dir: str):
    from transformers import TrainerCallback

    class CharonCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            reward = None
            for entry in reversed(state.log_history or []):
                if "reward" in entry or "reward_mean" in entry:
                    reward = entry.get("reward_mean", entry.get("reward"))
                    break
            charon(
                "milestone",
                f"checkpoint step={state.global_step} reward={reward} dir={output_dir}",
            )

        def on_train_end(self, args, state, control, **kwargs):
            charon("milestone", f"grpo training complete step={state.global_step}")

    return CharonCallback()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=DEFAULT_MODEL)
    p.add_argument("--sft-checkpoint", default=None,
                   help="Path to SFT-warmed PEFT adapter (outputs/coder-sft/final).")
    p.add_argument("--data-glob", default=DEFAULT_DATA_GLOB)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--repo-cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--max-prompt-length", type=int, default=4096)
    p.add_argument("--per-device-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--beta", type=float, default=0.04,
                   help="KL coefficient for GRPO")
    p.add_argument("--sandbox-timeout", type=int, default=120)
    p.add_argument("--mxfp4-attn", action="store_true", default=True)
    p.add_argument("--no-mxfp4-attn", dest="mxfp4_attn", action="store_false")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--reshape-only", action="store_true")
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
    grpo_data = out_dir / "coder-grpo.prompts.jsonl"

    charon(
        "milestone",
        f"launch: base={args.base_model} sft={args.sft_checkpoint} "
        f"epochs={args.epochs} n_gen={args.num_generations} lr={args.lr} "
        f"cache={args.repo_cache_root}",
    )

    # --- Reshape data --------------------------------------------------
    globs = [g for g in args.data_glob.split(",") if g]
    t0 = time.time()
    n = reshape_for_grpo(globs, grpo_data, max_rows=args.max_rows)
    log.info("reshape: %d rows in %.1fs at %s", n, time.time() - t0, grpo_data)
    charon("milestone", f"reshape done, {n} rows")

    if args.reshape_only:
        return

    # --- Pre-warm repos ------------------------------------------------
    from mud_puppy.coder_sandbox import prewarm_repo
    for repo_key in ("llamacpp", "redis", "sqlite", "leveldb"):
        try:
            t0 = time.time()
            path = prewarm_repo(repo_key, cache_root=args.repo_cache_root)
            log.info("prewarm %s -> %s (%.1fs)", repo_key, path, time.time() - t0)
            charon("milestone", f"prewarm {repo_key} ok at {path}")
        except Exception as exc:
            log.warning("prewarm %s failed: %s", repo_key, exc)
            charon("failure", f"prewarm {repo_key} failed: {exc}")

    # --- Model ---------------------------------------------------------
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    model, tok = load_policy(args.base_model, args.sft_checkpoint, args.mxfp4_attn)
    model.train()

    ds = load_dataset("json", data_files=str(grpo_data))["train"]
    log.info("grpo dataset size: %d", len(ds))

    reward_fn = make_reward_fn(args.repo_cache_root, args.sandbox_timeout)

    fsdp_kwargs = {}
    if args.fsdp:
        fsdp_kwargs["fsdp"] = args.fsdp
        if args.fsdp_wrap_class:
            fsdp_kwargs["fsdp_config"] = {
                "transformer_layer_cls_to_wrap": [args.fsdp_wrap_class],
                "activation_checkpointing": True,
            }
        charon("milestone", f"fsdp={args.fsdp} wrap_class={args.fsdp_wrap_class}")

    grpo_cfg = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=3,
        bf16=True,
        beta=args.beta,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        **fsdp_kwargs,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=ds,
        reward_funcs=[reward_fn],
        processing_class=tok,
        callbacks=[make_charon_callback(str(out_dir))],
    )

    charon("milestone", f"grpo start, steps_per_epoch~={len(ds) // (args.per_device_batch * args.grad_accum)}")
    try:
        trainer.train()
    except Exception as exc:
        charon("failure", f"grpo crashed: {type(exc).__name__}: {exc}")
        raise

    trainer.save_model(str(out_dir / "final"))
    tok.save_pretrained(str(out_dir / "final"))
    charon("milestone", f"grpo final adapter at {out_dir / 'final'}")


if __name__ == "__main__":
    main()
