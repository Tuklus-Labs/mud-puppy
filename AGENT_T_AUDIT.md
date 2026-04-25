# Agent T Audit — Mud-Puppy

Generated 2026-04-22 by Agent T (training-pipeline verification subagent) for
the agent-router build. Scope: verify uncommitted work after a power outage,
confirm quant-training modules still import and run, and smoke-test the
training pipeline end-to-end on a small model. No real reviewer or coder
training runs were performed; those wait on Agent D's data and Gary's approval.

## 1. Uncommitted Work Inventory

`git status` at audit time shows the following untracked material (no tracked
files modified):

```
scripts/
training_data_sets/
```

### 1.1 `scripts/gguf_to_hf_gpt_oss.py`  (12 KB)

Purpose: Convert a BF16 gpt-oss GGUF into HuggingFace sharded safetensors.
Input must be all-BF16 (produce with
`llama-quantize --allow-requantize <in.gguf> <bf16.gguf> BF16`).

Maps GGUF tensor names to HF gpt-oss names, with special handling for MoE
experts:
- GGUF stores `ffn_gate_exps.weight`, `ffn_up_exps.weight`,
  `ffn_down_exps.weight` (shape `[E, intermediate, hidden]` or
  `[E, hidden, intermediate]`).
- HF gpt_oss expects fused `mlp.experts.gate_up_proj` of shape
  `[E, hidden, 2*intermediate]` with INTERLEAVED gate/up on the last
  axis (not concatenated). HF's unpack is `gate = gate_up[..., ::2]`
  and `up = gate_up[..., 1::2]`. The script uses `torch.stack(..., dim=-1).reshape`
  to produce the interleaved layout. This detail is easy to get wrong
  and would silently break SwiGLU.
- `down_proj` is transposed to `[E, intermediate, hidden]`.

Writes sharded safetensors + `model.safetensors.index.json`. Works off the
`gguf` Python package (present in the environment).

CLI entry:
```
python scripts/gguf_to_hf_gpt_oss.py \
    --gguf /home/aegis/Models/gpt-oss-20b-bf16.gguf \
    --hf-dir /home/aegis/Models/gpt-oss-20b-hf \
    --shard-size-gb 5
```

Already executed: `/home/aegis/Models/gpt-oss-20b-hf/` exists with multiple
`model-*.safetensors` shards dated 2026-04-22, so the script ran to
completion at least once post-outage. The BF16 GGUF input
(`/home/aegis/Models/gpt-oss-20b-bf16.gguf`, 39 GB) also exists.

Not tested by Agent T (file I/O only, ~40 GB, already executed successfully).
Import sanity pass: `import scripts.gguf_to_hf_gpt_oss` would require `gguf`
(present) and `safetensors` (present).

### 1.2 `scripts/trial_gpt_oss_qlora.py`  (7.6 KB)

Purpose: Phase-5 QLoRA smoke test on the newly converted gpt-oss-20b-hf.
This is the "does anything work?" trial.

What it does:
1. Loads gpt-oss-20b-hf in bf16 with `device_map="auto"` and a GPU budget
   (attention + small tensors on GPU, MoE experts overflow to CPU).
2. Optionally wraps attention `nn.Linear` layers with `MXFP4Linear` (from
   `mud_puppy.mxfp4_kernels`) when `--mxfp4-attn` is passed. Experts stay
   bf16 on CPU because wrapping them requires a dedicated MXFP4Experts
   module (documented as Phase 6 work, not yet present).
3. Attaches a PEFT LoRA on `q_proj`, `v_proj` only.
4. Trains `--steps N` on `opus46_final.jsonl` and prints loss per step.

Quant-training module used: `mud_puppy.mxfp4_kernels` (for `MXFP4Linear`
and `_set_module` helper) and PEFT for LoRA.

Known bug in script: argument default
`--data /home/aegis/Projects/mud-puppy/data/opus46_final.jsonl`
points at a non-existent directory. Real data lives at
`training_data_sets/opus46_final.jsonl`. Needs `--data` override or a
fix. This is a one-line fix but Agent T is not committing, so it is
noted here for Gary's review.

Not executed by Agent T (too heavy for a Phase-3 audit; it spins up the
full 20B model).

### 1.3 `training_data_sets/` contents

Pre-existing JSONL (Gary's prior work, do not touch):
- `aegis-reflib-qlora.jsonl` -- 11 MB, 9738 lines. AEGIS reference-library
  QLoRA data. Schema: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`.
- `opus46_final.jsonl` -- 13 MB, 9633 lines. Opus 4.6 distill data. Schema:
  `{"messages": [{"role": "system", ...}, {"role": "user", ...},
  {"role": "assistant", "content": ..., "reasoning": ...}]}`.

Agent D work in progress (visible at audit time):
- `_work/swe-bench-head.parquet` -- 128 KB (likely a head sample)
- `_work/swe-bench-train.parquet` -- 102 MB and growing; Agent D is
  actively downloading.
- `reviewer/` -- empty directory, target for curated reviewer JSONL
- `coder/` -- empty directory, target for curated coder JSONL

Layout matches the spec at `agent-router/SPEC.md`
lines 205-222.

## 2. Quant-Training Module Verification

Environment: system Python 3.14.4 at `/usr/bin/python3`.
`torch` 2.11.0+rocm7.2, `transformers` 5.2.0, `peft` 0.19.0, `trl` 1.1.0,
`datasets` 4.8.4. ROCm HIP 7.2.26015. 7900 XTX visible as `cuda:0`.

The mud-puppy venv at `/home/aegis/Projects/mud-puppy/venv/` is stale
(no torch). Everything here uses system Python.

| Module | Import | Smoke call | Status |
|--------|--------|-----------|--------|
| `mud_puppy.bnb_rocm`   | OK | `quantize_model_4bit` on a 2-layer MLP, forward returns expected shape in bf16 | PASS |
| `mud_puppy.mxfp4_rocm` | OK | `quantize_model_mx4` block_size=32 on a 2-layer MLP, forward returns expected shape | PASS |
| `mud_puppy.qat_rocm`   | OK | `apply_qat` then forward, then `convert_qat` then forward | PASS |
| `mud_puppy.gptq_rocm`  | OK | `quantize_model_gptq(bits=4, calibration_data=[...])` on a 2-layer MLP, forward returns expected shape | PASS |
| `mud_puppy.rl`         | OK | `GRPOTrainer` available from trl; `PPOTrainer` fallback is NOT available in trl 1.1.0 (trl moved to native GRPO, PPO API removed) | PASS |
| `mud_puppy.trainer`    | OK | `run_training` used in smoke test (see Section 4) | PASS |
| `mud_puppy.mxfp4_kernels` | OK | imports (used by `trial_gpt_oss_qlora.py`) | PASS |
| `mud_puppy.mxfp4_train`/`mxfp4_optim` | OK | imports | PASS |
| `mud_puppy.jax_lora`   | not tested | JAX not required for reviewer/coder training | n/a |

Non-blocking noise on import (these are present regardless of mud-puppy
and do not break anything):
- `transformers.integrations.hub_kernels` warning about `_nssio_kernels`
  circular import. Unrelated (leaks from the Aegis NSSIO package in the
  user's global site-packages). Harmless: falls back to Python.
- `torchao` fails to load two prebuilt `.so` files
  (`_C_cutlass_90a.abi3.so`, `_C_mxfp8.cpython-310-x86_64-linux-gnu.so`).
  These are CUDA kernels that do not apply to ROCm. Harmless.

No fixes applied. No files modified.

## 3. GRPO / Trainer Code-Path Verification

`mud_puppy/trainer.py` (48 KB, last modified 2026-04-22):
- `run_training(config)` is the main SFT / LoRA / QLoRA / full path.
- `MudPuppyTrainer` subclasses HF `Trainer` with ROCm-friendly defaults
  and a custom `get_train_dataloader` that supports `DynamicBatchSampler`.
- `load_model` handles quantization backend selection (int4 via
  `bnb_rocm`, mxfp4 via `mxfp4_rocm`) and raises a clear error if a
  weight file is passed instead of a model directory.
- `prepare_lora` auto-detects target modules via `_detect_lora_targets`
  with a fallback to the config default.

`mud_puppy/rl.py` (9 KB):
- `run_grpo_training(config)` dispatches to `_run_native_grpo` (TRL
  `GRPOTrainer`) when available, else `_run_ppo_grpo`. In this
  environment only the native path is available (trl 1.1.0 removed PPO).
- Reward function is a length heuristic by default. Real reviewer
  training will replace this with a verifier-grounded reward (compile
  and test success from the sandbox).

CLI entry (`mud_puppy.cli.build_parser` returns an `ArgumentParser` with all
expected flags): verified by constructing the parser and printing its help.
`python -m mud_puppy --help` does NOT work because `mud_puppy/__main__.py`
is absent. Use either:
- `python -c "from mud_puppy.cli import main; main()"` with `sys.argv` set, or
- The console-script `mud-puppy` once `pip install -e .` succeeds.

Install note: `pip install -e .` currently fails because
`pyproject.toml` has no `[tool.setuptools.packages.find]` or explicit
`packages = [...]`. Setuptools can't figure out what to package.
This is a one-line `pyproject.toml` addition. Not fixed (no commits
authorised). Recommended fix:

```toml
[tool.setuptools.packages.find]
include = ["mud_puppy*"]
```

Smoke tests import mud-puppy directly from the source tree by setting
`PYTHONPATH=/home/aegis/Projects/mud-puppy`, so this does not block
Phase 3.

## 4. Smoke Test — TinyLlama-1.1B, 10 LoRA Steps

Goal: prove the end-to-end pipeline still works (data loader, model load,
LoRA wrapping, trainer.train()) on a small model with minimal data.

Setup:
- Model: `/home/aegis/Models/TinyLlama` (local HF dir, 2 GB, Llama-arch
  1.1B). The GGUF version was rejected (mud-puppy requires an HF
  directory, not a weight blob).
- Data: first 50 lines of `training_data_sets/aegis-reflib-qlora.jsonl`
  copied to `/tmp/mudpuppy_smoke/tiny.jsonl`. Original file untouched.
- Config: `finetuning_method="lora"`, `precision="bf16"`,
  `batch_size=1`, `max_seq_length=256`, `lora_r=8`,
  `lora_target_modules=["q_proj","v_proj"]`, `max_steps=10`,
  `save_strategy="no"`, `report_to="none"`.
- Driver: `/tmp/mudpuppy_smoke/run_smoke.py` (monkey-patches
  `create_training_args` to inject `max_steps=10`).

Results:
```
trainable params: 6,307,840 || all params: 1,106,356,224 || trainable%: 0.5701
step 1  loss=3.076  grad_norm=1.843  lr=0.00020
step 2  loss=2.060  grad_norm=1.456  lr=0.00018
step 3  loss=2.484  grad_norm=1.624  lr=0.00016
step 4  loss=1.912  grad_norm=1.453  lr=0.00014
step 5  loss=2.628  grad_norm=1.331  lr=0.00012
step 6  loss=2.736  grad_norm=1.422  lr=0.00010
step 7  loss=3.214  grad_norm=2.403  lr=0.00008
step 8  loss=2.130  grad_norm=1.515  lr=0.00006
step 9  loss=2.006  grad_norm=1.457  lr=0.00004
step 10 loss=2.204  grad_norm=1.628  lr=0.00002
train_runtime=1.282s  train_samples_per_second=7.80  steps/s=7.80
total wall (incl. model load & save): 3.8 s
peak VRAM: 3.12 GB
```

Interpretation:
- Loss is finite, decreases on net, grad norms are sane. The pipeline
  is operational.
- 7.8 steps/s on TinyLlama at seq_len 256 is reasonable.
- 3.12 GB peak for a 1.1B model in bf16 with a small LoRA and
  gradient checkpointing disabled is expected (roughly 2x weights +
  activations for seq_len 256, batch 1).
- No NaNs, no CUDA OOM, no data-loader crashes, no LoRA target-module
  resolution issues.

Artifact locations (Agent T-created, ephemeral):
- Driver: `/tmp/mudpuppy_smoke/run_smoke.py`
- Data: `/tmp/mudpuppy_smoke/tiny.jsonl`
- Output: `/tmp/mudpuppy_smoke/outputs/`  (save_strategy="no", empty)

No code in the mud-puppy repo was modified.

## 5. Findings and Recommendations

### Works as-is (no action needed from Gary to proceed)
- All four ROCm quant modules (bnb/mxfp4/qat/gptq) import and run.
- `run_training` SFT/LoRA path works end-to-end on a real model.
- `run_grpo_training` entry is present; `GRPOTrainer` from trl 1.1.0 is
  importable and will be used when we train the reviewer.
- `scripts/gguf_to_hf_gpt_oss.py` has already run to completion
  (`/home/aegis/Models/gpt-oss-20b-hf/` exists with shards).

### Small fixes worth making before real training (not committed by Agent T)
1. `pyproject.toml`: add `[tool.setuptools.packages.find]` section so
   `pip install -e .` succeeds and the `mud-puppy` console script works.
2. `scripts/trial_gpt_oss_qlora.py`: default `--data` path points at
   `data/opus46_final.jsonl` which does not exist; real file lives under
   `training_data_sets/opus46_final.jsonl`.
3. Optional: add `mud_puppy/__main__.py` so `python -m mud_puppy` works.

### Blockers for Phase 5/6 (reviewer/coder training)
- Phase 5 (reviewer): waiting on Agent D's data curation for
  `reviewer/swe-bench-train.jsonl`, `bug-injection-multilang.jsonl`,
  `pr-reviews-osint.jsonl`, `codereviewer-sft-warmup.jsonl`.
- Phase 6 (coder): waiting on Agent D's data for
  `coder/{llamacpp,redis,sqlite,leveldb}-commits.jsonl`, AND on the
  Phase-6 work described in `trial_gpt_oss_qlora.py` header (an
  MXFP4Experts module for MoE expert quantization). Attention-only
  MXFP4 on gpt-oss-20b works now; expert MXFP4 does not.

### Data loader compatibility with reviewer/coder formats
- The existing `aegis-reflib-qlora.jsonl` and `opus46_final.jsonl` use
  `messages` schema which mud-puppy's trainer auto-detects via
  `use_chat_template=True`.
- Reviewer data will most likely also be `messages`-formatted once Agent
  D curates it. Coder data will probably be `prompt` / `completion` or
  `messages` — mud-puppy handles both per the README.

## 6. Next Step Recommendation

Agent T is done with its phase-3 scope. Agent D is still running.
Proposed next actions, in order:

1. Hold until Agent D reports reviewer/coder JSONL curation complete.
2. Before launching phase-5 reviewer training, apply the two
   `pyproject.toml` + `trial_gpt_oss_qlora.py` data-path fixes as part of
   a single "Phase-3 cleanup" commit (owner: Gary, not Agent T).
3. Follow `TRAINING_RUNBOOK.md` (next to this file) for launch commands.
4. Phase-6 coder training is deferred until the MXFP4Experts module
   lands (Gary's call; could be skipped by training coder with
   attention-only MXFP4 + bf16 experts, accepting the ~half-quant
   memory savings).
