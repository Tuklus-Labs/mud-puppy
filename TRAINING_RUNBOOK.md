# Mud-Puppy Training Runbook (Reviewer + Coder, agent-router Phase 5/6)

This runbook is for the operator (Gary, or a successor agent with USER
provenance) who will kick off the real reviewer and coder training runs
after Agent D finishes curating data. Agent T (phase 3) wrote it but
DOES NOT run it.

All commands assume:
- cwd: `/home/aegis/Projects/mud-puppy`
- Python: system `/usr/bin/python3` (3.14.4 with ROCm-torch 2.11.0)
- GPU: 7900 XTX visible as `cuda:0`
- OC profile: Gary's standard `gpu-oc apply` (see global CLAUDE.md)
  may or may not be active; reviewer/coder training does not require it
  and it is OK to skip for stability on overnight runs.

## 0. One-time pre-flight

### 0.1 Apply the pyproject fix and install mud-puppy editable

`pyproject.toml` currently cannot build (no package discovery). Add:

```toml
[tool.setuptools.packages.find]
include = ["mud_puppy*"]
```

Then:

```
cd /home/aegis/Projects/mud-puppy
pip install -e . --user
```

Verify: `mud-puppy --help` prints the parser.

If you would rather not touch `pyproject.toml` for any reason, every
command below can be run via
`PYTHONPATH=/home/aegis/Projects/mud-puppy python3 -m mud_puppy.cli ...`
(requires adding `mud_puppy/__main__.py` with
`from .cli import main; main()`) or via an explicit
`PYTHONPATH=... python3 -c "from mud_puppy.cli import main; main()" ...`.

### 0.2 Confirm data is present

```
ls -la training_data_sets/reviewer/ training_data_sets/coder/
```

Expected (after Agent D finishes):

```
reviewer/
  swe-bench-train.jsonl
  bug-injection-multilang.jsonl
  pr-reviews-osint.jsonl
  codereviewer-sft-warmup.jsonl     # optional SFT warm-start
coder/
  llamacpp-commits.jsonl
  redis-commits.jsonl
  sqlite-commits.jsonl
  leveldb-commits.jsonl
```

If any file is missing, STOP. Do not launch training on partial data.

### 0.3 Confirm base models

```
# Reviewer base
ls /home/aegis/Models/Ministral-3-14B-Reasoning/config.json

# Coder base (converted HF form, from gguf_to_hf_gpt_oss.py)
ls /home/aegis/Models/gpt-oss-20b-hf/config.json
ls /home/aegis/Models/gpt-oss-20b-hf/model.safetensors.index.json
```

Both should exist. `gpt-oss-20b-hf` was created by
`scripts/gguf_to_hf_gpt_oss.py` and is present as of 2026-04-22.

### 0.4 Clear GPU and pre-warm the allocator

```
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HSA_OVERRIDE_GFX_VERSION=11.0.0     # if your ROCm needs it; mine does not
rocm-smi --showuse
```

## 1. Phase 5: Reviewer Training (Ministral-3-14B-Reasoning)

### 1.1 Data mix

Per spec (agent-router/SPEC.md lines 163-181):
- 60 % SWE-Bench-Train  (non-held-out split)
- 25 % bug-injection multilang  (C/C++/Python)
- 10 % high-quality PR review comments
- 5 % style/lint pairs

Agent D concatenates the sources into a single JSONL
`training_data_sets/reviewer/mix.jsonl` with per-sample weights OR
just a simple interleaved union. Confirm with Agent D's README
under `training_data_sets/reviewer/README.md`.

### 1.2 Optional SFT warm-start on CodeReviewer

Only if Gary wants the reviewer to learn the review output format
before GRPO:

```
mud-puppy \
    /home/aegis/Models/Ministral-3-14B-Reasoning \
    training_data_sets/reviewer/codereviewer-sft-warmup.jsonl \
    --method qlora \
    --quant-backend int4 \
    --precision bf16 \
    --batch-size 1 \
    --gradient-accumulation 16 \
    --learning-rate 1e-4 \
    --epochs 1 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-targets q_proj,k_proj,v_proj,o_proj \
    --max-seq-length 2048 \
    --output /home/aegis/Projects/mud-puppy/outputs/reviewer-sft-warmup \
    --log-with tensorboard \
    --early-stopping 3
```

Budget: ~4-6 hours overnight on 7900 XTX, ~16 GB VRAM with QLoRA.
If it OOMs: drop batch to 1 + gradient-accumulation 32, or seq-length 1536.

### 1.3 Main reviewer training (GRPO with verifier-grounded rewards)

This is the real training run. It plugs into mud-puppy's
`run_grpo_training`, which uses TRL's native `GRPOTrainer`.

IMPORTANT: `mud_puppy/rl.py` uses a length-heuristic reward by default.
The agent-router spec says rewards should be verifier-grounded
(`test pass/fail` from the sandbox). Before launching, either:

(a) Replace `_compute_heuristic_reward` in `mud_puppy/rl.py` with a
    reward function that reads a per-sample ground-truth label from the
    dataset (e.g. `{"prompt": ..., "reward": 1.0}` for test-pass or
    `0.0` for test-fail). Agent D's reviewer JSONL should include the
    label.

(b) Or write a custom driver (like the smoke test in
    `/tmp/mudpuppy_smoke/run_smoke.py`) that constructs a `GRPOTrainer`
    directly with a `reward_funcs=[...]` callable that Gary defines.

Option (b) is cleaner if we want the reward to call out to a test
sandbox live during training. Option (a) is simpler if rewards are
precomputed.

Once that's chosen, launch:

```
mud-puppy \
    /home/aegis/Projects/mud-puppy/outputs/reviewer-sft-warmup \
    training_data_sets/reviewer/mix.jsonl \
    --method rl \
    --precision bf16 \
    --batch-size 1 \
    --gradient-accumulation 8 \
    --learning-rate 5e-6 \
    --epochs 1 \
    --max-seq-length 4096 \
    --output /home/aegis/Projects/mud-puppy/outputs/reviewer-grpo \
    --log-with tensorboard \
    --resume
```

Base: the SFT-warmed checkpoint from 1.2 (or the raw Ministral-3 if you
skip warm-start).

Budget: 1-2 nights on 7900 XTX. Expected peak ~18-22 GB VRAM for
group-of-4 GRPO at 4K context. If OOM: drop `num_generations` in
GRPOConfig to 2, or seq to 2048.

### 1.4 Reviewer success criterion

After each eval step, compute accuracy on a held-out 500-sample
SWE-Bench slice:
- Predict `pass` vs `fail` on a `(code, patch, tests)` triple.
- Target: ≥ 75 % held-out accuracy. Anything under 60 % means the
  reward signal is broken.

Monitor via `tensorboard --logdir outputs/reviewer-grpo/runs`.

## 2. Phase 6: Coder Training (gpt-oss-20b, MXFP4 native)

### 2.1 Pre-flight

Phase-6 requires an MXFP4-capable MoE module. The current
`scripts/trial_gpt_oss_qlora.py` only wraps attention Linears; experts
stay bf16 on CPU. Options:

(a) Skip MoE quantization. Train coder with attention-only MXFP4 +
    bf16 experts. Accepts higher PCIe traffic for the experts, slower
    per-step time. Workable on 7900 XTX if batch size stays at 1.
(b) Wait for MXFP4Experts module (Phase-6 work in the script's
    docstring). Gary's call.

For now assume (a).

### 2.2 Data mix

Per spec (lines 188-200):
- C/C++ commits from llama.cpp, Redis, SQLite, LevelDB (majority)
- minimal Python (base model is already strong there)
- skip ASM, skip CSS

Agent D produces `training_data_sets/coder/mix.jsonl` (or a concat of the
four files). Each sample is a `(pre-commit code, commit message,
post-commit code, tests passed?)` trajectory.

### 2.3 Launch coder training

Coder uses GRPO with compile-and-test rewards. Same caveat as reviewer:
the default reward function must be replaced with a real compile-and-test
harness. Suggestion: write a custom Python driver that:
1. Builds a `GRPOTrainer` with `reward_funcs=[compile_and_test_reward]`.
2. Inside the reward, spawns a sandboxed worker (bwrap) to `make && test`.
3. Returns 1.0 for test-pass, 0.0 for test-fail, -0.5 for compile-fail.

```
# Custom driver; mud-puppy CLI cannot express the reward hook directly.
cd /home/aegis/Projects/mud-puppy
PYTHONPATH=. python3 scripts/trial_gpt_oss_qlora.py \
    --model /home/aegis/Models/gpt-oss-20b-hf \
    --data training_data_sets/coder/mix.jsonl \
    --steps 5000 \
    --batch 1 \
    --seq-len 2048 \
    --lora-r 32 \
    --lr 1e-4 \
    --mxfp4-attn \
    --gpu-mem-gib 20
```

Note: `trial_gpt_oss_qlora.py` is a LoRA+SFT script, NOT a GRPO script
yet. For real coder training you need to either:
- Extend it to accept a reward-function hook and do GRPO rollouts, OR
- Write `scripts/coder_grpo.py` patterned on `mud_puppy/rl.py::_run_native_grpo`
  but with a custom reward function.

Budget: 1-2 nights. Peak VRAM budget ~22 GB.

### 2.4 Coder success criterion

Sample a held-out 100-commit test set from llama.cpp master. For each:
- Give model the `(pre-commit code, commit message)` prompt.
- Run the repo's test suite on the model's output.
- Target: ≥ 40 % test pass on held-out commits (coding improvement is
  hard; 40 % is a realistic floor for a 20B fine-tune).

## 3. Monitoring

### GPU utilization
```
watch -n 2 rocm-smi --showuse --showtemp --showmemuse
```

### Training dashboard (built into mud-puppy)
Add `--monitor` to any command. Dashboard at
`http://localhost:5980`. WebSocket feed for `mud-puppy-studio`
(if you have it running).

### TensorBoard
```
tensorboard --logdir outputs/<run-name>/runs
# Gary's usual: http://localhost:6006
```

### Nerve center emissions
Training runs should emit milestones. Insert at checkpoints:
```
charon-emit milestone "reviewer GRPO: epoch 1/1 step 1000, loss=X, reward=Y"
```

### Kairos
Auto-tracks project activity. No action needed.

## 4. Resume from checkpoint

mud-puppy writes HF trainer checkpoints every `save_steps` (default
500 or every epoch). To resume after a crash:

```
mud-puppy ... --resume --output /path/to/existing/run
```

The `--resume` flag tells the HF Trainer to pick up the most recent
`checkpoint-*` dir inside `--output`. Do NOT change the `--output`
path between the original launch and the resume, or resume will be
silent-ignored and training will restart from scratch.

If the run crashed mid-step (partial optimizer state), HF Trainer
rolls back to the last good checkpoint automatically.

## 5. What success looks like

### Reviewer
- Held-out SWE-Bench-Train accuracy: ≥ 75 %.
- Reward curve trending upward on tensorboard.
- Generated review text is structured: `[pass/fail]: [reason]`.
- VRAM stable, no OOM over a night.
- Final checkpoint at `outputs/reviewer-grpo/` has
  `adapter_config.json` (LoRA) plus `pytorch_model.bin`.

### Coder
- Held-out llama.cpp test-pass rate: ≥ 40 %.
- Reward curve trending upward.
- Compiles at rate > 80 % (compile success >> test success).
- Generated code matches the repo's style.

## 6. Post-training

After both runs finish and Gary approves the metrics:

1. Merge LoRA adapters into base weights:
   ```
   mud-puppy ... --merge-lora --merge-precision bf16 \
       --output outputs/reviewer-merged
   ```
2. Export to GGUF for llama-server hosting:
   ```
   mud-puppy ... --export-gguf --gguf-quant Q4_K_M
   ```
3. Serve via `llama-server` on ports 8001 (planner), 8002 (coder),
   8003 (reviewer). agent-router daemon then picks them up.

## 7. Safety reminders

- No `rm -rf outputs/`. Rename to `outputs-failed/` if a run is bad.
- No `git push --force` on the training branch.
- If a training run is clearly diverging (loss NaN, reward stuck at
  zero after 200+ steps), Ctrl-C and diagnose. Do not "just restart".
- The preexisting `aegis-reflib-qlora.jsonl` and `opus46_final.jsonl`
  are not part of reviewer/coder training. Do not mix them in.
