# Reviewer Training Pipeline (Stream A)

MI300X launch package for the Ministral-3-14B reviewer. Two stages: SFT
warmup on CodeReviewer-format messages, then GRPO with the verifier reward.

Produced by Agent Rv1 per MI300X_PLAN.md Stream A.

## Data

| Path | Rows | Format | Use |
|------|------|--------|-----|
| `training_data_sets/reviewer/codereviewer-sft-warmup.messages.jsonl` | 50000 | `{"messages": [...]}` | SFT warmup |
| `training_data_sets/reviewer/reviewer-grpo.jsonl`                    | 64808 | `{"prompt","expected_verdict","expected_reason_keywords","metadata"}` | GRPO |

Build the GRPO dataset locally (CPU-only, ~2 minutes):

```bash
python3 training_data_sets/_work/build_reviewer_grpo.py
```

Input files (untouched, read-only):

- `bug-injection-multilang.jsonl`  (3846 rows, 100% kept)
- `codereviewer-sft-warmup.jsonl`  (50000 rows, 72% kept)
- `pr-reviews-osint.jsonl`         (6596 rows, 100% kept)
- `swe-bench-train.jsonl`          (18350 rows, 100% kept)

Verdict distribution (kept 64808):

- fail 46155 (71.2%)
- pass 18653 (28.8%)

Keyword cap: 5 per row. Rows with zero extractable keywords (13984, mostly
low-signal reasons) are dropped.

## Smoke test

```bash
python3 scripts/test_reviewer_grpo_data.py
```

Loads 10 rows, asserts schema, and passes synthetic completions through
`reviewer_verdict_reward` covering all four reward branches (correct+kw,
correct, wrong, unparseable). Should print "OK: 40/40 correct".

## Launch on MI300X

### SFT warmup

```bash
python3 scripts/reviewer_sft.py \
    --data training_data_sets/reviewer/codereviewer-sft-warmup.messages.jsonl \
    --base /scratch/models/Ministral-3-14B-Reasoning \
    --out /scratch/runs/reviewer-sft
```

Config (defaults chosen for MI300X):

- Full bf16 weights + bf16 activations, NO int4 quantization
- LoRA r=32, alpha=64, target modules = {q,k,v,o,gate,up,down}_proj
- per_device_batch=4, grad_accum=4  (effective 16)
- max_length=2048
- lr=1e-4, cosine schedule, warmup 3%
- 1 epoch, save every 250 steps, log every 5
- gradient_checkpointing=True
- remove_unused_columns=False

Wall-clock estimate on one MI300X: 2 to 3 hours.

### GRPO

```bash
python3 scripts/reviewer_grpo.py \
    --data training_data_sets/reviewer/reviewer-grpo.jsonl \
    --base /scratch/models/Ministral-3-14B-Reasoning \
    --sft-checkpoint /scratch/runs/reviewer-sft \
    --out /scratch/runs/reviewer-grpo
```

Config:

- Reward: `mud_puppy.rl_verifier.reviewer_verdict_reward`
- num_generations=4, per_device_batch=2, grad_accum=4 (effective group 8x4=32 rollouts)
- max_completion_length=256
- lr=5e-6, 1 epoch
- save every 100 steps, log every 5
- remove_unused_columns=False  (REQUIRED so expected_verdict and expected_reason_keywords reach the reward)

Wall-clock estimate on one MI300X: 4 to 6 hours.

## Exfil

Per Stream C plan, `scripts/mi300x/exfil.sh` (Agent Sv1) will rsync the two
output directories back to:

- `~/Projects/mud-puppy/outputs/mi300x/reviewer-sft/`
- `~/Projects/mud-puppy/outputs/mi300x/reviewer-grpo/`

These are LoRA adapters, not full checkpoints. Total size expected under
1 GB combined.

## Charon milestones

Both drivers emit milestones at: start, data_loaded, model_loaded,
train_start, every save, train_complete, complete. Monitor via:

```bash
# From the pod
charon-emit --help      # just to sanity check it exists
# From Gary's workstation, tailing remote nerve-center via agent-router trace
curl localhost:5960/history | jq '.[] | select(.payload.message | contains("Agent Rv1"))'
```

## What each script does NOT do

- No git commits. No auto-push.
- No model downloads. Base model and data must exist at the given paths.
- No auto-registration with wandb. report_to=[] by default.
- No modification of the four input Agent D JSONLs. The build script is a pure reader.
- No merging the LoRA into the base at save time. Keep the adapters portable.

## Row schema, written out

```json
{
  "prompt": "You are a code reviewer...\n\n--- repo ---\nggml-org/llama.cpp\n\n--- hunk ---\n@@ ...\n\nReturn only the JSON verdict object, one line, as specified.",
  "expected_verdict": "fail",
  "expected_reason_keywords": ["AGENTS.local.md", "ggml-org/llama.cpp"],
  "metadata": {
    "source": "github:ggml-org/llama.cpp#22246",
    "task": "reviewer.comment_on_diff",
    "source_file": "pr-reviews-osint"
  }
}
```

`metadata.bug_class` is present on bug-injection rows.
