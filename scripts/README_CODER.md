# Coder Training Stream (gpt-oss-20b)

Stream B of the MI300X plan. SFT warm-start on commit trajectories,
then GRPO with a live compile-and-test reward.

All three deliverables (sandbox, SFT, GRPO) are complete and
verified locally. Do NOT launch training runs from this document.
Launch is the operator's job; the scripts here are ready to be
driven from `scripts/mi300x/launch.sh`.

## Artifacts

| Path | Role |
|------|------|
| `mud_puppy/coder_sandbox.py` | bwrap-jailed compile-and-test harness |
| `mud_puppy/rl_verifier.py::coder_compile_test_reward` | GRPO reward, delegates to sandbox |
| `scripts/coder_sft.py` | SFT driver, emits messages JSONL + trains |
| `scripts/coder_grpo.py` | GRPO driver, live sandbox reward |
| `tests/test_coder_sandbox.py` | 21 unit tests for the sandbox + reward |
| `tests/test_rl_verifier.py` | 34 unit tests for the reviewer reward (unchanged) |

## Sandbox design

### Guarantees per sample

For each generated patch, `coder_sandbox.run_sample` does the following:

1. **Early rejection.** Completions that are empty or contain no diff
   marker (`diff --git` or `--- `) short-circuit with `error="not_a_diff"`
   or `error="empty_patch"`.
2. **Per-sample worktree.** A fresh `git worktree` is created off the
   cached base clone at the exact `base_commit`. No lock contention with
   sibling GRPO completions; N=4 parallel evaluations are safe.
3. **git apply --check.** The patch is validated against the worktree.
   If it fails, we return early with `patch_applied=False`.
4. **git apply.** On success, the patch is applied in-tree.
5. **bwrap-jailed test run.** The repo's test command runs under:
   - `--ro-bind / /` (read-only system)
   - `--tmpfs /tmp` (fresh tmpfs)
   - `--bind <worktree> <worktree>` (writable repo, mounted AFTER tmpfs
     so the order does not wipe the bind when the worktree is under /tmp)
   - `--proc /proc --dev /dev`
   - `--unshare-net` (network off)
   - `--die-with-parent` (no orphans)
   - `--chdir <worktree>`
   - Some env scrubbing (`SSH_AUTH_SOCK`, `DBUS_SESSION_BUS_ADDRESS`)
6. **Classifier.** Exit code 0 -> compiled and tests passed. Non-zero
   with compile markers (`error:`, `undefined reference`, `make ***
   Error`, `ninja: build stopped`) -> compile failed. Otherwise ->
   compiled but tests failed.
7. **Worktree teardown.** The worktree is removed unconditionally in a
   finally block.

### Repo cache layout

```
<cache_root>/
  llamacpp/
    base/               # shallow clone of ggml-org/llama.cpp (depth 500)
    worktrees/
      <uuid>/           # per-sample, cleaned up after run_sample returns
  redis/
    base/
    worktrees/
  sqlite/
    base/
    worktrees/
  leveldb/
    base/
    worktrees/
```

Default `cache_root`:
- `/scratch/coder_repos` if `/scratch` exists and is writable (MI300X pod).
- Override via `MUD_PUPPY_CODER_CACHE` env var.
- Falls back to `/tmp/scratch/coder_repos` on dev machines without /scratch.

### Pre-staging on the pod

The GRPO launcher calls `prewarm_repo("llamacpp")` etc. at startup so the
first generation does not pay the clone cost. Stream C's bootstrap also
covers this; the launcher's prewarm is a belt-and-suspenders.

Manual equivalent:
```bash
python -c 'from mud_puppy.coder_sandbox import prewarm_repo; \
           [prewarm_repo(r) for r in ("llamacpp","redis","sqlite","leveldb")]'
```

### Disk budget

| Repo | Shallow (depth 500) | Notes |
|------|---------------------|-------|
| llamacpp | ~180 MB | Verified on local smoke test |
| redis    | ~50 MB  | Small codebase |
| sqlite   | ~50 MB  | Single big file |
| leveldb  | ~15 MB  | Tiny |

Base clones: ~300 MB total. Worktrees add the full checkout size per sample
(~60 MB for llamacpp), but they are ephemeral; at N=4 parallel generations
the peak is ~250 MB of worktrees in flight. Under our 500 MB per-sample
budget.

## Per-repo test commands

Configured in `mud_puppy/coder_sandbox.REPO_TEST_COMMANDS`. Each repo has
three tiers; GRPO defaults to `quick`, which is compile-only (fast signal,
no flaky test runs).

| Repo     | tier       | Command (abbreviated) |
|----------|------------|-----------------------|
| llamacpp | expensive  | `cmake -DLLAMA_BUILD_TESTS=ON && build test-tokenizer-0 && ctest` |
| llamacpp | quick      | `cmake -DLLAMA_BUILD_TESTS=ON && build test-tokenizer-0` |
| llamacpp | dryrun     | `true` (unit-test scaffolding only) |
| redis    | expensive  | `make -j test_unit` |
| redis    | quick      | `make -j src/redis-server` |
| sqlite   | expensive  | `make quicktest` |
| sqlite   | quick      | `./configure && make sqlite3.c` |
| leveldb  | expensive  | `cmake && build && ctest` |
| leveldb  | quick      | `cmake && build leveldb` |

Per-sample overrides via the `test_command` and `command_tier` dataset
columns. Gary's full `make test` for sqlite is intentionally skipped
because it takes hours; `make quicktest` is the upstream-recommended
CI tier.

## Reward contract

```
+1.0   patch_applied AND compiled AND tests_passed
+0.3   patch_applied AND compiled AND NOT tests_passed   (partial credit)
-0.5   patch_applied AND NOT compiled
-1.0   NOT patch_applied (malformed diff, wrong parent, context mismatch)
```

Timeout on the test command is treated as "compile failed" (stricter
penalty: wedging the build is worse than emitting a patch that fails a
test). The GRPO reward function uses 120 s as the default per-sample
wall-clock.

## Expected wall-clock on MI300X

Rough estimates based on the local smoke test latency and typical MI300X
generation throughput:

| Stage                  | Cost per sample | Notes |
|------------------------|-----------------|-------|
| Patch apply check      | 20-100 ms       | Shell out + git apply --check |
| Worktree create/remove | 200-500 ms      | `git worktree add` is the dominant cost |
| Test command (quick)   | 30-90 s         | llamacpp cmake build is the slow path |
| Test command (dryrun)  | <100 ms         | `true` in bwrap |
| **Total per sample (quick)** | **~60 s** | Budget 120 s with slop |

With `num_generations=4` and one sample per step, a GRPO step evaluating
sequentially costs ~4 minutes on the reward side alone. Generation itself
on MI300X at bf16 for gpt-oss-20b should run ~100 tok/s; 512
completion tokens x 4 generations = ~20 s of GPU time per step. Reward
is the bottleneck. A process pool around `run_sample` is a natural
follow-up; left out of v1 for simplicity.

For the 10-hour coder GRPO budget in `MI300X_PLAN.md`:

- 10 h * 3600 s/h / ~240 s per step = ~150 steps
- With `save_steps=50`, three checkpoints.

That is a tight number; in practice Gary will want to either:
- Parallelize the sandbox via a pool, OR
- Use the `dryrun` tier for the first N steps to bootstrap the diff
  format, then switch to `quick` for refinement, OR
- Accept fewer steps and rely on the SFT warm-start to do heavy lifting.

## Exfil path

Outputs land at:

- SFT:  `outputs/coder-sft/final/` (PEFT adapter)
- GRPO: `outputs/coder-grpo/final/` (PEFT adapter)

Stream C's `scripts/mi300x/exfil.sh` rsyncs both back to
`/home/aegis/Projects/mud-puppy/outputs/mi300x/<run>/` on Gary's
workstation.

## Local verification

### Unit tests (no network, no real clones)

```bash
cd /home/aegis/Projects/mud-puppy
pytest tests/test_coder_sandbox.py tests/test_rl_verifier.py -v
```

Result (2026-04-23): 55 tests pass in under 4 seconds.

### Smoke test (needs one shallow clone of ggml-org/llama.cpp)

The local smoke test in `/tmp/sandbox_smoke/` exercises the end-to-end
path on the real `llamacpp-930e0210d1ba` commit (a two-line .gitignore
change) and the adversarial path (same diff against the wrong parent).
Result: rewards `[1.0, -1.0]` as expected. See the final report in
`AGENT_Cv1_HANDOFF.md` if present.

## Known limitations

1. **Serial sandbox.** The reward function runs samples one at a time.
   For throughput, wrap `run_sample` in a `concurrent.futures.ProcessPoolExecutor`
   sized to `num_generations` or the number of physical CPU cores,
   whichever is smaller. Left as Phase 7 work.
2. **Shallow-clone deepening.** If a commit's SHA is outside the
   500-commit shallow window, the sandbox does up to 3 deepenings.
   If the SHA is still unreachable we return `error="base_commit_unreachable"`
   and score -1.0. This is rare in the coder dataset (SHAs are recent-ish).
3. **No cgroup resource caps.** bwrap jails the filesystem and network
   but does not cap CPU or memory. A pathological patch that triggers
   an OOM on the pod would take down the GRPO process. Adding
   `systemd-run --scope -p MemoryMax=...` as a wrapper is a future fix;
   not critical for the MI300X run because the four target repos have
   known-bounded build costs.
4. **Test classifier heuristics.** The compile-vs-test-failure
   classifier uses regex markers on stdout/stderr. It is conservative
   (will call marginal cases "test failed" rather than "compile
   failed") so the reward does not penalize the model too heavily for
   ambiguous outputs.
5. **bwrap fallback is loud, not safe.** If bwrap is absent, the
   sandbox falls back to an unjailed subprocess and logs a warning.
   The MI300X bootstrap must install bwrap; double-check on Stream C.
6. **No gpt-oss MoE quant.** Experts stay bf16 on GPU; only attention
   Linears are wrapped with MXFP4. This matches `trial_gpt_oss_qlora.py`
   and the plan's constraint to stay native MXFP4, but it leaves memory
   headroom on the table. Phase 6 expert-quant work can claim that
   headroom later.
