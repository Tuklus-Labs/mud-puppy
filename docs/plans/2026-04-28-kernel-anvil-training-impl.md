# Implementation plan: kernel-anvil → mud-puppy training-kernel autotune

Status: planned, ready to execute
Companion: `2026-04-28-kernel-anvil-training-kernels.md` (the original TODO)

## Survey findings (2026-04-28)

Three parallel surveys produced sharper scope than the TODO:

**kernel-anvil internals** (`/tmp/anvil-train-survey/anvil-internals.json`)
- REUSE_AS_IS: `verify.py` (~15 LOC tweak for tuple outputs).
- EXTEND: `sweep.py`, `profile.py`, `analyze.py`, `codegen.py`, `cli.py`.
- Dead end: `autoforge.py` (HIP standalone-binary path is inference-only).
- Runner adapter contract: a runner module exposes `setup() -> inputs`,
  `reference(inputs) -> tensor`, `run(inputs, **config) -> tensor`. Optional
  `BASELINE_CONFIG` and `DATA_BYTES` module attrs. Configs are splatted
  as kwargs.
- `rdna3.py` GPU table covers the RDNA chips mud-puppy targets. CDNA
  (MI210/MI300X) is out of scope per Gary's RDNA-only directive.

**mud-puppy kernel surface** (`/tmp/anvil-train-survey/mudpuppy-kernels.json`)
- Two Triton modules total: `mxfp4_kernels.py` (16 hand-picked configs)
  and `int4_kernels.py` (20 configs). Both use `@triton.autotune`
  already. Both have forward + grad-input only (no weight-grad — frozen
  base only). No fused optimizer kernel.
- Public API hooks: `triton_mxfp4_matmul`, `triton_mxfp4_grad_input`,
  `triton_int4_matmul`, `triton_int4_grad_input`.
- INT4 functions ALREADY accept `config: Optional[dict] = None` →
  anvil_loader integration on the INT4 side is essentially free.
- Production training shapes:
  - **gpt-oss-20b**: hidden=2880, intermediate=2880, 24 layers, 32
    experts (top_k=4). Attn fwd at (M=4096, N∈{4096,512,2880},
    K∈{2880,4096}), experts at irregular M with N=5760 K=2880 (gate_up)
    and N=2880 K=2880 (down).
  - **Qwen3-8B**: dense 4096/12288/36-layer. Four unique fwd shapes:
    (4096,4096,4096), (4096,1024,4096), (4096,12288,4096), (4096,4096,12288).
  - Production `scripts/gpt_oss_train_chain.sh` uses `--per-device-batch
    1 --grad-accum 16 --max-seq-length 2048` so M=2048 in real runs.
- Existing benchmark `benchmarks/bench_int4_triton.py` is the natural
  starting point (5 LLaMA/Ministral shapes; just needs Qwen3 + gpt-oss
  shapes added).
- Test gates: `test_mxfp4_kernels.py` and `test_int4_kernels.py` enforce
  1% rel-tol on outputs. Anvil-picked configs MUST pass these.
- Existing TODO in both kernels: `tl.trans` hurts RDNA3 coalesce —
  a structural-win flag worth investigating.

**Headroom benchmark** (`/tmp/anvil-train-survey/headroom.json`)
- VERDICT: **lean in.** Average 1.33x, max 1.57x (gpt-oss-20b FFN-up,
  M=2048 N=11520 K=2880) over the existing hand-picked autotune list.
  Smallest gap 1.05x (Qwen3 FFN-backward shape M=4096 N=4096 K=14336).
- PyTorch dequant+matmul reference: 1.37x–4.20x slower than the wide
  autotune winner — confirms the fused path matters at all.
- All 5 shapes converged on the SAME config that is NOT in
  `_FWD_CONFIGS`: `BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, GROUP_M=8,
  num_warps=8, num_stages=4`.

## Phase 0 — cheap win (target: same-day)

**Owner:** mud-puppy. No kernel-anvil changes.

Add the converging config (`BLOCK_M=128 BLOCK_N=64 BLOCK_K=32 GROUP_M=8
num_warps=8 num_stages=4`) to `_FWD_CONFIGS` in `mxfp4_kernels.py`. This
captures most of the benchmark's 1.33x average without any anvil
infrastructure.

Steps:
1. Edit `mud_puppy/mxfp4_kernels.py:73-92` to add the new config.
2. Run `pytest tests/test_mxfp4_kernels.py -v` to verify 1%-rel-tol
   correctness gate still passes on the new config.
3. Re-run `/tmp/headroom-bench.py` to confirm hand-picked list now
   matches wide autotune within ~3%.
4. Standalone PR titled "perf: add wide-autotune-discovered MXFP4 fwd
   config" with the headroom JSON as evidence.

Acceptance: `_FWD_CONFIGS` includes the new entry; correctness tests
pass; bench shows ≥1.25x average over the prior hand list on the 5
benchmark shapes.

## Phase 1 — anvil-train MVP

Two parallel work streams (kernel-anvil and mud-puppy) connected by a
JSON contract. The contract is fully specified below so both sides can
build independently.

### Contract: anvil-train JSON v1

```json
{
  "schema": "anvil-train/v1",
  "gpu": "gfx1100",
  "rocm_version": "7.1",
  "torch_version": "2.10.0+rocm7.1",
  "triton_version": "3.6.0",
  "kernel_hash": "sha256:<hex>",
  "model": "Qwen3-8B",
  "batch": 1,
  "seq": 4096,
  "ops": {
    "<op_name>": {
      "<m_bucket>,<n_bucket>,<k_bucket>": {
        "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
        "GROUP_M": 8, "num_warps": 8, "num_stages": 4,
        "speedup_vs_baseline": 1.33,
        "profiled_us": 18.4
      }
    }
  }
}
```

- `op_name` ∈ `{mxfp4_fwd, mxfp4_grad_input, int4_fwd, int4_grad_input}`.
- Bucket scheme: 3D, 5×5×5 cells.
  M boundaries: (1024, 4096, 8192, 16384) → 5 buckets.
  N boundaries: (1024, 4096, 8192, 16384) → 5 buckets.
  K boundaries: (1024, 4096, 8192, 16384) → 5 buckets.
- `kernel_hash` covers the kernel source file's content; mismatch means
  re-tune.
- Cache invalidation key: any of (gpu, rocm_version, torch_version,
  triton_version, kernel_hash) changing forces re-tune.
- Cache path: `~/.cache/anvil-train/<gpu>/<model>-<quant>-b<B>s<S>.json`.

### Phase 1a — kernel-anvil side (~3-5 days)

**Files added/modified:**
- NEW: `kernel_anvil/train_codegen.py` — JSON schema for v1, bucket
  helpers, atomic writers (mirrors `codegen.py` shape).
- NEW: `kernel_anvil/train_shapes.py` — extract op shapes from a
  Hugging Face transformers model config + (B, S) tuple. Output: list
  of `(op_name, M, N, K)`. Supports `LlamaConfig`, `Qwen3Config`, and
  the gpt-oss MoE config.
- NEW: `kernel_anvil/train_param_space.py` — config grid for training
  param search: BLOCK_M ∈ {32,64,128,256}, BLOCK_N ∈ {32,64,128,256},
  BLOCK_K ∈ {32,64,128}, GROUP_M ∈ {1,4,8,16}, num_warps ∈ {4,8},
  num_stages ∈ {2,3,4}. RDNA3-aware (skip configs that overflow
  VGPR/LDS).
- EXTEND: `kernel_anvil/cli.py` — new `train-optimize` subcommand:
  `kernel-anvil train-optimize <model_id> --quant {mxfp4,int4} --batch
  N --seq M [--ops ...] [--output <path>]`.
- EXTEND: `kernel_anvil/verify.py` — accept tuple outputs (grad-input
  returns activations + maybe backward state).
- TESTS: `tests/test_train_codegen.py`, `tests/test_train_shapes.py`,
  `tests/test_train_cli.py`.

### Phase 1b — mud-puppy side (~2-3 days)

**Files added/modified:**
- NEW: `mud_puppy/anvil_loader.py` — load anvil-train JSON, expose
  `get_kernel_config(op: str, M: int, N: int, K: int) -> dict | None`.
  Cache loaded JSONs in-process. Validate kernel_hash against current
  source on load; warn-and-fallback if mismatched.
- NEW: `mud_puppy/anvil_runner.py` — runner adapter modules satisfying
  kernel-anvil's runner contract for each op. Four runners:
  `mxfp4_fwd_runner.py`, `mxfp4_grad_input_runner.py`,
  `int4_fwd_runner.py`, `int4_grad_input_runner.py`. Each exposes
  `setup`, `reference`, `run`, `BASELINE_CONFIG`, `DATA_BYTES`.
- EXTEND: `mud_puppy/mxfp4_kernels.py` — add `config: Optional[dict] =
  None` parameter to `triton_mxfp4_matmul` and
  `triton_mxfp4_grad_input` (mirrors what INT4 already has). When
  `config` is provided, bypass the `@triton.autotune` decorator and
  use the provided BLOCK_M/N/K/etc directly.
- EXTEND: `mud_puppy/int4_kernels.py` — implement the existing-but-
  unused `config` slot to actually take effect.
- INTEGRATION (callers): `mud_puppy/mxfp4_train.py`,
  `mud_puppy/int4_kernels.py` callers — at construction time, query
  `anvil_loader.get_kernel_config(...)` once per layer (or per shape)
  and stash the result; pass it on each forward/backward call. This
  avoids per-step JSON parsing.
- TESTS:
  - `tests/test_anvil_loader.py` — JSON load, hash validation, cache
    invalidation, fallback.
  - Extend `tests/test_mxfp4_kernels.py` and `tests/test_int4_kernels.py`
    to verify anvil-picked configs still pass the 1% rel-tol gate
    against a synthetic anvil-train JSON.
- DOCS: brief section in README and `docs/plans/...kernel-anvil-training-kernels.md`
  status flip to "implemented".

## Phase 2 — explicit non-goals for this iteration

- INT4 grad-input is wired but not deeply optimized (already cheap).
- gpt-oss MoE per-expert irregular-M shapes — handled by the bucket
  scheme; no per-expert config (would explode the search space).
- Optimizer kernel — pure PyTorch today; not worth fusing yet.
- Flash attention codegen — `rocm_attn` is fine.
- CDNA (MI210/MI300X) — Gary scoped to RDNA. Add later via a fresh
  GPU-spec module.
- The `tl.trans` deferred-structural-win — separate investigation,
  not part of this PR.

## Risk register

| Risk | Mitigation |
|---|---|
| anvil-train autotune wallclock per (model, B, S) is too long for daily-driver use | Cache invalidation on (kernel hash, GPU, ROCm, Torch, Triton) only — re-tune is rare. First run accepts ~15 min cost (matches headroom bench). |
| Anvil-picked config fails the 1% rel-tol gate | Verify step is part of the sweep — kernel-anvil already does allclose checks via `verify.py`. Configs that don't pass the gate are filtered before being written to JSON. |
| Triton compile cost is the bottleneck (162 configs × 5 shapes ≈ 16 min wall in headroom bench) | Cap config grid by RDNA VGPR/LDS limits up front in `train_param_space.py`; aim for ≤30 candidates per shape. |
| MoE per-expert irregular shapes don't fit the bucket scheme | Out of scope for v1; bucket-fallback semantics give a sensible default. |
| Phase 1a + 1b race on the JSON contract | Contract is frozen above. Both sides build to it independently and test against fixture JSONs. |

## Acceptance criteria

- `kernel-anvil train-optimize <model> --quant mxfp4 --batch 1 --seq
  4096` runs end-to-end and writes a v1 JSON to
  `~/.cache/anvil-train/gfx1100/<model>-mxfp4-b1s4096.json`.
- mud-puppy starts a training step and `anvil_loader` selects the
  cached configs (verified via a logged "anvil: applied N configs"
  line at startup).
- `tests/test_mxfp4_kernels.py` and `tests/test_int4_kernels.py` pass
  with anvil-applied configs (1% rel-tol gate).
- A second `headroom-bench.py` run shows mud-puppy's anvil-applied path
  is within 5% of the wide-autotune winner across the 5 benchmark
  shapes.
- A real `scripts/gpt_oss_train_chain.sh` step shows ≥1.20x throughput
  over master.

## Build dispatch

Three parallel subagents:

1. **Phase 0 (mud-puppy)** — add converging config to `_FWD_CONFIGS`,
   validate, open standalone PR.
2. **Phase 1a (kernel-anvil)** — train_codegen, train_shapes,
   train_param_space, train-optimize CLI, verify.py extension, tests.
   Build to JSON contract above.
3. **Phase 1b (mud-puppy)** — anvil_loader, anvil_runner, kernel
   parameter passthrough, integration in callers, tests. Build to JSON
   contract above using fixture JSONs initially; integration with real
   1a output happens after both land.

Each subagent owns its own commit + PR. Phase 0 ships first
(independent). Phase 1a and 1b ship next, mergeable in either order.
The end-to-end smoke test (acceptance criterion #5) runs after both 1a
and 1b are merged.
