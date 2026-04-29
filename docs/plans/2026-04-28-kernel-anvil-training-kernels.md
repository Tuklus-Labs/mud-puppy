# TODO: kernel-anvil-generated training kernels for RDNA

Status: idea / unstarted

## Goal

Extend kernel-anvil (currently inference-only — MMVQ decode at batch=1) to
profile and generate optimal kernels for mud-puppy's training-time GEMM
shapes on RDNA (gfx1100, gfx1101, gfx1102, gfx1150, gfx1201, gfx1202).
Today mud-puppy ships hand-tuned Triton kernels for MXFP4 / INT4 training
(`mud_puppy/mxfp4_kernels.py`, `mud_puppy/int4_kernels.py`); kernel-anvil
should auto-tune them per (model arch, batch, seq, RDNA gen) instead of
relying on the hand-picked configs in those files.

## Why this isn't just inference autotune

Inference (kernel-anvil today): batch=1 GEMV, one direction, decode-path
only, MMVQ kernels in llama.cpp. Easy: 25 bucket cells × N quants.

Training is a different shape regime:

- **Forward GEMM** at training batch sizes (B*S = 1..32k tokens, hidden
  ~2k–8k, ffn ~8k–32k) — different occupancy regime than batch=1 GEMV.
- **Backward GEMM** — three matmuls per linear (input grad, weight grad,
  bias grad) with transposed shapes; weight grad is reduction-shaped
  (M,K accumulating into N).
- **Fused dequant+matmul** for MXFP4 / INT4 (already in
  `mxfp4_kernels.py`) — current configs are hand-picked at
  rows_per_block=1, BLOCK_M/N/K guesses; want auto-tuned per shape.
- **Optimizer step** (`mxfp4_optim.py`, stochastic rounding) — fused
  param update kernels with dequant/quant overhead.
- **Attention** — flash-attention via rocm_attn currently; out of scope
  for first cut, revisit if it's the bottleneck.

So the search space is bigger and the cost model differs (compute-bound,
not bandwidth-bound). The bucket scheme (5×5 N×K cells) probably stays,
but we add a batch-size axis and split forward/backward.

## Concrete entry points

The kernels mud-puppy needs auto-tuned, in priority order:

1. `mud_puppy/mxfp4_kernels.py` — MXFP4 fused dequant+matmul (forward)
   - Triton kernel `mxfp4_matmul` exists; configs are hand-picked.
   - Need: shape sweep across (B*S, K, N) for the actual training shapes
     of gpt-oss-20b and Qwen3-8B.
2. Backward matmul for MXFP4 — `mxfp4_train.py`. Dequant the packed
   weight, run dY @ W and X.T @ dY (weight grad). Both currently stock
   PyTorch.
3. INT4 path (`int4_kernels.py`) — same shape, different quant.
4. Optimizer kernel (`mxfp4_optim.py`) — stochastic-rounding update.

## kernel-anvil API extension

Today: `kernel-anvil gguf-optimize <gguf>`. Sketch for training:

    kernel-anvil mud-puppy-optimize <model_id> --batch 8 --seq 4096 \
        --quant mxfp4 --direction both \
        --output ~/.cache/anvil-train/<model_id>-mxfp4-b8s4096.json

- Pulls layer shapes from the model config (or a mud-puppy ModelProfile).
- Profiles forward + backward + optimizer shapes for the requested
  (B, S, quant) tuple on the active GPU.
- Writes a JSON config that mud-puppy reads at module import time
  (parallel to the smithy JSON for inference).
- Cache key includes RDNA gen so a config tuned on a 7900 XTX doesn't
  silently mis-apply on Strix Halo.

## Integration on the mud-puppy side

- New module `mud_puppy/anvil_loader.py`: load the JSON, expose
  `get_kernel_config(op, shape, dtype) -> dict` for the existing
  Triton kernels to consume.
- Patch `mxfp4_kernels.py` / `int4_kernels.py` to look up the config
  before falling back to current hand-picked defaults.
- `mud-puppy train ...` warns once at startup if no anvil config is
  cached for the (model, B, S, quant) tuple; suggests the
  `mud-puppy-optimize` command.

## Open questions

- Is the autotune cost (probably ~minutes per model+batch combo) worth
  paying once vs per-run? Cache + invalidate on (kernel hash, GPU,
  PyTorch+Triton+ROCm version).
- Backward weight-grad reduction shape: does the existing kernel-anvil
  bucket scheme work, or does that op need its own search space?
- MoE / Heretic post-train (gpt-oss MXFP4 experts) has irregular
  per-expert shapes — separate sweep?
- Worth a shared on-disk autotune cache between inference (smithy) and
  training (anvil-train) for shapes that actually overlap? Probably
  not — different occupancy regimes, configs won't transfer.

## Non-goals (for the first cut)

- Flash attention codegen (rocm_attn is good enough for now).
- CUDA / Metal support — kernel-anvil is RDNA-first, mud-puppy is too.
- JAX path (`jax_lora.py`, `jax_rl.py`) — TPU autotune is its own world.
