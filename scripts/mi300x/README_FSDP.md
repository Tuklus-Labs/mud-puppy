# MI300X FSDP / multi-GPU workflow

This is the sibling to `README.md`. The main README covers the
"four separate jobs, one GPU each" layout that runs reviewer + coder +
ablations in parallel. This document covers the other case: **one large
training job sharded across all 8 MI300X GPUs** via FSDP.

Use this path when:

- The model doesn't fit on a single 192 GB GPU in bf16 (70B full fine-tune
  with optimizer states ≈ 70 × 18 = 1260 GB, so you need sharding)
- You want maximum throughput on a single run
- You're running a smaller model (≤14B) but want 8x data parallelism for
  wall-clock speed

Don't use this path if you're running multiple independent small jobs;
the regular `launch.sh` is better for that.

## Quick start

From the pod, after `setup_pod.sh`:

```bash
cd /scratch/work/mud-puppy
./scripts/mud-puppy-launch --nproc 8 -- \
  meta-llama/Llama-3.3-70B-Instruct \
  training_data_sets/reviewer/reviewer-sft.jsonl \
  --method full --precision bf16 \
  --fsdp full_shard --fsdp-wrap-class LlamaDecoderLayer \
  --fsdp-activation-checkpointing \
  --use-gradient-checkpointing \
  --batch-size 1 --gradient-accumulation 16 \
  --learning-rate 1e-5 --epochs 1 \
  --output ./outputs/llama70b-sft
```

`--nproc` defaults to the detected GPU count if omitted.

## Flag decoder

| Flag | Meaning |
|------|---------|
| `--fsdp full_shard` | Shard params + grads + optimizer state across all GPUs. Matches DeepSpeed ZeRO-3. Right default for 70B+ on MI300x8. |
| `--fsdp shard_grad_op` | Shard grads + optimizer only. Replicates params. ZeRO-2 equivalent. Use when params fit per-GPU but optimizer doesn't. |
| `--fsdp hybrid_shard` | Shard within a node, replicate across nodes. Multi-node only. |
| `--fsdp no_shard` | DDP. Equivalent to omitting `--fsdp` entirely plus `--distributed`. |
| `--fsdp-wrap-class LlamaDecoderLayer` | Wrap each decoder block as one FSDP unit. **Strongly preferred** over `--fsdp-min-num-params` because unit boundaries align with activation-recomputation boundaries. |
| `--fsdp-activation-checkpointing` | Recompute activations inside each FSDP unit. Big memory savings; ~15-20% time cost. |
| `--fsdp-cpu-offload` | Offload sharded params to CPU. Rarely needed on MI300 (1.5 TB HBM pool). |
| `--distributed-backend nccl` | Default. PyTorch maps `nccl` to RCCL on ROCm transparently. |

## Model class hints for `--fsdp-wrap-class`

| Model family | Class name |
|--------------|------------|
| LLaMA 2/3/3.1/3.3 | `LlamaDecoderLayer` |
| Mistral | `MistralDecoderLayer` |
| Mixtral (MoE) | `MixtralDecoderLayer` |
| Qwen2 / Qwen2.5 | `Qwen2DecoderLayer` |
| Qwen3 | `Qwen3DecoderLayer` |
| Phi-3 | `Phi3DecoderLayer` |
| Gemma 2/3 | `Gemma2DecoderLayer` / `Gemma3DecoderLayer` |
| GPT-OSS | `GPTOSSDecoderLayer` |

If you're unsure, look at `model.model.layers[0].__class__.__name__` on
the loaded base model.

## What the launcher sets for you

`scripts/mud-puppy-launch` exports these RCCL defaults before invoking
torchrun; override any of them by pre-exporting:

```
NCCL_PROTO=Simple,LL,LL128        # MI300 xGMI sweet spot
NCCL_ALGO=Tree,Ring
NCCL_P2P_DISABLE=0                # enable xGMI P2P (default on anyway)
RCCL_MSCCL_ENABLE=1               # MSCCL collective fast path
NCCL_DEBUG=WARN                   # INFO is too chatty under 8-way runs
TOKENIZERS_PARALLELISM=false      # avoids HF warning spam under DDP
```

These are set via `os.environ.setdefault` in `trainer.setup_distributed`,
so any value you export beats the default.

## Sizing cheat sheet

With `--fsdp full_shard --fsdp-activation-checkpointing`:

| Model | Per-GPU memory (bf16 full FT, 8x MI300X) | Notes |
|-------|------------------------------------------|-------|
| 7B | ~20 GB | massive headroom; increase batch size |
| 13B | ~35 GB | comfortable |
| 34B | ~85 GB | comfortable |
| 70B | ~155 GB | tight; batch=1, grad_accum=16+ |
| 120B MoE | varies | Mixtral-style: active params matter more than total |
| 180B | doesn't fit | need CPU offload or more nodes |

Memory numbers assume batch=1, seq=2048. Double for seq=4096, etc.

## Multi-node (two MI300X pods)

```bash
# on node 0 (master)
./scripts/mud-puppy-launch --nproc 8 --nnodes 2 --node-rank 0 \
  --master-addr $NODE0_IP --master-port 29500 -- \
  <model> <data> --fsdp hybrid_shard <other args>

# on node 1
./scripts/mud-puppy-launch --nproc 8 --nnodes 2 --node-rank 1 \
  --master-addr $NODE0_IP --master-port 29500 -- \
  <model> <data> --fsdp hybrid_shard <other args>
```

`hybrid_shard` shards across the 8 GPUs inside each node (fast xGMI) and
replicates across nodes (slower network). Pure `full_shard` across nodes
is correct but ~2-4x slower due to inter-node all-gather.

## Verifying the shard before you burn compute

A smoke-test pass that builds the model, applies FSDP, runs one forward
+ backward, and shuts down:

```bash
./scripts/mud-puppy-launch --nproc 8 -- \
  <small model, e.g. Qwen/Qwen2.5-7B> \
  tests/fixtures/tiny.jsonl \
  --method full --precision bf16 \
  --fsdp full_shard --fsdp-wrap-class Qwen2DecoderLayer \
  --batch-size 1 --gradient-accumulation 1 \
  --epochs 1 --output /tmp/fsdp-smoke
```

If this completes in under ~5 minutes the distributed path is healthy
and you can kick off the real run with confidence.

## Troubleshooting

**RCCL hangs during init**: check that all 8 GPUs are visible
(`rocm-smi --showid`) and that no previous run left a stale rendezvous.
Kill any lingering `python -m mud_puppy.cli` processes on the pod.

**OOM with 70B at full_shard**: enable `--fsdp-activation-checkpointing`
AND `--use-gradient-checkpointing`. The two compose: FSDP checkpointing
saves activations at FSDP unit boundaries; gradient_checkpointing saves
activations at transformer-block boundaries. Both on = lowest memory.

**"backward_prefetch ignored" warning**: harmless, HF normalizes the key.

**Saved checkpoint is per-rank sharded**: FSDP saves shards by default.
Either load with the same FSDP config, or consolidate with the `torch.distributed.checkpoint`
consolidator before pulling back. `download_checkpoints.sh` does this
automatically for the regular single-GPU jobs; for FSDP runs, see the
PyTorch FSDP checkpoint docs.
