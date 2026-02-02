# mud-puppy

**ROCm-First LLM Fine-tuning Framework**

mud-puppy is a ROCm-first LLM fine-tuning framework inspired by [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl). It targets AMD GPUs and is optimized for the `GFX1100` architecture, but should run on any recent ROCm-capable GPU.

Read the [design whitepaper](https://tukluslabs.com/whitepaper-mud-puppy.html) for architecture details.

## Features

- **Full fine-tuning** using the HuggingFace `Trainer` API
- **LoRA** via the `peft` library and **QLoRA** with a built-in 4-bit quantizer
  - A lightweight module handles 4-bit quantization on all GPUs, so no `bitsandbytes` dependency is required
- **Flash Attention** via the `rocm_attn` helpers (`flash_attention` and `FlashMHA`), backed by PyTorch's `scaled_dot_product_attention` on ROCm
- **GPTQ** post-training quantization and **QAT**
  - GPTQ models are saved to the `gptq/` folder and use a simple ROCm-friendly int4 implementation, or `auto-gptq` on CUDA if available
  - QAT uses a simple ROCm module and saves an int8-ish model for CPU/ROCm inference
- **ROCm kernels** including qgemm/fbgemm, quantization helpers, and a quantized layernorm (naive but portable implementations)
- **Preference tuning** with DPO, IPO, KTO, and ORPO using the `trl` library
- **Reinforcement Learning** through native `GRPOTrainer` (with PPO fallback)
- **Multimodal support** (experimental flag; currently routes through the standard trainer)
- **Reward Modelling** / **Process Reward Modelling** for ranking models

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mud-puppy.git
cd mud-puppy

# Install in development mode
pip install -e .

# Or install with CUDA extras (for auto-gptq support)
pip install -e ".[cuda]"
```

### Dependencies

- Python >= 3.9
- PyTorch (ROCm or CUDA build)
- transformers
- datasets
- peft
- trl

## Quick Start

### Supervised Fine-Tuning (SFT)

```bash
# Full fine-tuning
mud-puppy meta-llama/Llama-3-8B data.jsonl --method full --output ./outputs

# LoRA fine-tuning
mud-puppy meta-llama/Llama-3-8B data.jsonl --method lora --output ./lora-outputs

# QLoRA (4-bit quantized base + LoRA)
mud-puppy meta-llama/Llama-3-8B data.jsonl --method qlora --output ./qlora-outputs

# QLoRA with MXFP4 (block-wise 4-bit, better for varying weight distributions)
mud-puppy meta-llama/Llama-3-8B data.jsonl --method qlora --quant-backend mxfp4 --output ./qlora-mxfp4
```

### Preference Tuning

```bash
# DPO (Direct Preference Optimization)
mud-puppy your-model prefs.jsonl --method preference --preference dpo --output ./dpo

# IPO (Identity Preference Optimization)
mud-puppy your-model prefs.jsonl --method preference --preference ipo --output ./ipo

# KTO (Kahneman-Tversky Optimization)
mud-puppy your-model prefs.jsonl --method preference --preference kto --output ./kto

# ORPO (Odds Ratio Preference Optimization)
mud-puppy your-model prefs.jsonl --method preference --preference orpo --output ./orpo
```

### Reinforcement Learning

```bash
# GRPO training
mud-puppy your-model prompts.jsonl --method rl --output ./grpo
```

### Reward Modeling

```bash
# Standard reward model
mud-puppy your-model rewards.jsonl --method rm --output ./reward-model

# Process reward model
mud-puppy your-model prm_data.jsonl --method prm --output ./prm-model
```

### Quantization

```bash
# GPTQ post-training quantization
mud-puppy your-model data.jsonl --method gptq --output ./gptq-model

# QAT (Quantization-Aware Training)
mud-puppy your-model data.jsonl --method qat --output ./qat-model
```

## Dataset Formats

mud-puppy supports several JSONL dataset formats:

### Chat Format (recommended for SFT)

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

### Simple Text Format

```json
{"text": "This is a training example."}
{"text": "Another example for fine-tuning."}
```

### Instruction/Response Format

```json
{"instruction": "Explain quantum computing", "response": "Quantum computing uses..."}
{"input": "Translate to French: Hello", "output": "Bonjour"}
```

### Prompt/Completion Format

```json
{"prompt": "The capital of France is", "completion": " Paris."}
```

### Preference Format (for DPO/IPO/KTO/ORPO)

```json
{"prompt": "What is 2+2?", "chosen": "The answer is 4.", "rejected": "I don't know."}
```

### Reward Format

```json
{"text": "This is a good response", "label": 0.95}
{"prompt": "Question", "chosen": "Good answer", "rejected": "Bad answer"}
```

### RL Format (for GRPO)

```json
{"prompt": "Write a haiku about coding"}
{"prompt": "Explain machine learning briefly"}
```

## CLI Reference

```
mud-puppy MODEL DATASET [OPTIONS]

Positional Arguments:
  MODEL                     Model name or path (HuggingFace hub or local)
  DATASET                   Dataset path (JSONL file)

Training Method:
  --method METHOD           Training method: full, lora, qlora, gptq, qat,
                           preference, rl, multimodal, rm, prm (default: full)
  --preference PREF         Preference algorithm: dpo, ipo, kto, orpo

Output:
  --output DIR              Output directory (default: ./outputs)

Precision:
  --precision {fp16,bf16,fp8}   Training precision (default: bf16)
  --compile                     Enable torch.compile for speed

Hyperparameters:
  --batch-size N            Per-device batch size
  --gradient-accumulation N Gradient accumulation steps
  --learning-rate LR        Learning rate
  --epochs N                Number of training epochs
  --lr-scheduler TYPE       Scheduler: linear, cosine, cosine_with_restarts, polynomial

LoRA Options:
  --lora-targets MODULES    Comma-separated list of target modules
  --merge-lora              Merge LoRA weights after training
  --merge-precision PREC    Precision for merged model: fp16, bf16, fp32
  --quant-backend BACKEND   Quantization backend for QLoRA: int4 (default) or mxfp4

Memory Optimization:
  --device-map MAP          Device map: auto, pipeline, or custom
  --stream                  Stream layers from CPU to GPU on demand
  --zero-offload            Offload optimizer states to CPU
  --tokens-per-batch N      Dynamic batching by token count

Data Processing:
  --no-chat-template        Disable chat template application
  --trust-remote-code       Allow custom model code from Hub
  --num-workers N           Dataloader workers
  --preprocess-workers N    Dataset preprocessing workers

Training Control:
  --resume                  Resume from last checkpoint
  --early-stopping N        Stop if no improvement for N evaluations
  --log-with BACKEND        Logging: none, tensorboard, wandb

Distributed:
  --distributed             Enable distributed training
  --local-rank N            Process rank for distributed training
```

## Python API

```python
from mud_puppy import TrainingConfig, run_training

# Configure training
config = TrainingConfig(
    model_name_or_path="meta-llama/Llama-3-8B",
    dataset_path="./data/chat.jsonl",
    output_dir="./outputs",
    finetuning_method="lora",  # full, lora, qlora, gptq, qat, preference, rl, rm, prm
    precision="bf16",
    batch_size=4,
    gradient_accumulation=4,
    learning_rate=2e-5,
    num_epochs=3,
    lora_r=8,
    lora_alpha=16,
    use_gradient_checkpointing=True,
)

# Run training
run_training(config)
```

### Quantization API

```python
from mud_puppy import quantize_model_4bit, quantize_model_gptq, apply_qat, convert_qat
from mud_puppy.mxfp4_rocm import quantize_model_mx4

# 4-bit quantization (for QLoRA) - row-wise scales
model = quantize_model_4bit(model, dtype=torch.bfloat16)

# MXFP4 quantization (for QLoRA) - block-wise scales, better for varying weight distributions
model = quantize_model_mx4(model, block_size=32)

# GPTQ quantization
model = quantize_model_gptq(model, bits=4)

# QAT
model = apply_qat(model, bits=8)  # Enable QAT during training
model = convert_qat(model, bits=8)  # Convert to quantized after training
```

### Flash Attention API

```python
from mud_puppy import flash_attention, FlashMHA

# Functional API
output = flash_attention(q, k, v, causal=True, dropout_p=0.1)

# Module API
mha = FlashMHA(embed_dim=768, num_heads=12, dropout=0.1)
output = mha(x, attention_mask=mask, causal=True)
```

### ROCm Kernels

```python
from mud_puppy.rocm_kernels import (
    quantize_per_tensor,
    quantize_per_channel,
    dequantize,
    qgemm,
    fbgemm,
    quantized_layernorm,
)

# Quantize a tensor
qtensor, scale, zero_point = quantize_per_tensor(tensor, bits=8)

# Dequantize
tensor = dequantize(qtensor, scale, zero_point)

# Quantized matrix multiply
result = qgemm(a_q, a_scale, b_q, b_scale)

# Fused GEMM + bias + activation
result = fbgemm(a_q, a_scale, b_q, b_scale, bias=bias, activation="relu")
```

## Interactive Mode

mud-puppy includes an interactive CLI for guided configuration:

```bash
mud-puppy-interactive
```

This launches a REPL where you can type `start` to configure a training run interactively.

## Memory Optimization

### For Large Models

```bash
# Use streaming to train models larger than GPU memory
mud-puppy large-model data.jsonl --stream --method lora

# Combine with optimizer offloading
mud-puppy large-model data.jsonl --stream --zero-offload --method lora

# Use device_map for automatic model parallelism
mud-puppy large-model data.jsonl --device-map auto --method full
```

### For Limited VRAM

```bash
# Use QLoRA with gradient checkpointing (enabled by default)
mud-puppy model data.jsonl --method qlora --batch-size 1 --gradient-accumulation 16

# Dynamic batching to maximize GPU utilization
mud-puppy model data.jsonl --tokens-per-batch 4096 --method lora
```

### ZeRO-Offload (ROCm-Native)

Offload optimizer states to CPU RAM for training larger models. This is a pure-PyTorch
implementation that works on ROCm without requiring DeepSpeed.

```bash
# Enable ZeRO-Offload for optimizer state offloading
mud-puppy large-model data.jsonl --zero-offload --method qlora

# Combined with other memory optimizations for maximum model size
mud-puppy meta-llama/Llama-3-70B data.jsonl \
    --method qlora \
    --zero-offload \
    --batch-size 1 \
    --gradient-accumulation 32 \
    --gradient-checkpointing
```

**Memory savings with ZeRO-Offload:**

| Model | Without Offload | With Offload | Savings |
|-------|-----------------|--------------|---------|
| 7B (QLoRA) | ~12GB VRAM | ~8GB VRAM | 33% |
| 20B (QLoRA) | ~28GB VRAM | ~16GB VRAM | 43% |
| 70B (QLoRA) | ~80GB VRAM | ~40GB VRAM | 50% |

The offloading moves AdamW momentum and variance tensors (2x model size) to CPU RAM
between training steps. With 192GB system RAM, you can train models that would
otherwise exceed GPU memory.

**Programmatic usage:**

```python
from mud_puppy import wrap_optimizer_for_offload

base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer = wrap_optimizer_for_offload(base_optimizer)

# Use like normal optimizer
optimizer.zero_grad()
loss.backward()
optimizer.step()  # States automatically offloaded after step
```

## ROCm Optimization

mud-puppy is designed to run efficiently on AMD GPUs:

- Defaults to `bf16` precision with gradient checkpointing
- Sets `PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128` for better memory allocation
- Uses `torch.nn.functional.scaled_dot_product_attention` for flash attention on ROCm
- All quantization kernels are pure PyTorch, no vendor-specific code required

For CUDA users, mud-puppy works seamlessly and can use `auto-gptq` for GPTQ if installed.

### Experimental FP8 Support

```bash
mud-puppy model data.jsonl --precision fp8 --method full
```

Requires PyTorch with FP8 support.

## Version History

### v0.3.0

- Native `GRPOTrainer` support for reinforcement learning
- `RewardTrainer` and `PRMTrainer` integration
- `StreamWrapper` for layer-by-layer GPU streaming
- `DynamicBatchSampler` for token-budget batching
- Pipeline parallelism with `--device-map pipeline`
- Early stopping with `--early-stopping`
- LoRA weight merging with `--merge-lora`
- IPO support via DPO loss variant
- Improved dataset format detection

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) for inspiration
- [HuggingFace](https://huggingface.co/) for transformers, datasets, peft, and trl
- AMD for ROCm and continued open-source GPU compute support
