# mud-puppy

mud-puppy is a ROCm-first LLM fine‑tuning framework inspired by [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl). It targets AMD GPUs and is optimized for the `GFX1100` architecture.

## Features

- **Full fine‑tuning** using the HuggingFace `Trainer` API
- **LoRA** via the `peft` library and **QLoRA** with a built-in 4-bit quantizer
  - A lightweight module handles 4-bit quantization on all GPUs, so no `bitsandbytes` dependency is required
- **Flash Attention** via the `rocm_attn` helpers (`flash_attention` and `FlashMHA`)
- **GPTQ** post-training quantization and **QAT**
  - GPTQ models are saved to the `gptq/` folder and use `auto-gptq` on CUDA or a simple ROCm implementation when running on AMD GPUs
  - QAT uses a simple ROCm module and saves an int8 model for CPU inference
- **ROCm kernels** including qgemm/fbgemm, quantization helpers, and a quantized layernorm
- **Preference tuning** with DPO using the `trl` library
- **Reinforcement Learning** through GRPO on top of PPOTrainer
- **Multimodal support** (experimental)
- **Reward Modelling** / **Process Reward Modelling** for ranking models

mud-puppy currently expects datasets in JSONL chat format. By default it applies a chat template if the tokenizer supports it, but this can be disabled with `--no-chat-template`.

### New features

Version 0.3 expands on this with early stopping, dynamic batching and model parallelism. You can train huge models with `--device-map auto`, keep batches to a token budget via `--tokens-per-batch`, choose a scheduler with `--lr-scheduler`, and stop early using `--early-stopping`.
Pipeline parallelism can be enabled with `--device-map pipeline` when your PyTorch build includes `torch.distributed.pipeline`.

Streaming mode is available with `--stream` to offload layers to CPU swap and move them to the GPU one at a time.

Optimizer states can be offloaded to CPU with `--zero-offload` to further reduce GPU memory usage.
Adapters from LoRA or QLoRA runs can be merged back into the base model using `--merge-lora` and saved in your choice of precision with `--merge-precision`.

The framework is still experimental but includes working implementations of the major algorithms described above.


### Quick start

Install from source and launch a run:

```bash
pip install -e .

mud-puppy your-model your-dataset --method lora --output ./finetuned
```
Add `--trust-remote-code` if the model requires custom code from the Hub, and
use `--no-chat-template` to skip template formatting during preprocessing.

## ROCm optimization

mud-puppy is designed to run efficiently on AMD GPUs. It defaults to `bf16` precision and enables gradient checkpointing to keep memory usage low on `GFX1100` cards. You can also opt into experimental `fp8` training by passing `--precision fp8` if your hardware supports it. Set `--compile` to enable `torch.compile` for additional speed.
The trainer configures ROCm with `PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128` at import time and enables TF32 matrix multiply for additional speed when available. Use `--num-workers` and `--preprocess-workers` to increase dataloader and tokenization parallelism.


