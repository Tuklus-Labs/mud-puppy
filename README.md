# mud-puppy

mud-puppy is a ROCm-first LLM fine‑tuning framework inspired by [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl). It targets AMD GPUs and is optimized for the `GFX1100` architecture.

## Features

- **Full fine‑tuning** using the HuggingFace `Trainer` API
- **LoRA** and **QLoRA** via the `peft` library
- **GPTQ** post-training quantization and **QAT** via `torch.ao.quantization`
- **Preference tuning** with DPO/IPO/KTO/ORPO using the `trl` library
- **Reinforcement Learning** through GRPO on top of PPOTrainer
- **Multimodal support** (experimental)
- **Reward Modelling** / **Process Reward Modelling** for ranking models

mud-puppy currently expects datasets in JSONL chat format. Conversations are converted using the Qwen3 chat template and tokenized with HuggingFace tokenizers.

The framework is still experimental but includes working implementations of the major algorithms described above.

### Quick start

Install from source and launch a run:

```bash
pip install -e .

mud-puppy your-model your-dataset --method lora --output ./finetuned
```

## ROCm optimization

mud-puppy is designed to run efficiently on AMD GPUs. It defaults to `bf16` precision and enables gradient checkpointing to keep memory usage low on `GFX1100` cards. You can also opt into experimental `fp8` training by passing `--precision fp8` if your hardware supports it. Set `--compile` to enable `torch.compile` for additional speed.
The trainer configures ROCm with `PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128` at import time and enables TF32 matrix multiply for additional speed when available. Use `--num-workers` and `--preprocess-workers` to increase dataloader and tokenization parallelism.

