# mud-puppy

mud-puppy is a ROCm-first LLM fine‑tuning framework inspired by [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl). It targets AMD GPUs and is optimized for the `GFX1100` architecture.

## Features

- **Full fine‑tuning** using the HuggingFace `Trainer` API
- **LoRA** and **QLoRA** via the `peft` library
- **GPTQ** and **QAT** hooks (placeholders)
- **Preference tuning** with support for DPO, IPO, KTO and ORPO (placeholders)
- **Reinforcement Learning** with GRPO (placeholder)
- **Multimodal support** (placeholder)
- **Reward Modelling** / **Process Reward Modelling** (placeholder)

mud-puppy currently expects datasets in JSONL chat format. Conversations are converted using the Qwen3 chat template and tokenized with HuggingFace tokenizers.

The framework is still experimental and many advanced algorithms are yet to be implemented.

### Quick start

Install from source and launch a run:

```bash
pip install -e .

mud-puppy your-model your-dataset --method lora --output ./finetuned
```

## ROCm optimization

mud-puppy is designed to run efficiently on AMD GPUs. It defaults to `bf16` precision and enables gradient checkpointing to keep memory usage low on `GFX1100` cards.
The trainer configures ROCm with `PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128` and
enables TF32 matrix multiply for additional speed when available.

