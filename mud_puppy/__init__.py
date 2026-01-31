"""mud-puppy: A ROCm-first LLM fine-tuning framework.

This package exposes a small public API intended primarily for programmatic
use from Python. The command-line entry points are defined in ``pyproject.toml``.

High-level usage::

    from mud_puppy import TrainingConfig, run_training

    config = TrainingConfig(
        model_name_or_path="meta-llama/Llama-3-8B",
        dataset_path="./data/chat.jsonl",
        output_dir="./outputs",
        finetuning_method="lora",
    )

    run_training(config)

Lower-level helpers for quantization and ROCm kernels are also re-exported
for convenience, but are considered experimental and may change.
"""

from importlib.metadata import version, PackageNotFoundError

try:  # pragma: no cover - metadata lookup
    __version__ = version("mud-puppy")
except PackageNotFoundError:  # pragma: no cover - editable installs
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "TrainingConfig",
    "run_training",
    "quantize_model_4bit",
    "flash_attention",
    "FlashMHA",
    "apply_qat",
    "convert_qat",
    "quantize_model_gptq",
    "save_quantized",
    "load_quantized",
    # JAX/Flax RL
    "GRPOTrainer",
    "GRPOConfig",
    "run_jax_grpo_training",
    # JAX/Flax LoRA
    "LoRAConfig",
    "QLoRAConfig",
    "LoRADense",
    "LoRAAdapter",
    "QLoRADense",
    "merge_lora_params",
    "init_lora_from_pretrained",
    # ZeRO-Offload (ROCm-native)
    "CPUOffloadOptimizer",
    "OffloadConfig",
    "wrap_optimizer_for_offload",
]

from .config import TrainingConfig


def run_training(*args, **kwargs):
    """Lazy import and run the training routine.

    This indirection keeps import time light and avoids importing heavy
    dependencies (e.g. Transformers, TRL) until needed.
    """

    from .trainer import run_training as _run_training

    return _run_training(*args, **kwargs)


def quantize_model_4bit(*args, **kwargs):
    """Quantize a model's linear layers to 4-bit weights on ROCm.

    This is a lightweight, bitsandbytes-free quantizer suitable for QLoRA-style
    fine-tuning. It operates on standard ``nn.Linear`` layers.
    """

    from .bnb_rocm import quantize_model_4bit as _quantize_model_4bit

    return _quantize_model_4bit(*args, **kwargs)


def flash_attention(*args, **kwargs):
    """FlashAttention-style wrapper around ``scaled_dot_product_attention``.

    See :func:`mud_puppy.rocm_attn.flash_attention` for details.
    """

    from .rocm_attn import flash_attention as _flash_attention

    return _flash_attention(*args, **kwargs)


def FlashMHA(*args, **kwargs):
    """Multi-head attention module built on top of :func:`flash_attention`."""

    from .rocm_attn import FlashMHA as _FlashMHA

    return _FlashMHA(*args, **kwargs)


def apply_qat(*args, **kwargs):
    """Enable quantization-aware training (QAT) on a model.

    Wraps linear layers with QAT-friendly modules that simulate int8
    quantization during training.
    """

    from .qat_rocm import apply_qat as _apply_qat

    return _apply_qat(*args, **kwargs)


def convert_qat(*args, **kwargs):
    """Convert a QAT-trained model to a quantized inference form.

    Currently produces int8-like linear layers backed by float weights.
    """

    from .qat_rocm import convert_qat as _convert_qat

    return _convert_qat(*args, **kwargs)


def quantize_model_gptq(*args, **kwargs):
    """Post-training GPTQ-style quantization helper.

    On ROCm, this uses a simple int4 quantized linear implementation. On
    CUDA, users may prefer the more feature-complete ``auto-gptq`` package.
    """

    from .gptq_rocm import quantize_model_gptq as _quantize_model_gptq

    return _quantize_model_gptq(*args, **kwargs)


def save_quantized(*args, **kwargs):
    """Save a quantized model in a Hugging Face-compatible format."""

    from .gptq_rocm import save_quantized as _save_quantized

    return _save_quantized(*args, **kwargs)


def load_quantized(*args, **kwargs):
    """Load a previously quantized model.

    This is a thin wrapper around ``AutoModelForCausalLM.from_pretrained`` and
    friends, depending on the model class passed in.
    """

    from .gptq_rocm import load_quantized as _load_quantized

    return _load_quantized(*args, **kwargs)


# JAX/Flax RL exports (lazy imports to avoid heavy dependencies)


def GRPOTrainer(*args, **kwargs):
    """JAX/Flax GRPO (Group Relative Policy Optimization) trainer.

    GRPO is a reinforcement learning algorithm similar to PPO but with
    group-level relative rewards - generating multiple completions per prompt
    and computing advantages relative to the group mean.

    See :class:`mud_puppy.jax_rl.GRPOTrainer` for full documentation.
    """

    from .jax_rl import GRPOTrainer as _GRPOTrainer

    return _GRPOTrainer(*args, **kwargs)


def GRPOConfig(*args, **kwargs):
    """Configuration for GRPO training.

    See :class:`mud_puppy.jax_rl.GRPOConfig` for full documentation.
    """

    from .jax_rl import GRPOConfig as _GRPOConfig

    return _GRPOConfig(*args, **kwargs)


def run_jax_grpo_training(*args, **kwargs):
    """Run GRPO training using JAX backend.

    This is the JAX equivalent of :func:`run_training` for reinforcement
    learning with GRPO.

    See :func:`mud_puppy.jax_rl.run_jax_grpo_training` for full documentation.
    """

    from .jax_rl import run_jax_grpo_training as _run_jax_grpo_training

    return _run_jax_grpo_training(*args, **kwargs)


# JAX/Flax LoRA exports (lazy imports to avoid heavy dependencies)


def LoRAConfig(*args, **kwargs):
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning.

    See :class:`mud_puppy.jax_lora.LoRAConfig` for full documentation.
    """

    from .jax_lora import LoRAConfig as _LoRAConfig

    return _LoRAConfig(*args, **kwargs)


def QLoRAConfig(*args, **kwargs):
    """Configuration for QLoRA (quantized LoRA) fine-tuning.

    Extends LoRAConfig with INT4 base weight quantization settings.

    See :class:`mud_puppy.jax_lora.QLoRAConfig` for full documentation.
    """

    from .jax_lora import QLoRAConfig as _QLoRAConfig

    return _QLoRAConfig(*args, **kwargs)


def LoRADense(*args, **kwargs):
    """JAX/Flax Dense layer with LoRA adaptation.

    See :class:`mud_puppy.jax_lora.LoRADense` for full documentation.
    """

    from .jax_lora import LoRADense as _LoRADense

    return _LoRADense(*args, **kwargs)


def LoRAAdapter(*args, **kwargs):
    """Standalone LoRA adapter for adding to existing layers.

    See :class:`mud_puppy.jax_lora.LoRAAdapter` for full documentation.
    """

    from .jax_lora import LoRAAdapter as _LoRAAdapter

    return _LoRAAdapter(*args, **kwargs)


def QLoRADense(*args, **kwargs):
    """Combined QLoRA layer: quantized base + LoRA adapters.

    See :class:`mud_puppy.jax_lora.QLoRADense` for full documentation.
    """

    from .jax_lora import QLoRADense as _QLoRADense

    return _QLoRADense(*args, **kwargs)


def merge_lora_params(*args, **kwargs):
    """Merge LoRA weights back into base model weights.

    See :func:`mud_puppy.jax_lora.merge_lora_params` for full documentation.
    """

    from .jax_lora import merge_lora_params as _merge_lora_params

    return _merge_lora_params(*args, **kwargs)


def init_lora_from_pretrained(*args, **kwargs):
    """Initialize LoRA parameters for a pretrained model.

    See :func:`mud_puppy.jax_lora.init_lora_from_pretrained` for full documentation.
    """

    from .jax_lora import init_lora_from_pretrained as _init_lora_from_pretrained

    return _init_lora_from_pretrained(*args, **kwargs)


def CPUOffloadOptimizer(*args, **kwargs):
    """CPU offload optimizer for ZeRO-style memory savings.

    Offloads optimizer states to CPU RAM, enabling training of
    larger models on limited VRAM.
    """
    from .zero_offload import CPUOffloadOptimizer as _CPUOffloadOptimizer
    return _CPUOffloadOptimizer(*args, **kwargs)


def OffloadConfig(*args, **kwargs):
    """Configuration for CPU offloading."""
    from .zero_offload import OffloadConfig as _OffloadConfig
    return _OffloadConfig(*args, **kwargs)


def wrap_optimizer_for_offload(*args, **kwargs):
    """Wrap any optimizer with CPU offloading.

    Args:
        optimizer: Base PyTorch optimizer
        offload_optimizer: Offload optimizer states to CPU
        offload_gradients: Accumulate gradients on CPU
        pin_memory: Use pinned memory for transfers

    Returns:
        Wrapped optimizer with CPU offloading
    """
    from .zero_offload import wrap_optimizer_for_offload as _wrap
    return _wrap(*args, **kwargs)
