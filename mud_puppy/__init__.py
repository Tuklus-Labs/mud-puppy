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
