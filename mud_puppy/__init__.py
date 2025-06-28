"""mud-puppy: A ROCm-first LLM fine-tuning framework."""

__all__ = [
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
    """Lazy import and run the training routine."""
    from .trainer import run_training as _run_training

    return _run_training(*args, **kwargs)


def quantize_model_4bit(*args, **kwargs):
    """Lazy import the ROCm 4-bit quantizer."""
    from .bnb_rocm import quantize_model_4bit as _quantize_model_4bit

    return _quantize_model_4bit(*args, **kwargs)


def flash_attention(*args, **kwargs):
    """Lazy import the FlashAttention kernel."""
    from .rocm_attn import flash_attention as _flash_attention

    return _flash_attention(*args, **kwargs)


def FlashMHA(*args, **kwargs):
    """Lazy import the FlashMHA module."""
    from .rocm_attn import FlashMHA as _FlashMHA

    return _FlashMHA(*args, **kwargs)


def apply_qat(*args, **kwargs):
    """Lazy import QAT utilities."""
    from .qat_rocm import apply_qat as _apply_qat

    return _apply_qat(*args, **kwargs)


def convert_qat(*args, **kwargs):
    """Lazy import QAT conversion."""
    from .qat_rocm import convert_qat as _convert_qat

    return _convert_qat(*args, **kwargs)


def quantize_model_gptq(*args, **kwargs):
    """Lazy import the GPTQ quantizer."""
    from .gptq_rocm import quantize_model_gptq as _quantize_model_gptq

    return _quantize_model_gptq(*args, **kwargs)


def save_quantized(*args, **kwargs):
    """Lazy import the GPTQ save function."""
    from .gptq_rocm import save_quantized as _save_quantized

    return _save_quantized(*args, **kwargs)


def load_quantized(*args, **kwargs):
    """Lazy import the GPTQ load function."""
    from .gptq_rocm import load_quantized as _load_quantized

    return _load_quantized(*args, **kwargs)
