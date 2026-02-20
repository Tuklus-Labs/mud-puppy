"""Simplified GPTQ-style implementation for ROCm.

This module provides a reference int4 quantization path for linear layers.
It is *not* a full GPTQ implementation, but it follows a similar spirit:
per-channel quantization of weights for efficient inference. It is fully
portable between ROCm and CUDA builds of PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bnb_rocm import _set_module
from .rocm_kernels import quantize_per_channel, dequantize


class GPTQLinear(nn.Module):
    """Linear layer with per-channel int4 weights for inference."""

    def __init__(self, linear: nn.Linear, bits: int = 4):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.bias = linear.bias

        # Quantize weight per output channel (dim=0)
        qweight, scale = quantize_per_channel(
            linear.weight.detach().to(torch.float32), bits=bits, dim=0
        )
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on-the-fly; for real speedups this would be fused
        weight = dequantize(self.qweight, self.scale)
        return F.linear(x, weight, self.bias)


def quantize_model_gptq(
    model: nn.Module,
    bits: int = 4,
    skip_modules=None,
    min_size: int = 1024,
) -> nn.Module:
    """Convert Linear layers to :class:`GPTQLinear`.

    This is a post-training quantization pass; call it on a trained model
    before saving for int4 inference. Skips lm_head, embeddings, and small
    layers by default.
    """
    skip_modules = skip_modules or [
        "lm_head",
        "embed_tokens",
        "word_embeddings",
        "wte",
        "wpe",
        "score",
        "classifier",
    ]

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(skip in name.lower() for skip in skip_modules):
            continue
        if module.weight.numel() < min_size:
            continue
        qlinear = GPTQLinear(module, bits=bits)
        _set_module(model, name, qlinear)
    return model


def save_quantized(model: nn.Module, path: str):
    """Save quantized model weights.

    The quantized modules are standard ``nn.Module``s, so this is just a
    wrapper around ``save_pretrained``.
    """
    model.save_pretrained(path)


def load_quantized(model_cls, path: str, trust_remote_code: bool = False) -> nn.Module:
    """Load a quantized model.

    This assumes the model at ``path`` was previously saved with
    :func:`save_quantized` after running :func:`quantize_model_gptq`.
    """
    model = model_cls.from_pretrained(path, trust_remote_code=trust_remote_code)
    return model
