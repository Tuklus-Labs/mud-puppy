"""Simplified GPTQ implementation for ROCm."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bnb_rocm import _set_module
from .rocm_kernels import quantize, dequantize


class GPTQLinear(nn.Module):
    """Linear layer with int4 weights for inference."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias
        qweight, scale = quantize(linear.weight.detach().to(torch.float32), bits=4)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize(self.qweight, float(self.scale))
        return F.linear(x, weight, self.bias)


def quantize_model_gptq(model: nn.Module) -> nn.Module:
    """Convert all Linear layers to :class:`GPTQLinear`."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            qlinear = GPTQLinear(module)
            _set_module(model, name, qlinear)
    return model


def save_quantized(model: nn.Module, path: str):
    """Save quantized model weights."""
    model.save_pretrained(path)


def load_quantized(model_cls, path: str, trust_remote_code: bool = False) -> nn.Module:
    """Load a quantized model."""
    model = model_cls.from_pretrained(path, trust_remote_code=trust_remote_code)
    return model
