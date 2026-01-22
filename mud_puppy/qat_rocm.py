"""Quantization-aware training (QAT) helpers for ROCm.

These utilities provide simple QAT-ready linear layers and conversion
functions. They are designed to be portable across ROCm and CUDA builds and
serve as a reference implementation rather than a high-performance kernel
library.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bnb_rocm import _set_module
from .rocm_kernels import quantize_per_tensor, dequantize


class QATLinear(nn.Module):
    """Quantization-aware training linear layer.

    During training, weights are passed through a fake-quantization op so
    that gradients flow while still modeling int8 quantization effects.
    During evaluation, the dequantized int8 weights are used.
    """

    def __init__(self, linear: nn.Linear, bits: int = 8):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1

        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = linear.bias

        # Per-tensor scale/zero_point parameters for weight quantization
        self.register_buffer("weight_scale", torch.ones(1))
        self.register_buffer("weight_zero_point", torch.zeros(1))

    def _fake_quantize_weight(self) -> torch.Tensor:
        # Use torch's built-in fake quantization for training-time simulation
        scale = self.weight_scale
        zero_point = torch.round(self.weight_zero_point)
        return torch.fake_quantize_per_tensor_affine(
            self.weight, scale, zero_point, self.qmin, self.qmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self._fake_quantize_weight()
        else:
            # At eval time, quantize and dequantize once to approximate int8
            qweight, scale, zp = quantize_per_tensor(self.weight, bits=self.bits)
            weight = dequantize(qweight, scale, float(zp))
        return F.linear(x, weight, self.bias)


def apply_qat(model: nn.Module, bits: int = 8) -> nn.Module:
    """Wrap all ``nn.Linear`` layers with :class:`QATLinear` modules."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            qat = QATLinear(module, bits=bits)
            _set_module(model, name, qat)
    return model


def convert_qat(model: nn.Module, bits: int = 8) -> nn.Module:
    """Convert :class:`QATLinear` layers to int8-backed ``nn.Linear`` layers.

    The converted layers store dequantized float32 weights corresponding to
    the int8 quantization, making them easy to use with standard PyTorch
    runtimes while still benefiting from reduced storage and potential
    deployment-time quantization.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, QATLinear):
            qweight, scale, zp = quantize_per_tensor(module.weight, bits=bits)
            deq_weight = dequantize(qweight, scale, float(zp))
            linear = nn.Linear(module.in_features, module.out_features)
            linear.weight = nn.Parameter(deq_weight)
            linear.bias = module.bias
            _set_module(model, name, linear)
    return model
