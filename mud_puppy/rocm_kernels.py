"""Naive ROCm-friendly kernels for quantized operations.

These helpers are intentionally simple and CPU/ROCm portable. They are not
intended to compete with vendor libraries, but to provide a clear reference
implementation that works with both HIP and CUDA builds of PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quantize_per_tensor(tensor: torch.Tensor, bits: int = 8):
    """Uniform per-tensor affine quantization.

    Returns (qtensor, scale, zero_point).
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    if tensor.numel() == 0:
        scale = torch.tensor(1.0, device=tensor.device, dtype=torch.float32)
        zero_point = torch.tensor(0.0, device=tensor.device, dtype=torch.float32)
        return torch.zeros_like(tensor, dtype=torch.int8), scale, zero_point
    max_val = tensor.abs().max()
    scale = max_val / qmax if max_val > 0 else torch.tensor(1.0, device=tensor.device)
    scale = scale.to(torch.float32) + 1e-8
    zero_point = torch.tensor(0.0, device=tensor.device, dtype=torch.float32)
    qt = torch.clamp((tensor / scale).round(), qmin, qmax).to(torch.int8)
    return qt, scale, zero_point


def quantize_per_channel(tensor: torch.Tensor, bits: int = 8, dim: int = 0):
    """Per-channel symmetric quantization along a given dimension.

    Returns (qtensor, scale) where scale is broadcastable to ``tensor``.
    Zero-point is implicitly 0 for all channels.
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    max_val = tensor.abs().amax(dim=dim, keepdim=True)
    scale = max_val / qmax
    scale = scale.to(torch.float32) + 1e-8

    qt = torch.clamp((tensor / scale).round(), qmin, qmax).to(torch.int8)
    return qt, scale


def dequantize(qtensor: torch.Tensor, scale: torch.Tensor, zero_point: float = 0.0) -> torch.Tensor:
    """Dequantize an int8 tensor given scale and (optional) zero_point."""
    return (qtensor.float() - zero_point) * scale


def qgemm(a_q: torch.Tensor, a_scale: torch.Tensor, b_q: torch.Tensor, b_scale: torch.Tensor):
    """Quantized matrix multiply via dequantize -> matmul.

    This is a reference implementation; real deployments should use fused
    kernels from vendor libraries where available.
    """
    a = dequantize(a_q, a_scale)
    b = dequantize(b_q, b_scale)
    return a @ b


def fbgemm(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_q: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
):
    """Fused quantized GEMM + bias + activation (naive)."""
    out = qgemm(a_q, a_scale, b_q, b_scale)
    if bias is not None:
        out = out + bias
    if activation == "relu":
        out = F.relu(out)
    return out


def quantized_layernorm(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
):
    """Quantized layernorm using dequantize -> layer_norm -> requantize."""
    x = dequantize(x_q, x_scale)
    y = F.layer_norm(x, weight.shape, weight, bias, eps)
    y_q, y_scale, _ = quantize_per_tensor(y, bits=8)
    return y_q, y_scale


# Backwards-compatible aliases for older code
quantize = quantize_per_tensor
