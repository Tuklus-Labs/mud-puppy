"""Naive ROCm kernels for quantized operations."""

import torch
import torch.nn.functional as F


def quantize(tensor: torch.Tensor, bits: int = 8):
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    scale = tensor.abs().max() / qmax if tensor.numel() > 0 else 1.0
    scale = scale + 1e-8
    qt = torch.clamp((tensor / scale).round(), qmin, qmax).to(torch.int8)
    zero_point = 0
    return qt, scale, zero_point


def dequantize(qtensor: torch.Tensor, scale: float, zero_point: float = 0.0) -> torch.Tensor:
    return (qtensor.float() - zero_point) * scale


def qgemm(a_q: torch.Tensor, a_scale: float, b_q: torch.Tensor, b_scale: float):
    a = dequantize(a_q, a_scale)
    b = dequantize(b_q, b_scale)
    return a @ b


def fbgemm(
    a_q: torch.Tensor,
    a_scale: float,
    b_q: torch.Tensor,
    b_scale: float,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
):
    out = qgemm(a_q, a_scale, b_q, b_scale)
    if bias is not None:
        out = out + bias
    if activation == "relu":
        out = F.relu(out)
    return out


def quantized_layernorm(
    x_q: torch.Tensor,
    x_scale: float,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
):
    x = dequantize(x_q, x_scale)
    y = F.layer_norm(x, weight.shape, weight, bias, eps)
    y_q, y_scale, y_zero = quantize(y, bits=8)
    return y_q, y_scale
