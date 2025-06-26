import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear4bit(nn.Module):
    """Simple 4-bit quantized linear layer for ROCm."""

    def __init__(self, linear: nn.Linear, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias
        self.dtype = dtype

        weight = linear.weight.detach().to(torch.float32)
        max_val = weight.abs().amax(dim=1, keepdim=True)
        scale = max_val / 7.5 + 1e-8
        qweight = torch.clamp((weight / scale).round() + 8, 0, 15).to(torch.uint8)

        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = (self.qweight.to(torch.float32) - 8) * self.scale
        return F.linear(input.to(weight.dtype), weight.to(self.dtype), self.bias)


def _set_module(model: nn.Module, name: str, module: nn.Module):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


def quantize_model_4bit(model: nn.Module, dtype: torch.dtype = torch.float16) -> nn.Module:
    """Replace Linear layers with 4-bit quantized versions."""

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            qmodule = Linear4bit(module, dtype=dtype)
            _set_module(model, name, qmodule)
    return model
