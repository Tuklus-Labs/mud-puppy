import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear4bit(nn.Linear):
    """4-bit quantized linear layer for ROCm.

    Inherits from nn.Linear so PEFT/LoRA recognizes it as a valid target.
    The actual computation uses dequantized weights from 4-bit storage.
    """

    def __init__(self, linear: nn.Linear, dtype: torch.dtype = torch.float16):
        # Initialize as nn.Linear with same dimensions
        super().__init__(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=dtype,
        )
        self.dtype = dtype

        # Quantize weights to 4-bit
        weight = linear.weight.detach().to(torch.float32)
        max_val = weight.abs().amax(dim=1, keepdim=True)
        scale = max_val / 7.5 + 1e-8
        qweight = torch.clamp((weight / scale).round() + 8, 0, 15).to(torch.uint8)

        # Store quantized weights as buffers (not parameters)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)

        # Pre-compute dequantized weights for forward pass
        dequant = (self.qweight.to(torch.float32) - 8) * self.scale

        # Replace the weight parameter with dequantized version
        # This allows LoRA to work since it modifies self.weight
        with torch.no_grad():
            self.weight.copy_(dequant.to(dtype))
            self.weight.requires_grad = False  # Freeze base weights

        # Copy bias if present
        if linear.bias is not None:
            with torch.no_grad():
                self.bias.copy_(linear.bias.to(dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Use the standard nn.Linear forward which uses self.weight
        # LoRA modifies self.weight, so this will include LoRA contributions
        return F.linear(input.to(self.dtype), self.weight, self.bias)

    def dequantize(self) -> None:
        """Refresh dequantized weights from quantized storage."""
        with torch.no_grad():
            dequant = (self.qweight.to(torch.float32) - 8) * self.scale
            self.weight.copy_(dequant.to(self.dtype))


def _set_module(model: nn.Module, name: str, module: nn.Module):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


def quantize_model_4bit(
    model: nn.Module,
    dtype: torch.dtype = torch.float16,
    skip_modules: list = None
) -> nn.Module:
    """Replace Linear layers with 4-bit quantized versions.

    Args:
        model: The model to quantize
        dtype: Data type for dequantized weights
        skip_modules: List of module name patterns to skip (e.g., ['lora', 'lm_head'])
    """
    skip_modules = skip_modules or []

    for name, module in list(model.named_modules()):
        # Skip if name matches any skip pattern
        if any(skip in name.lower() for skip in skip_modules):
            continue

        if isinstance(module, nn.Linear) and not isinstance(module, Linear4bit):
            qmodule = Linear4bit(module, dtype=dtype)
            _set_module(model, name, qmodule)

    return model
