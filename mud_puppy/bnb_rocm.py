import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class Linear4bit(nn.Linear):
    """4-bit quantized linear layer for ROCm.

    Inherits from nn.Linear so PEFT/LoRA recognizes it via isinstance check.
    Skips nn.Linear.__init__ to avoid allocating a full-size weight Parameter.

    Stores weights as packed INT4 (2 values per uint8 byte) with row-wise
    symmetric quantization. Dequantizes on-the-fly during forward pass.

    VRAM per layer: out * ceil(in/2) bytes (packed) + out * 4 bytes (scale)
    vs nn.Linear:   out * in * 2 bytes (bf16)
    Savings: ~4x for bf16 base, true INT4 density.
    """

    def __init__(self, linear: nn.Linear, dtype: torch.dtype = torch.float16):
        # Use nn.Module.__init__ to skip nn.Linear's weight allocation.
        # We still inherit from nn.Linear for isinstance checks (PEFT compatibility).
        nn.Module.__init__(self)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.dtype = dtype

        # Quantize weights to symmetric 4-bit: range [-7, 7]
        w = linear.weight.detach().float()
        max_val = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = max_val / 7.0
        qw = torch.clamp((w / scale).round(), -7, 7).to(torch.int8) + 8  # shift to [1, 15]

        # Pack 2 INT4 values per uint8: low nibble = even col, high nibble = odd col
        if self.in_features % 2:
            qw = F.pad(qw, (0, 1), value=8)  # pad with zero-point
        packed = (qw[:, 0::2] | (qw[:, 1::2] << 4)).to(torch.uint8)

        self.register_buffer("qweight", packed)
        self.register_buffer("scale", scale)  # keep fp32 for precision

        # Bias as frozen parameter (negligible VRAM)
        if linear.bias is not None:
            self.register_parameter(
                "bias", nn.Parameter(linear.bias.detach().to(dtype), requires_grad=False)
            )
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self):
        """Dequantize on access. PEFT may read this for dtype/device info."""
        return self._dequantize()

    @weight.setter
    def weight(self, value):
        # No-op: PEFT or _prepare_model_for_kbit_training may try to assign.
        pass

    def _dequantize(self):
        """Unpack INT4 nibbles and apply row-wise scale."""
        low = (self.qweight & 0x0F).to(self.dtype) - 8.0
        high = ((self.qweight >> 4) & 0x0F).to(self.dtype) - 8.0
        w = torch.stack([low, high], dim=-1).reshape(self.out_features, -1)
        w = w[:, :self.in_features]
        return w * self.scale.to(self.dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input.to(self.dtype), self._dequantize(), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quant=int4_packed"
        )


def _set_module(model: nn.Module, name: str, module: nn.Module):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


def quantize_model_4bit(
    model: nn.Module,
    dtype: torch.dtype = torch.float16,
    skip_modules: Optional[List[str]] = None,
    min_size: int = 1024,
) -> nn.Module:
    """Replace Linear layers with 4-bit quantized versions.

    Args:
        model: The model to quantize
        dtype: Data type for dequantized weights
        skip_modules: List of module name patterns to skip (e.g., ['lora', 'lm_head'])
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
        # Skip if name matches any skip pattern
        if any(skip in name.lower() for skip in skip_modules):
            continue

        if isinstance(module, nn.Linear) and not isinstance(module, Linear4bit):
            if module.weight.numel() < min_size:
                continue
            qmodule = Linear4bit(module, dtype=dtype)
            _set_module(model, name, qmodule)

    return model
