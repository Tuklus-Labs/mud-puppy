"""MXFP4 (Microscaling FP4) quantization for ROCm.

Block-wise 4-bit quantization with shared exponents, similar to OCP MX formats.
This provides an alternative to the INT4 quantization in bnb_rocm.py.

Key differences from INT4:
- Block-wise scales instead of row-wise (adapts to local value ranges)
- Better suited for models with varying weight distributions per block
- Compatible with potential future hardware FP4 support

Memory: ~4x compression (packed 4-bit weights + fp16 scales per block)

Usage:
    from mud_puppy.mxfp4_rocm import LinearMX4, quantize_model_mx4

    # Quantize a single layer
    mx4 = LinearMX4.from_linear(original_linear, block_size=32)

    # Quantize an entire model
    model = quantize_model_mx4(model, block_size=32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

log = logging.getLogger(__name__)


def _dequantize_mx4_packed(packed_weight, scales, out_features, in_padded, in_features, num_blocks, block_size, dtype):
    """Unpack INT4 nibbles and apply block-wise scales."""
    low = (packed_weight & 0x0F).to(dtype) - 8
    high = ((packed_weight >> 4) & 0x0F).to(dtype) - 8
    weight_q = torch.stack([low, high], dim=-1).reshape(out_features, in_padded)
    weight_blocks = weight_q.view(out_features, num_blocks, block_size)
    weight_dequant = weight_blocks * scales.unsqueeze(2) / 7.0
    return weight_dequant.reshape(out_features, -1)[:, :in_features].contiguous()


class _LinearMX4Fn(torch.autograd.Function):
    """Custom autograd for MXFP4 linear that saves packed weights instead of dequantized.

    Same principle as _Linear4bitFn in bnb_rocm.py: autograd saves the full
    dequantized bf16 weight for backward (needed for input gradients). For a 3B
    model with 252 layers, that's ~5 GB of saved tensors -- negating INT4 savings.

    This function saves only packed_weight (uint8) + scales (fp16) and
    re-dequantizes during backward. Trades ~2x compute for ~4x memory.
    """

    @staticmethod
    def forward(ctx, input, packed_weight, scales, bias, out_features, in_padded, in_features, num_blocks, block_size, dtype):
        w = _dequantize_mx4_packed(packed_weight, scales, out_features, in_padded, in_features, num_blocks, block_size, dtype)
        output = F.linear(input, w, bias)

        ctx.save_for_backward(packed_weight, scales)
        ctx.out_features = out_features
        ctx.in_padded = in_padded
        ctx.in_features = in_features
        ctx.num_blocks = num_blocks
        ctx.block_size = block_size
        ctx.dtype = dtype
        ctx.has_bias = bias is not None

        return output

    @staticmethod
    def backward(ctx, grad_output):
        packed_weight, scales = ctx.saved_tensors

        w = _dequantize_mx4_packed(
            packed_weight, scales, ctx.out_features, ctx.in_padded,
            ctx.in_features, ctx.num_blocks, ctx.block_size, ctx.dtype,
        )

        grad_input = grad_output @ w

        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.reshape(-1, ctx.out_features).sum(0)

        # No grads for packed_weight, scales, or shape params
        return grad_input, None, None, grad_bias, None, None, None, None, None, None


class LinearMX4(nn.Linear):
    """Linear layer with MXFP4 (block-wise 4-bit) quantized weights.

    Inherits from nn.Linear so PEFT/LoRA recognizes it via isinstance check.
    Skips nn.Linear.__init__ to avoid allocating a full-size weight Parameter.

    Weights are stored as:
    - packed_weight: [out_features, in_padded // 2] uint8 (2 values per byte)
    - scales: [out_features, num_blocks] fp16 (per-block scale factors)

    During forward pass, weights are dequantized on-the-fly from packed storage.
    No full-precision weight copy is kept in memory.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 32,
        device=None,
        dtype: torch.dtype = torch.float16,
    ):
        # Skip nn.Linear.__init__ to avoid allocating full weight tensor.
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.dtype = dtype

        # Pad input dimension to multiple of block_size
        self.in_padded = ((in_features + block_size - 1) // block_size) * block_size
        self.num_blocks = self.in_padded // block_size

        # Packed weights: 2 int4 values per uint8
        self.register_buffer(
            "packed_weight",
            torch.zeros(out_features, self.in_padded // 2, dtype=torch.uint8, device=device),
        )

        # Per-block scales
        self.register_buffer(
            "scales",
            torch.ones(out_features, self.num_blocks, dtype=torch.float16, device=device),
        )

        # Bias as frozen parameter
        if bias:
            self.register_parameter(
                "bias", nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device), requires_grad=False)
            )
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self):
        """Dequantize on access. PEFT may read this for dtype/device info."""
        return self._dequantize_packed()

    @weight.setter
    def weight(self, value):
        # No-op: PEFT or kbit training prep may try to assign.
        pass

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size: int = 32) -> "LinearMX4":
        """Create a quantized layer from an existing Linear layer."""
        mx4 = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        mx4.quantize_from(linear)
        return mx4

    @torch.no_grad()
    def quantize_from(self, linear: nn.Linear) -> None:
        """Quantize weights from an existing Linear layer."""
        weight = linear.weight.to(self.dtype)  # [out, in]

        # Pad input dimension if needed
        weight_padded = weight
        if self.in_features != self.in_padded:
            weight_padded = F.pad(weight, (0, self.in_padded - self.in_features))

        # Reshape to blocks: [out, num_blocks, block_size]
        weight_blocks = weight_padded.view(self.out_features, self.num_blocks, self.block_size)

        # Compute per-block scales (max abs value)
        scales = weight_blocks.abs().amax(dim=2).clamp(min=1e-8)
        self.scales.copy_(scales)

        # Normalize and quantize to [-7, 7] (4-bit signed)
        weight_norm = weight_blocks / scales.unsqueeze(2)
        weight_q = torch.round(weight_norm * 7.0).clamp(-7, 7).to(torch.int8)

        # Pack pairs into bytes: low nibble = even indices, high nibble = odd indices
        weight_flat = weight_q.view(self.out_features, -1)
        packed = (weight_flat[:, 0::2] + 8).to(torch.uint8) | (
            (weight_flat[:, 1::2] + 8).to(torch.uint8) << 4
        )
        self.packed_weight.copy_(packed)

        # Copy bias if present
        if linear.bias is not None and self.bias is not None:
            self.bias.data.copy_(linear.bias.to(self.dtype))

    def _dequantize_packed(self) -> torch.Tensor:
        """Unpack INT4 nibbles and apply block-wise scales."""
        return _dequantize_mx4_packed(
            self.packed_weight, self.scales, self.out_features,
            self.in_padded, self.in_features, self.num_blocks,
            self.block_size, self.dtype,
        )

    def dequantize_weight(self) -> torch.Tensor:
        """Public API: dequantize packed weights to full precision."""
        return self._dequantize_packed()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization."""
        return _LinearMX4Fn.apply(
            x.to(self.dtype),
            self.packed_weight,
            self.scales,
            self.bias,
            self.out_features,
            self.in_padded,
            self.in_features,
            self.num_blocks,
            self.block_size,
            self.dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, block_size={self.block_size}"
        )


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    """Set a module by its dot-separated name."""
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


def quantize_model_mx4(
    model: nn.Module,
    block_size: int = 32,
    skip_modules: Optional[List[str]] = None,
    min_size: int = 1024,
) -> nn.Module:
    """Replace Linear layers with MXFP4 quantized versions.

    Args:
        model: The model to quantize
        block_size: Block size for quantization (larger = fewer scales, less accurate)
        skip_modules: List of module name patterns to skip (e.g., ['lora', 'lm_head'])
        min_size: Minimum number of elements to quantize (skip small layers)

    Returns:
        Model with quantized Linear layers
    """
    skip_modules = skip_modules or [
        "lora",
        "lm_head",
        "embed_tokens",
        "word_embeddings",
        "wte",
        "wpe",
        "score",
        "classifier",
    ]

    quantized_count = 0
    skipped_count = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        # Check skip patterns
        if any(skip in name.lower() for skip in skip_modules):
            log.debug(f"Skipping {name} (matches skip pattern)")
            skipped_count += 1
            continue

        # Check minimum size
        if module.weight.numel() < min_size:
            log.debug(f"Skipping {name} (too small: {module.weight.numel()} < {min_size})")
            skipped_count += 1
            continue

        # Quantize
        mx4_module = LinearMX4.from_linear(module, block_size=block_size)
        _set_module(model, name, mx4_module)
        quantized_count += 1

    log.info(f"Quantized {quantized_count} layers to MXFP4, skipped {skipped_count}")
    return model


def _prepare_model_for_mx4_training(model: nn.Module) -> nn.Module:
    """Prepare a quantized model for training.

    Freezes base weights and ensures biases are trainable.
    """
    for name, module in model.named_modules():
        if isinstance(module, LinearMX4):
            # Packed weights and scales are buffers (already frozen)
            # Bias is a Parameter (trainable by default)
            if module.bias is not None:
                module.bias.requires_grad = True
        elif isinstance(module, nn.Linear):
            # Non-quantized linear layers (lm_head, etc.) - freeze weights
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False

    return model


# Alias for consistency with bnb_rocm
quantize_4bit = quantize_model_mx4
Linear4bit = LinearMX4
