import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

log = logging.getLogger(__name__)


def _dequantize_packed(qweight, scale, out_features, in_features, dtype):
    """Unpack INT4 nibbles and apply row-wise scale."""
    low = (qweight & 0x0F).to(dtype) - 8.0
    high = ((qweight >> 4) & 0x0F).to(dtype) - 8.0
    w = torch.stack([low, high], dim=-1).reshape(out_features, -1)
    w = w[:, :in_features]
    return w * scale.to(dtype)


class _Linear4bitFn(torch.autograd.Function):
    """Custom autograd for INT4 linear that saves packed weights instead of dequantized.

    Standard F.linear saves the full dequantized bf16 weight in the autograd graph
    (needed to compute input gradients for LoRA). For 252 layers in a 3B model,
    that's ~5.2 GB of saved tensors -- negating INT4 savings entirely.

    This function saves only qweight (packed uint8) + scale (fp32) and
    re-dequantizes during backward. Trades ~2x compute for ~4x memory.

    When ``MUD_PUPPY_INT4_TRITON=1`` and Triton is available, forward and
    backward call the fused Triton kernels in ``int4_kernels`` which
    dequant+matmul inline without ever materializing the full weight.
    ~3-5x faster on the 7900 XTX for the typical QLoRA linear shapes
    (4096x11008 MLP, 5120x5120 attn). Off by default; see
    ``int4_kernels._is_enabled``.
    """

    @staticmethod
    def forward(ctx, input, qweight, scale, bias, out_features, in_features, dtype):
        # Try the Triton fused path first if it's enabled. Any failure
        # (kernel absent, unsupported shape, runtime error) falls back to
        # the pure-PyTorch dequant+linear path below so training never
        # stops just because a kernel misbehaved.
        output = None
        used_triton = False
        try:
            from . import int4_kernels
            if int4_kernels._is_enabled():
                output = int4_kernels.triton_int4_matmul(
                    input.contiguous(), qweight, scale
                )
                if bias is not None:
                    output = output + bias
                used_triton = True
        except Exception as exc:  # pragma: no cover - fallback path
            log.warning(
                "int4 triton forward failed (%s); falling back to pytorch",
                exc,
            )
            output = None

        if output is None:
            w = _dequantize_packed(qweight, scale, out_features, in_features, dtype)
            output = F.linear(input, w, bias)

        # Save packed tensors (tiny) instead of dequantized weight (huge)
        ctx.save_for_backward(qweight, scale)
        ctx.out_features = out_features
        ctx.in_features = in_features
        ctx.dtype = dtype
        ctx.has_bias = bias is not None
        ctx.used_triton = used_triton

        return output

    @staticmethod
    def backward(ctx, grad_output):
        qweight, scale = ctx.saved_tensors
        grad_input = None

        # Mirror the forward-path selection: if forward used Triton, try
        # the Triton backward too. Fall through to pytorch on any failure.
        if ctx.used_triton:
            try:
                from . import int4_kernels
                grad_input = int4_kernels.triton_int4_grad_input(
                    grad_output.contiguous(), qweight, scale,
                    in_features=ctx.in_features,
                )
            except Exception as exc:  # pragma: no cover - fallback path
                log.warning(
                    "int4 triton backward failed (%s); falling back to pytorch",
                    exc,
                )
                grad_input = None

        if grad_input is None:
            # Re-dequantize during backward (~0.1ms per layer, saves ~20MB per layer)
            w = _dequantize_packed(
                qweight, scale, ctx.out_features, ctx.in_features, ctx.dtype
            )
            # grad_input = grad_output @ W (needed for LoRA gradient flow)
            grad_input = grad_output @ w

        # grad_bias = sum over batch+seq dims
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.reshape(-1, ctx.out_features).sum(0)

        # No grads for qweight, scale, out_features, in_features, dtype
        return grad_input, None, None, grad_bias, None, None, None


class Linear4bit(nn.Linear):
    """4-bit quantized linear layer for ROCm.

    Inherits from nn.Linear so PEFT/LoRA recognizes it via isinstance check.
    Skips nn.Linear.__init__ to avoid allocating a full-size weight Parameter.

    Stores weights as packed INT4 (2 values per uint8 byte) with row-wise
    symmetric quantization. Dequantizes on-the-fly during forward pass.

    Uses a custom autograd Function that saves packed INT4 tensors instead of
    dequantized bf16 weights in the computation graph. This prevents the ~5 GB
    autograd memory overhead that made QLoRA training OOM on 24 GB cards.

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
        return _dequantize_packed(
            self.qweight, self.scale, self.out_features, self.in_features, self.dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _Linear4bitFn.apply(
            input.to(self.dtype),
            self.qweight,
            self.scale,
            self.bias,
            self.out_features,
            self.in_features,
            self.dtype,
        )

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
