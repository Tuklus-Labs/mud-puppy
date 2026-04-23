"""Quantization-aware training modules for MXFP4 (real OCP format).

Distinct from ``qat_rocm.py`` (symmetric INT8 QAT with per-channel scale)
and from ``mxfp4_rocm.py`` (inference-time block-wise INT4 with fp16 scale).
This module targets the same grid hardware like B200 / MI350 / MI400
executes natively, so training and deployment agree bit-for-bit.

Forward path:
    w_mxfp4 = quantize_mxfp4(w_master, block_size=32)   # on-grid fp tensor
    y = F.linear(x, w_mxfp4, bias)

Backward path (STE):
    dL/dw_master = dL/dw_mxfp4         # gradient passes through quant as identity

Master weights stay in bf16 (or fp32) so optimizer updates are smooth;
the MXFP4 grid is only applied in the forward pass. Phase 3 adds a
stochastic-rounding optimizer that snaps masters to the grid on each
update so the final checkpoint is deploy-ready without any conversion.

Scope notes:
    * Activations are left in bf16; only weights are MXFP4-quantized.
      Full activation MXFP4 (A4W4) is out of scope for phase 2 --
      accuracy is more fragile and needs per-token stochastic rounding.
    * Embeddings, layernorms, and lm_head are skipped by default. They
      can be included via ``skip_names=()`` but expect accuracy loss.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bnb_rocm import _set_module
from .mxfp4 import pack_mxfp4, quantize_mxfp4, unpack_mxfp4


class _MXFP4FakeQuant(torch.autograd.Function):
    """Straight-through fake-quantize for MXFP4.

    Forward runs ``quantize_mxfp4``. Backward returns the upstream
    gradient untouched (STE), which is the standard QAT trick -- the
    optimizer sees gradients as if the quantization were identity, so
    the master weight can still move smoothly even though the forward
    value sits on the MXFP4 grid.
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor, block_size: int) -> torch.Tensor:
        return quantize_mxfp4(w, block_size=block_size, stochastic=False)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # STE: gradient flows as identity. No grad for block_size (int).
        return grad_out, None


def fake_quantize_mxfp4(w: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Public entry for the STE fake-quant op. Autograd-safe."""
    return _MXFP4FakeQuant.apply(w, block_size)


class MXFP4QATLinear(nn.Linear):
    """Linear layer whose weight is fake-quantized to MXFP4 on every forward.

    Inherits from ``nn.Linear`` so PEFT/LoRA tooling recognizes it via
    isinstance. The master weight is a regular bf16 (or fp32) Parameter;
    only the forward-time value sits on the MXFP4 grid.

    At deploy time, call ``self.pack()`` to produce the compressed
    (uint8 nibbles, uint8 scales) representation. Because the forward
    uses round-to-nearest, packing the master weight at any point during
    training yields a self-consistent checkpoint with the same forward
    output as the QAT model.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 32,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.block_size = int(block_size)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size: int = 32) -> "MXFP4QATLinear":
        """Wrap an existing nn.Linear, adopting its weights and bias."""
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            mod.weight.copy_(linear.weight)
            if linear.bias is not None:
                mod.bias.copy_(linear.bias)
        return mod

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fq = fake_quantize_mxfp4(self.weight, self.block_size)
        return F.linear(x, w_fq, self.bias)

    # ------------------------------------------------------------------
    # Deploy: pack master weights into MXFP4 storage
    # ------------------------------------------------------------------

    def pack(self):
        """Quantize and pack the current master weight.

        Returns ``(nibbles, scales, shape)`` suitable for storing alongside
        bias. Use ``unpack_mxfp4`` (or a native MXFP4 Linear at phase 4)
        to reconstruct the dequantized weight at inference time.
        """
        with torch.no_grad():
            nibbles, scales = pack_mxfp4(self.weight.detach(), self.block_size)
        return nibbles, scales, tuple(self.weight.shape)

    def unpacked_weight(self) -> torch.Tensor:
        """Quantize and dequantize the current weight *without autograd*.

        Useful for validating that training checkpoints match the
        deployed MXFP4 tensor. At convergence (with stochastic-rounding
        optimizer), this should equal ``self.weight`` up to fp rounding.
        """
        nibbles, scales, shape = self.pack()
        return unpack_mxfp4(nibbles, scales, shape, self.block_size, self.weight.dtype)

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return f"{base}, mxfp4_block={self.block_size}"


# ---------------------------------------------------------------------------
# Model-wide helpers
# ---------------------------------------------------------------------------


_DEFAULT_SKIP = ("lm_head", "embed_tokens", "score", "token_embd")


def apply_mxfp4_qat(
    model: nn.Module,
    block_size: int = 32,
    skip_names: Optional[Iterable[str]] = None,
) -> nn.Module:
    """Wrap every ``nn.Linear`` with :class:`MXFP4QATLinear`.

    Modules whose qualified name contains any of ``skip_names`` are left
    alone. Default skip list protects embedding/output tensors where
    4-bit quantization typically costs more perplexity than it saves memory.
    """
    skip = tuple(skip_names if skip_names is not None else _DEFAULT_SKIP)
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, MXFP4QATLinear):
            continue  # already wrapped
        if any(s in name for s in skip):
            continue
        new_module = MXFP4QATLinear.from_linear(module, block_size=block_size)
        _set_module(model, name, new_module)
    return model


def convert_mxfp4_qat_to_linear(model: nn.Module) -> nn.Module:
    """Convert :class:`MXFP4QATLinear` layers to plain ``nn.Linear`` with
    their on-grid weights baked in.

    After conversion, the model is a standard nn.Module with weights that
    already sit on the MXFP4 grid. Saving/loading via state_dict works;
    an MXFP4-native inference runtime can re-quantize losslessly because
    the weights are already grid-aligned.

    For true packed storage (4x memory savings), use ``pack_mxfp4_qat``
    which returns a dict of packed tensors.
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, MXFP4QATLinear):
            continue
        new_linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        with torch.no_grad():
            # Use the on-grid quantized weight (no autograd -- we're deploying).
            new_linear.weight.copy_(module.unpacked_weight())
            if module.bias is not None:
                new_linear.bias.copy_(module.bias)
        _set_module(model, name, new_linear)
    return model


def pack_mxfp4_qat(model: nn.Module) -> dict:
    """Return a dict of ``{name: (nibbles, scales, shape)}`` for every
    :class:`MXFP4QATLinear` in ``model``.

    This is what phase 4's native-MXFP4 inference path will consume.
    Phase 3 will add checkpoint serialization on top of this.
    """
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, MXFP4QATLinear):
            out[name] = module.pack()
    return out
