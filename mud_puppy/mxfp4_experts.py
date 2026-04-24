"""Packed MXFP4 replacement for the HuggingFace ``GptOssExperts`` module.

This is the Phase-6 piece that makes gpt-oss-20b fit on a 24 GB GPU.

The native ``GptOssExperts`` stores expert weights as raw 3-D parameters:
    gate_up_proj       [E, H, 2I]  bf16    -- 32 x 2880 x 5760 ~ 1.0 GB per layer
    down_proj          [E, I, H]   bf16    -- 32 x 2880 x 2880 ~ 0.5 GB per layer
    gate_up_proj_bias  [E, 2I]     bf16    -- negligible
    down_proj_bias     [E, H]      bf16    -- negligible

At 24 layers that is ~36 GB just for the experts. Our ``MXFP4Linear``
only wraps ``nn.Linear`` modules, so it cannot touch these Parameters.
This module does.

Storage layout (matches ``mxfp4.pack_mxfp4``'s 2-D+ convention):
    gate_up_qweight    uint8 [E, 2I, H//2]
    gate_up_scales     uint8 [E, 2I, H//32]
    down_qweight       uint8 [E, H, I//2]
    down_scales        uint8 [E, H, I//32]

The reduction axis has to be the innermost axis for ``triton_mxfp4_matmul``
(which computes ``x @ dequant(W).T``). That means we transpose the native
HF layout before packing:
    gate_up_proj [E, H, 2I] -> transpose -> [E, 2I, H] -> pack along H
    down_proj    [E, I, H]  -> transpose -> [E, H, I]  -> pack along I

Biases stay in bf16 (~0.1 MB per layer; not worth quantizing).

Memory per MoE layer: ~(H * 2I + H * I) * E * 0.5 bytes + scales
                    = (2880 * 5760 + 2880 * 2880) * 32 * 0.5 + scales
                    ~ 400 MB of weights + 12 MB of scales
                    vs 1.5 GB bf16
                    = 3.76x compression, 24 layers -> ~10 GB total.

This is a QLoRA-style setup: experts are FROZEN (stored as buffers).
Training the expert weights themselves is a separate design (phase 7;
see MXFP4QATLinear for the pattern). For now, LoRA on the attention
projections does all the learning.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# MXFP4Experts
# --------------------------------------------------------------------------


class MXFP4Experts(nn.Module):
    """Drop-in replacement for transformers' ``GptOssExperts``.

    Construct from an existing ``GptOssExperts`` via :meth:`from_gpt_oss`.
    Forward signature matches the HF original exactly so
    ``transformers.models.gpt_oss`` doesn't know the difference.
    """

    # The HF implementation's SwiGLU constants.
    _ALPHA = 1.702
    _LIMIT = 7.0

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if hidden_size % 32 != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by 32 "
                f"for MXFP4 block alignment"
            )
        if intermediate_size % 32 != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by 32"
            )
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype

        # Register empty buffers; ``from_gpt_oss`` fills them.
        # Shapes computed so that .to(device) works before population.
        H = hidden_size
        I = intermediate_size
        E = num_experts
        self.register_buffer(
            "gate_up_qweight",
            torch.zeros(E, 2 * I, H // 2, dtype=torch.uint8),
        )
        self.register_buffer(
            "gate_up_scales",
            torch.zeros(E, 2 * I, H // 32, dtype=torch.uint8),
        )
        self.register_buffer(
            "down_qweight",
            torch.zeros(E, H, I // 2, dtype=torch.uint8),
        )
        self.register_buffer(
            "down_scales",
            torch.zeros(E, H, I // 32, dtype=torch.uint8),
        )
        # Biases -- frozen parameters so they appear in state_dict for checkpoint
        # round-trip, but requires_grad=False (QLoRA pattern).
        self.register_parameter(
            "gate_up_proj_bias",
            nn.Parameter(torch.zeros(E, 2 * I, dtype=dtype), requires_grad=False),
        )
        self.register_parameter(
            "down_proj_bias",
            nn.Parameter(torch.zeros(E, H, dtype=dtype), requires_grad=False),
        )
        # Surface attributes the HF code reads (alpha, limit) so a bare
        # instance behaves like GptOssExperts.
        self.alpha = self._ALPHA
        self.limit = self._LIMIT

    # ----------------------------------------------------------------------
    # Construction from a live GptOssExperts
    # ----------------------------------------------------------------------

    @classmethod
    def from_gpt_oss(cls, source: nn.Module, dtype: torch.dtype = torch.bfloat16) -> "MXFP4Experts":
        """Quantize and pack an existing GptOssExperts module.

        Source is consumed destructively -- caller should ``del`` it and
        ``gc.collect()`` afterward to free the bf16 expert tensors.
        """
        from .mxfp4 import pack_mxfp4

        H = source.hidden_size
        I = source.intermediate_size
        E = source.num_experts

        inst = cls(
            num_experts=E,
            hidden_size=H,
            intermediate_size=I,
            dtype=dtype,
        )

        with torch.no_grad():
            # gate_up_proj: native [E, H, 2I] -> transpose last two -> [E, 2I, H]
            # -> pack along last axis (H). fp32 for numerical stability of pack.
            gu = source.gate_up_proj.detach()  # [E, H, 2I]
            gu_t = gu.transpose(-1, -2).contiguous().to(torch.float32)  # [E, 2I, H]
            gu_q, gu_s = pack_mxfp4(gu_t, block_size=32)
            inst.gate_up_qweight.copy_(gu_q)
            inst.gate_up_scales.copy_(gu_s)
            del gu, gu_t, gu_q, gu_s

            # down_proj: native [E, I, H] -> transpose -> [E, H, I] -> pack along I
            d = source.down_proj.detach()  # [E, I, H]
            d_t = d.transpose(-1, -2).contiguous().to(torch.float32)  # [E, H, I]
            d_q, d_s = pack_mxfp4(d_t, block_size=32)
            inst.down_qweight.copy_(d_q)
            inst.down_scales.copy_(d_s)
            del d, d_t, d_q, d_s

            # Biases: direct copy at the target dtype.
            inst.gate_up_proj_bias.data.copy_(
                source.gate_up_proj_bias.detach().to(dtype)
            )
            inst.down_proj_bias.data.copy_(
                source.down_proj_bias.detach().to(dtype)
            )

        return inst

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _mxfp4_matmul(
        x: torch.Tensor,
        qw: torch.Tensor,
        scales: torch.Tensor,
        in_features: int,
    ) -> torch.Tensor:
        """Dispatch matmul to the Triton kernel if enabled, else pure PyTorch.

        qw shape is [N, K//2], scales [N, K//32], x is [..., K].
        Returns [..., N] with the same dtype as x.
        """
        try:
            from .mxfp4_kernels import _is_enabled, triton_mxfp4_matmul
            if _is_enabled():
                return triton_mxfp4_matmul(
                    x.contiguous(), qw, scales, in_features=in_features,
                )
        except Exception:
            pass

        # Fallback: unpack + F.linear
        from .mxfp4 import unpack_mxfp4
        N = qw.shape[0]
        padded_k = qw.shape[-1] * 2
        w = unpack_mxfp4(
            qw, scales, (N, padded_k), block_size=32, dtype=x.dtype,
        )
        w = w[:, :in_features].contiguous()
        return F.linear(x, w)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        """Bit-identical port of GptOssExperts._apply_gate."""
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        return gated_output

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bit-identical logic to GptOssExperts.forward, routed through MXFP4 matmul.

        hidden_states:   [T, H]          flattened (batch*seq) activations
        router_indices:  [T, top_k]      which experts each token hits
        routing_weights: [T, top_k]      per-expert mixing weights
        returns:         [T, H]
        """
        next_states = torch.zeros_like(
            hidden_states, dtype=hidden_states.dtype, device=hidden_states.device,
        )
        with torch.no_grad():
            # One-hot expert mask [T, top_k, E+1]; padding class is E (masked out).
            expert_mask = F.one_hot(router_indices, num_classes=self.num_experts)
            # [E, top_k, T] so the outer loop picks active experts.
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:  # padding class, skip
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]  # [T_e, H]

            # gate_up = current_state @ gate_up_proj[e] + bias
            # Packed weight is the TRANSPOSE of native gate_up_proj[e] so
            # triton_mxfp4_matmul (x @ W.T) gives x @ gate_up_proj[e].
            gate_up = self._mxfp4_matmul(
                current_state,
                self.gate_up_qweight[expert_idx],
                self.gate_up_scales[expert_idx],
                in_features=self.hidden_size,
            )
            gate_up = gate_up + self.gate_up_proj_bias[expert_idx]

            gated_output = self._apply_gate(gate_up)  # [T_e, I]

            # out = gated @ down_proj[e] + bias
            out = self._mxfp4_matmul(
                gated_output,
                self.down_qweight[expert_idx],
                self.down_scales[expert_idx],
                in_features=self.intermediate_size,
            )
            out = out + self.down_proj_bias[expert_idx]

            weighted = out * routing_weights[token_idx, top_k_pos, None]
            next_states.index_add_(
                0, token_idx, weighted.to(hidden_states.dtype)
            )

        return next_states

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, quant=mxfp4_packed"
        )


# --------------------------------------------------------------------------
# Model-level swap helper
# --------------------------------------------------------------------------


def swap_gpt_oss_experts_to_mxfp4(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Replace every ``GptOssExperts`` in ``model`` with ``MXFP4Experts``.

    Returns the number of modules swapped. Safe to call on a model that
    has already been (partially) swapped -- existing MXFP4Experts are
    left alone.

    Usage::

        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        swap_gpt_oss_experts_to_mxfp4(model)
        model.to("cuda")
    """
    # Import lazily so this file doesn't hard-require transformers.
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers gpt_oss model is required to swap experts"
        ) from exc

    swapped = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, MXFP4Experts):
            continue
        if isinstance(module, GptOssExperts):
            new_mod = MXFP4Experts.from_gpt_oss(module, dtype=dtype)
            # Walk to parent and setattr
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_mod)
            swapped += 1
    return swapped
