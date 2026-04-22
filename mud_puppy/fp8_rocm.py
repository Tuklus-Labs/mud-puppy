"""FP8 training support for ROCm (RDNA4-ready, RDNA3-emulated).

Wires FP8 E4M3 matmul into ``nn.Linear`` layers using the delayed-scaling
recipe popularized by NVIDIA's Transformer Engine, adapted for ROCm's
``torch._scaled_mm``.

Hardware support
~~~~~~~~~~~~~~~~

* **MI300+** (CDNA3) and **RDNA4** (e.g. 9070 XT): hardware FP8 via WMMA
  instructions. ``torch._scaled_mm`` dispatches to native kernels.
* **RDNA3** (7900 XTX, 7900 XT): no hardware FP8; this module falls back
  to cast-and-dequantize + regular matmul. Correct but not faster than
  bf16. Present so you can prototype recipes on RDNA3 and land them on
  RDNA4 without code changes.
* **CPU / no-GPU**: emulated path only.

Recipe
~~~~~~

Per ``FP8Linear`` instance we maintain two EMA-tracked amax values (one
each for input activations and weights). Every forward pass:

1. Cast ``x`` and ``weight`` to ``float8_e4m3fn`` using the previous
   step's amax to derive per-tensor scale. This is "delayed scaling":
   the scale lags one step so there's no global reduction inside the
   forward hot path.
2. Call ``torch._scaled_mm`` (hardware) or the cast-dequant fallback.
3. Update amax observers with the current step's magnitudes, so the
   next forward sees them.

Backward uses E5M2 for gradients by convention (wider dynamic range,
less precision, appropriate for gradients whose magnitude varies
wildly). This module handles forward; PyTorch's autograd takes care of
backward in fp32/bf16 against the master weight copy, which is standard
mixed-precision FP8 training behavior.

Limitations
~~~~~~~~~~~

* No per-channel weight scaling yet. Per-tensor is the Transformer
  Engine default; per-channel would be a future refinement.
* No fp8-specific gradient kernels. Gradients run in the master dtype
  (bf16). That's correct and what TE does; just documenting it.
* Replaces ``nn.Linear`` in place. ``lm_head`` and embeddings are
  skipped by default (both are sensitive to fp8 quantization noise).
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bnb_rocm import _set_module

log = logging.getLogger(__name__)


# E4M3 dynamic range: {-448, ..., 448}. Values beyond this saturate.
_FP8_E4M3_MAX = 448.0
# Amax floor to prevent scale explosion when a tensor is all-zero.
_AMAX_EPS = 1e-8


def _fp8_cast_ste(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Cast `t` to FP8 E4M3 and back to its original dtype, with a
    straight-through gradient.

    Forward: ``clamp(t * scale, -448, 448).to(fp8).to(t.dtype) * (1/scale)``
    Backward: identity wrt ``t``. Scale is treated as a constant.

    The explicit clamp is mandatory: ``float8_e4m3fn`` has only a NaN
    encoding at out-of-range values, not a saturating infinity, so any
    cast that would overflow the representable range becomes NaN and
    poisons downstream gradients.  Without STE, gradients through the
    cast can also go NaN for perfectly in-range values — Transformer
    Engine and PyTorch's native fp8 recipes both use this pattern.
    """
    with torch.no_grad():
        scale_c = scale.to(torch.float32)
        t_scaled = (t.to(torch.float32) * scale_c).clamp(
            -_FP8_E4M3_MAX, _FP8_E4M3_MAX
        )
        t_fp8 = t_scaled.to(torch.float8_e4m3fn)
        t_deq = t_fp8.to(t.dtype) / scale_c.to(t.dtype)
    # STE: forward value = t_deq, gradient flows through t unchanged.
    return t + (t_deq - t).detach()


def is_fp8_hardware_available() -> bool:
    """Return True if ``torch._scaled_mm`` will actually run on this device.

    Probes by trying a small scaled mm. Not cached: caching at module level
    would permanently return False if called before torch.cuda is initialized
    (e.g. in test environments), and the probe cost (~2ms) is negligible
    compared to any real use.
    """
    if not (hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn")):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        a = torch.zeros(16, 32, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        b = torch.zeros(32, 16, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        sa = torch.tensor(1.0, device="cuda")
        sb = torch.tensor(1.0, device="cuda")
        torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        return True
    except Exception as exc:
        log.debug("FP8 hardware probe failed: %s", exc)
        return False


class FP8Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` that matmuls in FP8 E4M3.

    The master weight stays in its original dtype (typically bf16). On
    every forward we cast an FP8 view of the weight and the input,
    matmul, then unscale. Delayed scaling: the cast uses the amax from
    the *previous* step, so there's no global reduction in the hot path.
    """

    def __init__(
        self,
        linear: nn.Linear,
        amax_momentum: float = 0.95,
        fp8_format: str = "e4m3",
    ) -> None:
        super().__init__()
        if fp8_format != "e4m3":
            raise ValueError("only E4M3 forward currently supported")
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Master weight in original dtype; optimizer updates this copy.
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = linear.bias

        # Delayed-scaling observers: store amax as a buffer so it
        # survives checkpoint round-trips. Initialized to a conservative
        # 1.0 so the first step has a reasonable starting scale.
        self.register_buffer("input_amax", torch.tensor(1.0))
        self.register_buffer("weight_amax", torch.tensor(1.0))
        self.amax_momentum = float(amax_momentum)
        self.fp8_format = fp8_format

        self._hw = is_fp8_hardware_available()

    # ------------------------------------------------------------------
    # Scaling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_scale(amax: torch.Tensor) -> torch.Tensor:
        """Scale factor to fit ``amax`` inside the E4M3 range."""
        return _FP8_E4M3_MAX / torch.clamp(amax, min=_AMAX_EPS)

    @torch.no_grad()
    def _update_amax(self, buf: torch.Tensor, new_amax: torch.Tensor) -> None:
        """EMA update of an amax buffer with the observed max.

        C6 fix: clamp the stored amax at _AMAX_EPS so it can never decay to
        exactly zero. Without the floor, if new_amax stays at 0 (e.g. all-zero
        weights during warmup), the exponential decay drives buf -> 0 after
        ~150 steps, which makes _compute_scale blow up to infinity and poisons
        all subsequent forward passes with NaN/Inf outputs.

        A1 fix: sanitize NaN/Inf in new_amax before the EMA. torch.clamp and
        torch.maximum both propagate NaN (clamp(NaN, min=x) == NaN), so a
        single NaN weight would permanently corrupt the amax buffer.
        """
        new_amax = torch.nan_to_num(
            new_amax.detach(),
            nan=_AMAX_EPS,
            posinf=_FP8_E4M3_MAX,
            neginf=_AMAX_EPS,
        )
        decayed = buf * self.amax_momentum
        buf.copy_(torch.maximum(decayed, new_amax).clamp(min=_AMAX_EPS))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep fp32 scales; scaled_mm wants fp32 scales and the cast
        # itself is numerically insensitive to scale dtype.
        x_scale = self._compute_scale(self.input_amax.to(x.device)).to(torch.float32)
        w_scale = self._compute_scale(self.weight_amax.to(x.device)).to(torch.float32)

        # Collapse leading batch dims so the matmul is 2D (scaled_mm
        # only accepts 2D operands).
        orig_shape = x.shape
        x2 = x.reshape(-1, self.in_features)

        out_dtype = x.dtype if x.dtype != torch.float8_e4m3fn else torch.bfloat16

        if self._hw:
            # Hardware path: use torch._scaled_mm. Cast under no_grad
            # because scaled_mm itself takes the backward via master
            # weights; forward-only cast is all we need. Clamp to the
            # E4M3 range before cast (see _fp8_cast_ste note on NaN).
            with torch.no_grad():
                x_fp8 = (
                    (x2.to(torch.float32) * x_scale)
                    .clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
                    .to(torch.float8_e4m3fn)
                )
                w_fp8 = (
                    (self.weight.to(torch.float32) * w_scale)
                    .clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
                    .to(torch.float8_e4m3fn)
                )
                inv_xs = (1.0 / x_scale).to(torch.float32)
                inv_ws = (1.0 / w_scale).to(torch.float32)
                out_no_grad = torch._scaled_mm(
                    x_fp8,
                    w_fp8.t(),
                    scale_a=inv_xs,
                    scale_b=inv_ws,
                    out_dtype=out_dtype,
                )
            # Emulated forward that matches out_no_grad numerically but
            # has an STE gradient path so autograd can flow. We compute
            # it from the STE-casted tensors and replace the forward
            # value with the hardware result for accumulator precision.
            x_q = _fp8_cast_ste(x2, x_scale)
            w_q = _fp8_cast_ste(self.weight, w_scale)
            ste_out = F.linear(x_q, w_q, None)
            out2 = ste_out + (out_no_grad - ste_out).detach()
        else:
            # RDNA3 / CPU: pure STE path. Correct and differentiable,
            # but no hardware speedup.
            x_q = _fp8_cast_ste(x2, x_scale)
            w_q = _fp8_cast_ste(self.weight, w_scale)
            out2 = F.linear(x_q, w_q, None)

        # Refresh amax observers for next step. Use the magnitudes of
        # the real tensors (not the quantized ones).
        self._update_amax(self.input_amax, x2.detach().abs().amax())
        self._update_amax(self.weight_amax, self.weight.detach().abs().amax())

        out = out2.reshape(*orig_shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"format={self.fp8_format}, hw={self._hw}"
        )


def apply_fp8(
    model: nn.Module,
    amax_momentum: float = 0.95,
    skip_names: Optional[Iterable[str]] = None,
) -> nn.Module:
    """Replace every ``nn.Linear`` in ``model`` with :class:`FP8Linear`.

    ``skip_names`` is a tuple of substrings; any module whose qualified
    name matches any entry is left alone. Defaults to
    ``("lm_head", "embed_tokens", "score")``.
    """
    skip = tuple(
        skip_names if skip_names is not None else ("lm_head", "embed_tokens", "score")
    )
    n = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, FP8Linear):
            continue
        if any(s in name for s in skip):
            continue
        fp8 = FP8Linear(module, amax_momentum=amax_momentum)
        _set_module(model, name, fp8)
        n += 1
    log.info(
        "FP8: replaced %d nn.Linear layers (hw=%s)", n, is_fp8_hardware_available()
    )
    return model


def fp8_layer_count(model: nn.Module) -> int:
    """Number of :class:`FP8Linear` modules currently installed."""
    return sum(1 for m in model.modules() if isinstance(m, FP8Linear))
