"""Quantization-aware training (QAT) helpers for ROCm.

QAT replaces every ``nn.Linear`` with a ``QATLinear`` that runs a fake
quantization op during the forward pass (straight-through estimator for
gradients) so the model learns weight distributions that quantize cleanly.

Implementation notes
~~~~~~~~~~~~~~~~~~~~

* **Per-channel scales.** Scale is stored per output channel rather than
  per-tensor. This is the difference between "QAT that produces a usable
  int8 model" and "QAT that tolerates int8 vaguely"; per-tensor scales
  collapse information in the tails of wide weight matrices.
* **Calibrated on construction.** ``weight_scale`` is initialized from
  ``max(|w|) / qmax`` per channel at module swap time. The previous
  version of this file left the scale at ``1.0`` so fake-quant was a
  no-op clamp; see 2026-04-22 audit.
* **Optional EMA scale update.** Call ``qat_update_scales(model, momentum)``
  at a fixed cadence during training (a trainer callback does this for
  you) to re-calibrate scales as the weight distribution shifts.
* **convert_qat** folds the learned scales into a regular ``nn.Linear``
  with the int8-rounded-then-dequantized weights, so the output is
  drop-in compatible with standard PyTorch inference.

Portability: all tensor ops are pure PyTorch; works on HIP and CUDA
builds alike.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bnb_rocm import _set_module


class QATLinear(nn.Module):
    """Quantization-aware training linear layer (per-channel int8 by default).

    The weight is stored in full precision and passed through
    ``torch.fake_quantize_per_channel_affine`` during training. Gradients
    flow through as a straight-through estimator. At eval time the same
    fake-quant path is used so the two modes are bit-identical (no train
    vs eval drift).
    """

    def __init__(
        self,
        linear: nn.Linear,
        bits: int = 8,
        per_channel: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.per_channel = per_channel
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1

        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = linear.bias

        # Calibrate scale from the freshly cloned weights. `clamp(min=eps)`
        # protects against zero-weight rows (rare but not impossible).
        if per_channel:
            # Per output channel: weight shape (out, in) -> scale shape (out,)
            max_abs = self.weight.detach().abs().amax(dim=1)
        else:
            max_abs = self.weight.detach().abs().amax().reshape(1)
        scale = torch.clamp(max_abs / self.qmax, min=1e-8).to(torch.float32)
        zero_point = torch.zeros_like(scale, dtype=torch.int32)

        self.register_buffer("weight_scale", scale)
        self.register_buffer("weight_zero_point", zero_point)

    # ------------------------------------------------------------------
    # Calibration / EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def recalibrate(self, momentum: float = 0.01) -> None:
        """Re-estimate weight scales from the current weight values.

        ``momentum == 1.0`` is a hard reset (replaces the scale). Anything
        in (0, 1) is an EMA update:  ``s <- (1-m)*s + m*s_new``. The
        typical cadence is every ~50-100 training steps with
        ``momentum = 0.01`` so the fake-quant range tracks slow shifts
        without oscillating.
        """
        if not (0.0 < momentum <= 1.0):
            raise ValueError(f"momentum must be in (0, 1], got {momentum}")

        if self.per_channel:
            new_max = self.weight.abs().amax(dim=1)
        else:
            new_max = self.weight.abs().amax().reshape(1)
        new_scale = torch.clamp(new_max / self.qmax, min=1e-8).to(torch.float32)

        if momentum >= 1.0:
            self.weight_scale.data.copy_(new_scale)
        else:
            self.weight_scale.data.mul_(1.0 - momentum).add_(
                new_scale, alpha=momentum
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _fake_quantize_weight(self) -> torch.Tensor:
        if self.per_channel:
            return torch.fake_quantize_per_channel_affine(
                self.weight,
                self.weight_scale,
                self.weight_zero_point,
                axis=0,
                quant_min=self.qmin,
                quant_max=self.qmax,
            )
        # Per-tensor path (legacy)
        return torch.fake_quantize_per_tensor_affine(
            self.weight,
            self.weight_scale.reshape(()),
            self.weight_zero_point.reshape(()),
            self.qmin,
            self.qmax,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._fake_quantize_weight(), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bits={self.bits}, per_channel={self.per_channel}"
        )


def apply_qat(
    model: nn.Module,
    bits: int = 8,
    per_channel: bool = True,
    skip_names: Optional[Iterable[str]] = None,
) -> nn.Module:
    """Wrap all ``nn.Linear`` layers with :class:`QATLinear` modules.

    ``skip_names`` is an iterable of substrings. Any module whose qualified
    name contains one of them is left unwrapped. Defaults to
    ``("lm_head", "embed_tokens", "score")`` because quantizing the output
    projection and embedding table usually hurts more than it helps.
    """
    skip = tuple(
        skip_names if skip_names is not None else ("lm_head", "embed_tokens", "score")
    )
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in skip):
            continue
        qat = QATLinear(module, bits=bits, per_channel=per_channel)
        _set_module(model, name, qat)
    return model


def qat_update_scales(model: nn.Module, momentum: float = 0.01) -> int:
    """Re-estimate QAT scales across every ``QATLinear`` in ``model``.

    Returns the number of layers updated, for logging. Safe to call from
    a training callback after every N steps.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, QATLinear):
            module.recalibrate(momentum=momentum)
            n += 1
    return n


def convert_qat(model: nn.Module, bits: int = 8) -> nn.Module:
    """Convert :class:`QATLinear` layers to plain ``nn.Linear`` with the
    quantized-then-dequantized weights baked in.

    The resulting model is standard PyTorch (no custom modules) and can
    be saved / loaded like any other checkpoint. The weights are stored
    in the dtype of the trained weight tensor; downstream int8 inference
    runtimes can re-quantize them losslessly because they already live
    on the int8 quantization grid.
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, QATLinear):
            continue
        with torch.no_grad():
            weight = module._fake_quantize_weight().clone()
        linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
        linear.weight = nn.Parameter(weight.to(module.weight.dtype))
        if module.bias is not None:
            linear.bias = module.bias
        _set_module(model, name, linear)
    return model


# ---------------------------------------------------------------------------
# Trainer callback
# ---------------------------------------------------------------------------


class QATScaleCallback:
    """HuggingFace Trainer callback that calls :func:`qat_update_scales`
    every ``interval`` steps with the configured EMA ``momentum``.

    Duck-typed to the TrainerCallback protocol so it works without a hard
    transformers dependency at import time.
    """

    def __init__(self, interval: int = 50, momentum: float = 0.01) -> None:
        self.interval = max(1, int(interval))
        self.momentum = float(momentum)
        self._model: Optional[nn.Module] = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):  # noqa: D401
        self._model = model

    def on_step_end(self, args, state, control, **kwargs):  # noqa: D401
        if self._model is None:
            return
        if state.global_step > 0 and state.global_step % self.interval == 0:
            qat_update_scales(self._model, momentum=self.momentum)
