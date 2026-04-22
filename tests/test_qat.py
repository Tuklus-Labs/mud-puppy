"""Tests for QAT (quantization-aware training) correctness.

The pre-2026-04-22 implementation initialised ``weight_scale`` to 1.0 and
never updated it, so fake-quant was effectively a clamp. These tests lock
in the fix: per-channel scale calibrated from real weights, EMA update
path, and convert_qat producing a plain Linear whose weights live on the
int8 quantization grid.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from mud_puppy.qat_rocm import (
    QATLinear,
    QATScaleCallback,
    apply_qat,
    convert_qat,
    qat_update_scales,
)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_qat_linear_scale_calibrated_on_construction():
    """Scale must be populated from |w|/qmax at init, not left at 1.0."""
    torch.manual_seed(0)
    lin = nn.Linear(32, 16)
    # Force known weight magnitudes so we can check the scale exactly.
    with torch.no_grad():
        lin.weight.zero_()
        lin.weight[0, :] = 1.0  # row 0 max = 1.0
        lin.weight[1, :] = 0.5  # row 1 max = 0.5
        lin.weight[2, :] = 0.25  # row 2 max = 0.25
    qat = QATLinear(lin, bits=8)
    # qmax for int8 = 127
    assert qat.weight_scale.shape == (16,), "per-channel scale expected"
    assert math.isclose(qat.weight_scale[0].item(), 1.0 / 127, rel_tol=1e-4)
    assert math.isclose(qat.weight_scale[1].item(), 0.5 / 127, rel_tol=1e-4)
    assert math.isclose(qat.weight_scale[2].item(), 0.25 / 127, rel_tol=1e-4)
    # Zero-weight rows should get the floor, not zero.
    assert qat.weight_scale[3].item() > 0.0
    # Floor sits at ~1e-8 (fp32 precision).
    assert qat.weight_scale[3].item() < 1e-6


def test_qat_linear_per_tensor_mode():
    """per_channel=False returns a scalar-shaped scale calibrated from the whole tensor."""
    torch.manual_seed(1)
    lin = nn.Linear(32, 16)
    qat = QATLinear(lin, bits=8, per_channel=False)
    assert qat.weight_scale.shape == (1,)
    expected = lin.weight.abs().amax().item() / 127
    assert math.isclose(qat.weight_scale.item(), expected, rel_tol=1e-4)


def test_qat_linear_forward_differs_from_plain_linear():
    """With real calibrated scales, fake-quant should actually change the output."""
    torch.manual_seed(2)
    lin = nn.Linear(64, 32)
    qat = QATLinear(lin, bits=4)  # 4 bits = bigger quant error, easier to detect
    x = torch.randn(8, 64)
    plain_out = lin(x)
    qat_out = qat(x)
    # The difference should be nonzero (fake-quant is actually doing something)
    # but small (STE doesn't totally destroy the output).
    diff = (plain_out - qat_out).abs().mean().item()
    assert diff > 1e-4, "fake-quant should perturb output"
    assert diff < 1.0, "fake-quant shouldn't blow up output"


# ---------------------------------------------------------------------------
# Gradients (straight-through estimator)
# ---------------------------------------------------------------------------


def test_qat_gradients_flow_through_fake_quant():
    """Backward must pass gradients through to the underlying weight."""
    torch.manual_seed(3)
    lin = nn.Linear(16, 8)
    qat = QATLinear(lin, bits=8)
    x = torch.randn(4, 16, requires_grad=True)
    loss = qat(x).sum()
    loss.backward()
    assert qat.weight.grad is not None
    assert qat.weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# EMA scale update
# ---------------------------------------------------------------------------


def test_recalibrate_hard_reset_matches_weight_distribution():
    torch.manual_seed(4)
    lin = nn.Linear(16, 4)
    qat = QATLinear(lin, bits=8)
    with torch.no_grad():
        qat.weight.mul_(10.0)  # blow up weights so scale goes stale
    qat.recalibrate(momentum=1.0)  # hard reset
    expected = qat.weight.abs().amax(dim=1).item() if False else None
    # Per-channel shape preserved
    assert qat.weight_scale.shape == (4,)
    # New scale should reflect 10x the original magnitudes
    for i in range(4):
        row_max = qat.weight[i].abs().amax().item()
        assert math.isclose(qat.weight_scale[i].item(), row_max / 127, rel_tol=1e-4)


def test_recalibrate_ema_moves_toward_new_value():
    torch.manual_seed(5)
    lin = nn.Linear(16, 4)
    qat = QATLinear(lin, bits=8)
    initial_scale = qat.weight_scale.clone()
    with torch.no_grad():
        qat.weight.mul_(10.0)
    qat.recalibrate(momentum=0.1)  # 10% EMA step
    # Scale should have moved toward new value but not reached it
    row0_target = qat.weight[0].abs().amax().item() / 127
    assert initial_scale[0].item() < qat.weight_scale[0].item() < row0_target


def test_recalibrate_rejects_invalid_momentum():
    lin = nn.Linear(4, 2)
    qat = QATLinear(lin)
    with pytest.raises(ValueError):
        qat.recalibrate(momentum=0.0)
    with pytest.raises(ValueError):
        qat.recalibrate(momentum=1.5)


# ---------------------------------------------------------------------------
# apply_qat / convert_qat end-to-end
# ---------------------------------------------------------------------------


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.lm_head = nn.Linear(16, 100)

    def forward(self, x):
        return self.lm_head(self.body(x))


def test_apply_qat_wraps_body_linears_but_skips_lm_head():
    model = ToyModel()
    apply_qat(model, bits=8)
    # Body Linears should be QATLinear now
    assert isinstance(model.body[0], QATLinear)
    assert isinstance(model.body[2], QATLinear)
    # lm_head should be left alone (default skip list)
    assert isinstance(model.lm_head, nn.Linear) and not isinstance(
        model.lm_head, QATLinear
    )


def test_convert_qat_produces_plain_linear_with_grid_weights():
    model = ToyModel()
    apply_qat(model, bits=8)
    # Train step so weights drift from init values
    x = torch.randn(4, 16)
    out = model(x)
    out.sum().backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data -= 0.01 * p.grad

    # Capture the stored scales BEFORE convert_qat rewrites the modules.
    saved_scales = {
        name: m.weight_scale.detach().clone()
        for name, m in model.named_modules()
        if isinstance(m, QATLinear)
    }

    convert_qat(model, bits=8)
    # Every wrapped layer must now be plain nn.Linear
    assert isinstance(model.body[0], nn.Linear) and not isinstance(
        model.body[0], QATLinear
    )
    assert isinstance(model.body[2], nn.Linear) and not isinstance(
        model.body[2], QATLinear
    )
    # Weights should live on the int8 grid defined by the PRE-CONVERT
    # per-channel scale (convert uses fake_quantize_per_channel_affine
    # which snaps weight to round(w/s)*s, zero_point=0).
    for name in ("body.0", "body.2"):
        layer = model.get_submodule(name)
        scale = saved_scales[name]
        grid = layer.weight / scale.unsqueeze(1)
        rounding_err = (grid - grid.round()).abs().max().item()
        assert rounding_err < 1e-3, (
            f"{name}: weights not on int8 grid (err={rounding_err})"
        )
        # And the integer values should sit inside [qmin, qmax] = [-128, 127].
        grid_int = grid.round()
        assert grid_int.max().item() <= 127
        assert grid_int.min().item() >= -128


# ---------------------------------------------------------------------------
# qat_update_scales helper
# ---------------------------------------------------------------------------


def test_qat_update_scales_counts_layers():
    model = ToyModel()
    apply_qat(model, bits=8)
    n = qat_update_scales(model, momentum=0.5)
    # body[0] + body[2] = 2 QATLinear layers (lm_head skipped)
    assert n == 2


# ---------------------------------------------------------------------------
# Callback duck-typing
# ---------------------------------------------------------------------------


def test_qat_scale_callback_fires_on_interval():
    cb = QATScaleCallback(interval=10, momentum=0.1)
    model = ToyModel()
    apply_qat(model, bits=8)
    # Stash original scales to detect changes
    original = [m.weight_scale.clone() for m in model.modules() if isinstance(m, QATLinear)]

    # Simulate a few Trainer events
    class _S:
        global_step = 0

    state = _S()
    cb.on_train_begin(None, state, None, model=model)

    # Step 5: no update (not a multiple of 10)
    state.global_step = 5
    cb.on_step_end(None, state, None)
    for m, o in zip(
        [mm for mm in model.modules() if isinstance(mm, QATLinear)], original
    ):
        assert torch.equal(m.weight_scale, o), "scale shouldn't have changed at step 5"

    # Step 10: update fires. Modify weights first so EMA visibly moves.
    for m in model.modules():
        if isinstance(m, QATLinear):
            with torch.no_grad():
                m.weight.mul_(5.0)
    state.global_step = 10
    cb.on_step_end(None, state, None)
    for m, o in zip(
        [mm for mm in model.modules() if isinstance(mm, QATLinear)], original
    ):
        assert not torch.equal(m.weight_scale, o), "scale should have updated at step 10"
