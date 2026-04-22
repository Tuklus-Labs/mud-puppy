"""Tests for FP8 training infrastructure.

The tests exercise the emulated path (no hardware FP8 required) so they
run on RDNA3 and CPU. On RDNA4 / MI300+ the hardware path takes over
automatically; the module-level behavior tested here is identical.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from mud_puppy.fp8_rocm import (
    FP8Linear,
    apply_fp8,
    fp8_layer_count,
    is_fp8_hardware_available,
)


# ---------------------------------------------------------------------------
# Build + replacement
# ---------------------------------------------------------------------------


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.lm_head = nn.Linear(16, 50)

    def forward(self, x):
        return self.lm_head(self.body(x))


def test_apply_fp8_wraps_body_linears_skips_lm_head():
    m = ToyModel()
    apply_fp8(m)
    assert isinstance(m.body[0], FP8Linear)
    assert isinstance(m.body[2], FP8Linear)
    # lm_head in the default skip list
    assert isinstance(m.lm_head, nn.Linear) and not isinstance(m.lm_head, FP8Linear)


def test_fp8_layer_count_reports_correctly():
    m = ToyModel()
    apply_fp8(m)
    assert fp8_layer_count(m) == 2


def test_apply_fp8_idempotent_does_not_double_wrap():
    m = ToyModel()
    apply_fp8(m)
    apply_fp8(m)
    assert fp8_layer_count(m) == 2  # second call is a no-op


# ---------------------------------------------------------------------------
# Forward numerics (emulation path)
# ---------------------------------------------------------------------------


def test_fp8_linear_forward_shape_matches_plain_linear():
    lin = nn.Linear(32, 16)
    fp8 = FP8Linear(lin)
    x = torch.randn(4, 32)
    y = fp8(x)
    assert y.shape == (4, 16)


def test_fp8_linear_forward_approximates_plain_linear():
    torch.manual_seed(0)
    lin = nn.Linear(32, 16)
    fp8 = FP8Linear(lin)
    # Warm up amax buffers with a few forward passes; delayed scaling
    # means the first step sees stale scales.
    x = torch.randn(8, 32)
    for _ in range(5):
        _ = fp8(x)
    y_plain = lin(x)
    y_fp8 = fp8(x)
    # Max error should be small relative to signal magnitude
    rel = (y_plain - y_fp8).abs().max().item() / (y_plain.abs().max().item() + 1e-6)
    assert rel < 0.1, f"FP8 forward diverges: rel={rel:.4f}"


def test_fp8_linear_with_leading_batch_dims():
    """FP8Linear must handle (..., in_features) like nn.Linear does."""
    lin = nn.Linear(16, 8)
    fp8 = FP8Linear(lin)
    x = torch.randn(2, 3, 4, 16)  # three leading batch dims
    y = fp8(x)
    assert y.shape == (2, 3, 4, 8)


# ---------------------------------------------------------------------------
# Amax / scale tracking
# ---------------------------------------------------------------------------


def test_amax_buffers_initialized():
    lin = nn.Linear(8, 8)
    fp8 = FP8Linear(lin)
    assert fp8.input_amax.item() == pytest.approx(1.0)
    assert fp8.weight_amax.item() == pytest.approx(1.0)


def test_amax_updates_after_forward():
    torch.manual_seed(1)
    lin = nn.Linear(8, 8)
    fp8 = FP8Linear(lin, amax_momentum=0.5)  # fast decay for test visibility
    x = torch.randn(4, 8) * 5.0  # large-magnitude inputs
    before = fp8.input_amax.item()
    _ = fp8(x)
    after = fp8.input_amax.item()
    # Weight amax should also reflect actual weight magnitudes (not just 1.0)
    assert fp8.weight_amax.item() == pytest.approx(
        max(0.5, lin.weight.abs().amax().item()), rel=0.01
    )
    # Input amax should have grown (started at 1.0, input had |x|~5)
    assert after > before


def test_amax_decays_when_inputs_shrink():
    """EMA decay means small inputs eventually lower the scale."""
    torch.manual_seed(2)
    lin = nn.Linear(8, 8)
    fp8 = FP8Linear(lin, amax_momentum=0.5)
    big = torch.randn(4, 8) * 10
    _ = fp8(big)
    high = fp8.input_amax.item()
    # Now feed small inputs for many steps
    small = torch.randn(4, 8) * 0.1
    for _ in range(30):
        _ = fp8(small)
    low = fp8.input_amax.item()
    assert low < high, f"amax did not decay: {low} vs {high}"


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_fp8_linear_gradients_flow_to_master_weight():
    torch.manual_seed(3)
    lin = nn.Linear(16, 8)
    fp8 = FP8Linear(lin)
    x = torch.randn(4, 16, requires_grad=True)
    out = fp8(x)
    loss = out.sum()
    loss.backward()
    assert fp8.weight.grad is not None
    assert fp8.weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def test_is_fp8_hardware_available_returns_bool():
    # Just make sure it doesn't crash and returns a bool.
    r = is_fp8_hardware_available()
    assert isinstance(r, bool)


# ---------------------------------------------------------------------------
# State_dict round-trip
# ---------------------------------------------------------------------------


def test_fp8_linear_state_dict_roundtrip():
    torch.manual_seed(4)
    lin = nn.Linear(16, 8)
    fp8 = FP8Linear(lin)
    # Drive amax buffers off their init values
    for _ in range(3):
        _ = fp8(torch.randn(2, 16) * 3)
    state = fp8.state_dict()
    assert "weight" in state
    assert "input_amax" in state
    assert "weight_amax" in state

    # Round-trip
    lin2 = nn.Linear(16, 8)
    fp8b = FP8Linear(lin2)
    fp8b.load_state_dict(state)
    # The amax buffers should match
    assert torch.allclose(fp8b.input_amax, fp8.input_amax)
    assert torch.allclose(fp8b.weight_amax, fp8.weight_amax)
    assert torch.allclose(fp8b.weight, fp8.weight)
