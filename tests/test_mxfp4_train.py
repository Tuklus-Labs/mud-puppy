"""Correctness tests for MXFP4 quantized-training modules.

Locks in the STE backward, module swap, deploy conversion, and toy
training-loop convergence. If any of these fail, the whole "train on
the MXFP4 grid and deploy without conversion" story breaks.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mud_puppy.mxfp4 import quantize_mxfp4
from mud_puppy.mxfp4_train import (
    MXFP4QATLinear,
    apply_mxfp4_qat,
    convert_mxfp4_qat_to_linear,
    fake_quantize_mxfp4,
    pack_mxfp4_qat,
)


# ---------------------------------------------------------------------------
# Fake-quant op: STE gradient
# ---------------------------------------------------------------------------


def test_fake_quantize_forward_matches_quantize_mxfp4() -> None:
    """The autograd op is a thin wrapper on quantize_mxfp4; values must match."""
    torch.manual_seed(0)
    w = torch.randn(64, 32, dtype=torch.float32, requires_grad=True)
    y_op = fake_quantize_mxfp4(w, block_size=32)
    y_ref = quantize_mxfp4(w.detach(), block_size=32)
    assert torch.equal(y_op.detach(), y_ref)


def test_fake_quantize_ste_gradient() -> None:
    """STE: gradient through the quant op is identity."""
    w = torch.randn(64, 32, dtype=torch.float32, requires_grad=True)
    y = fake_quantize_mxfp4(w, block_size=32)
    upstream = torch.randn_like(y)
    y.backward(upstream)
    assert w.grad is not None
    assert torch.equal(w.grad, upstream), (
        "STE must pass gradients through unchanged; a bug here biases training"
    )


# ---------------------------------------------------------------------------
# Module: forward + backward
# ---------------------------------------------------------------------------


def test_linear_forward_on_grid() -> None:
    """Output of the module's forward is equivalent to F.linear(x, on_grid_w, bias)."""
    torch.manual_seed(1)
    base = nn.Linear(128, 64, bias=True)
    mxfp = MXFP4QATLinear.from_linear(base, block_size=32)

    x = torch.randn(8, 128)
    y = mxfp(x)
    # Reference: fake-quantize the same weight, then linear
    w_fq = quantize_mxfp4(base.weight, block_size=32)
    y_ref = x @ w_fq.T + base.bias
    assert torch.allclose(y, y_ref, atol=1e-5)


def test_linear_backward_reaches_weight() -> None:
    """Gradients must flow through fake-quant to the master weight."""
    torch.manual_seed(2)
    base = nn.Linear(32, 16, bias=False)
    mxfp = MXFP4QATLinear.from_linear(base, block_size=32)

    x = torch.randn(4, 32)
    mxfp(x).sum().backward()
    assert mxfp.weight.grad is not None
    assert mxfp.weight.grad.abs().sum() > 0
    assert torch.isfinite(mxfp.weight.grad).all()


def test_linear_preserves_dtype() -> None:
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        base = nn.Linear(32, 16).to(dtype)
        mxfp = MXFP4QATLinear.from_linear(base)
        x = torch.randn(4, 32, dtype=dtype)
        y = mxfp(x)
        assert y.dtype == dtype


def test_linear_bias_preserved() -> None:
    base = nn.Linear(32, 16, bias=True)
    mxfp = MXFP4QATLinear.from_linear(base)
    assert torch.equal(mxfp.bias.detach(), base.bias.detach())


# ---------------------------------------------------------------------------
# Deploy: pack and convert
# ---------------------------------------------------------------------------


def test_pack_matches_forward_output() -> None:
    """``pack()`` + ``unpack()`` must equal the QAT forward's internal quant."""
    torch.manual_seed(3)
    base = nn.Linear(128, 64, bias=False)
    mxfp = MXFP4QATLinear.from_linear(base, block_size=32)
    # Forward value of the quantized weight (what the layer actually uses)
    w_forward = fake_quantize_mxfp4(mxfp.weight, block_size=32).detach()
    w_packed = mxfp.unpacked_weight()
    assert torch.equal(w_forward, w_packed)


def test_convert_to_plain_linear_preserves_output() -> None:
    """convert_mxfp4_qat_to_linear must yield forward equal to the QAT forward."""
    torch.manual_seed(4)
    model = nn.Sequential(
        MXFP4QATLinear.from_linear(nn.Linear(64, 128, bias=True)),
        nn.ReLU(),
        MXFP4QATLinear.from_linear(nn.Linear(128, 32, bias=True)),
    )
    x = torch.randn(8, 64)
    y_qat = model(x).detach().clone()
    convert_mxfp4_qat_to_linear(model)
    # Every MXFP4QATLinear must now be a plain nn.Linear.
    for m in model.modules():
        assert not isinstance(m, MXFP4QATLinear)
    y_deploy = model(x)
    assert torch.allclose(y_qat, y_deploy, atol=1e-5), (
        f"deploy drift: {(y_qat-y_deploy).abs().max()}"
    )


# ---------------------------------------------------------------------------
# Model-wide helpers
# ---------------------------------------------------------------------------


class _ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.lm_head = nn.Linear(32, 100)

    def forward(self, x):
        return self.lm_head(self.body(x))


def test_apply_mxfp4_qat_wraps_body_skips_head() -> None:
    model = _ToyNet()
    apply_mxfp4_qat(model, block_size=32)
    assert isinstance(model.body[0], MXFP4QATLinear)
    assert isinstance(model.body[2], MXFP4QATLinear)
    # lm_head skipped by default
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, MXFP4QATLinear)


def test_apply_mxfp4_qat_custom_skip() -> None:
    model = _ToyNet()
    apply_mxfp4_qat(model, skip_names=())  # no skips -- wrap everything
    assert isinstance(model.lm_head, MXFP4QATLinear)


def test_apply_mxfp4_qat_idempotent() -> None:
    model = _ToyNet()
    apply_mxfp4_qat(model)
    apply_mxfp4_qat(model)  # second call should be a no-op
    count = sum(1 for m in model.modules() if isinstance(m, MXFP4QATLinear))
    assert count == 2  # body[0] and body[2]


def test_pack_mxfp4_qat_returns_all_layers() -> None:
    model = _ToyNet()
    apply_mxfp4_qat(model)
    packed = pack_mxfp4_qat(model)
    assert len(packed) == 2
    for name, (nibbles, scales, shape) in packed.items():
        assert nibbles.dtype == torch.uint8
        assert scales.dtype == torch.uint8


# ---------------------------------------------------------------------------
# Toy training: loss must drop
# ---------------------------------------------------------------------------


def test_training_loop_reduces_loss() -> None:
    """Training steps on an MXFP4-QAT model must drive MSE loss down.

    MXFP4 adds quantization noise on every forward (roughly scale/2 per
    weight), which makes SGD slower to converge than fp training. Adam
    handles the noisy gradient signal well. Threshold 40% drop over 80
    steps is comfortably in the "actually learning" regime -- a broken
    STE path stalls near 0% or diverges.
    """
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 8),
    )
    apply_mxfp4_qat(model)

    x = torch.randn(64, 16)
    target = torch.randn(64, 8)

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(80):
        opt.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.6, (
        f"training failed to reduce loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    # No NaNs during training
    assert all(l == l for l in losses), "NaN encountered during QAT training"


def test_weights_stay_valid_after_many_steps() -> None:
    """After many optimizer steps, master weights must remain finite and
    the quantized forward must stay bounded."""
    torch.manual_seed(8)
    layer = MXFP4QATLinear(32, 16, bias=False, block_size=32)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
    for _ in range(100):
        x = torch.randn(8, 32)
        loss = layer(x).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(layer.weight).all()
    out = layer(torch.randn(1, 32))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# GPU smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA/HIP")
def test_on_gpu() -> None:
    layer = MXFP4QATLinear(64, 32, bias=True, block_size=32).cuda().to(torch.bfloat16)
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert y.device.type == "cuda"
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
