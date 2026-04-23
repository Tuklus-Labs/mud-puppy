"""Correctness tests for the MXFP4Linear packed-storage module.

These gate:
    * Drop-in replacement semantics: output matches the source nn.Linear
      within MXFP4 quantization error.
    * Memory: the wrapped module holds only packed qweight + scales, not
      a bf16 master weight.
    * Backward flows through to input gradient (needed for LoRA on top).
    * PEFT compatibility: ``isinstance(mxfp, nn.Linear)`` is True; the
      ``.weight`` property dequantizes on access without crashing.
    * Fallback: with Triton disabled, forward still produces matching
      output (via pure-PyTorch unpack + F.linear).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mud_puppy.mxfp4_kernels import MXFP4Linear, quantize_model_mxfp4


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_wraps_linear_with_packed_buffers() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(128, 64, bias=True)
    mx = MXFP4Linear(lin)
    # Packed weight lives in qweight, not weight
    assert "qweight" in dict(mx.named_buffers())
    assert "scales" in dict(mx.named_buffers())
    # In/out features preserved
    assert mx.in_features == 128
    assert mx.out_features == 64
    # Bias carried across as a frozen parameter
    assert mx.bias is not None
    assert mx.bias.requires_grad is False
    # Packed size: N * K/2 = 64 * 64 = 4096 bytes; scales: 64 * 4 = 256 bytes
    assert mx.qweight.numel() == 64 * 64
    assert mx.scales.numel() == 64 * 4


def test_no_bias_path() -> None:
    lin = nn.Linear(128, 64, bias=False)
    mx = MXFP4Linear(lin)
    assert mx.bias is None


def test_pads_odd_k_to_block_boundary() -> None:
    """K=100 must be padded to 128 (next multiple of 32) for block alignment."""
    lin = nn.Linear(100, 32, bias=False)
    mx = MXFP4Linear(lin)
    assert mx.in_features == 100
    assert mx._padded_in == 128
    # Packed stride still accounts for the padded width
    assert mx.qweight.shape[-1] == 64  # 128 / 2


# ---------------------------------------------------------------------------
# Forward: fallback path (no Triton)
# ---------------------------------------------------------------------------


def test_forward_matches_source_linear_within_quant_error(monkeypatch) -> None:
    """Output must match source nn.Linear up to MXFP4 quantization error.

    Compare absolute error against the expected per-output magnitude
    (bounded by the grid spacing at this scale). For a 64-dim dot
    product of ~N(0,1) weights with N(0,1) inputs, outputs have std ~8.
    MXFP4 relative error is ~3% average, so we expect RMS error <= 0.5.
    """
    monkeypatch.delenv("MUD_PUPPY_MXFP4_TRITON", raising=False)
    torch.manual_seed(1)
    lin = nn.Linear(64, 32, bias=False).to(torch.bfloat16)
    mx = MXFP4Linear(lin, dtype=torch.bfloat16)

    x = torch.randn(16, 64, dtype=torch.bfloat16)  # bigger batch for stable stats
    y_ref = lin(x).float()
    y_mx = mx(x).float()

    # Compare RMS relative to RMS of reference. Outputs with magnitude
    # close to zero are dominated by bf16 rounding, not MXFP4 error.
    rms_err = ((y_ref - y_mx) ** 2).mean().sqrt().item()
    rms_ref = (y_ref ** 2).mean().sqrt().item()
    rel_rms = rms_err / (rms_ref + 1e-6)
    assert rel_rms < 0.10, (
        f"RMS relative error too high: {rel_rms:.3f} (rms_err={rms_err:.3f}, rms_ref={rms_ref:.3f})"
    )


def test_forward_no_bias() -> None:
    lin = nn.Linear(64, 32, bias=False)
    mx = MXFP4Linear(lin)
    x = torch.randn(4, 64)
    y = mx(x)
    assert y.shape == (4, 32)
    assert torch.isfinite(y).all()


def test_forward_handles_padded_k(monkeypatch) -> None:
    """K=100 (padded to 128) forward must still produce output of shape [..., N]."""
    monkeypatch.delenv("MUD_PUPPY_MXFP4_TRITON", raising=False)
    lin = nn.Linear(100, 32, bias=False)
    mx = MXFP4Linear(lin)
    x = torch.randn(4, 100)
    y = mx(x)
    assert y.shape == (4, 32)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def test_backward_grad_input_flows() -> None:
    torch.manual_seed(2)
    lin = nn.Linear(64, 32, bias=False)
    mx = MXFP4Linear(lin)
    x = torch.randn(4, 64, requires_grad=True)
    y = mx(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_backward_no_grad_for_qweight() -> None:
    """Packed qweight is a buffer -- must not try to accumulate gradient."""
    mx = MXFP4Linear(nn.Linear(32, 16, bias=False))
    x = torch.randn(4, 32, requires_grad=True)
    mx(x).sum().backward()
    # Buffers don't have .grad; explicitly verify no gradient was plumbed.
    assert not hasattr(mx.qweight, "grad") or mx.qweight.grad is None
    assert not hasattr(mx.scales, "grad") or mx.scales.grad is None


def test_backward_with_bias() -> None:
    torch.manual_seed(3)
    lin = nn.Linear(32, 16, bias=True)
    mx = MXFP4Linear(lin)
    x = torch.randn(4, 32, requires_grad=True)
    mx(x).sum().backward()
    assert x.grad is not None
    # Bias is frozen: no gradient update expected.
    assert mx.bias.requires_grad is False


# ---------------------------------------------------------------------------
# PEFT / nn.Linear compatibility
# ---------------------------------------------------------------------------


def test_isinstance_nn_linear() -> None:
    """PEFT identifies quantizable layers via isinstance(..., nn.Linear)."""
    mx = MXFP4Linear(nn.Linear(32, 16))
    assert isinstance(mx, nn.Linear)


def test_weight_property_dequantizes() -> None:
    """PEFT may read .weight to copy dtype/device metadata. Must not crash."""
    lin = nn.Linear(32, 16, bias=False)
    mx = MXFP4Linear(lin)
    w = mx.weight
    assert w.shape == (16, 32)
    assert w.dtype == torch.bfloat16


def test_weight_setter_is_noop() -> None:
    """PEFT sometimes assigns to .weight; must not throw or actually change state."""
    mx = MXFP4Linear(nn.Linear(32, 16))
    old_qw = mx.qweight.clone()
    mx.weight = torch.randn(16, 32)  # no-op
    assert torch.equal(mx.qweight, old_qw)


# ---------------------------------------------------------------------------
# Model-wide helper
# ---------------------------------------------------------------------------


class _ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.lm_head = nn.Linear(128, 1000)


def test_quantize_model_mxfp4_swaps_body_skips_head() -> None:
    model = _ToyNet()
    quantize_model_mxfp4(model)
    assert isinstance(model.body[0], MXFP4Linear)
    assert isinstance(model.body[2], MXFP4Linear)
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, MXFP4Linear)


def test_quantize_model_mxfp4_skips_small_layers() -> None:
    """min_size threshold avoids quantizing tiny layers where packed
    storage overhead outweighs savings."""
    model = nn.Sequential(
        nn.Linear(4, 4),           # 16 elements, tiny
        nn.Linear(128, 128),       # 16384 elements, worth wrapping
    )
    quantize_model_mxfp4(model, min_size=1024)
    assert not isinstance(model[0], MXFP4Linear)  # skipped
    assert isinstance(model[1], MXFP4Linear)


# ---------------------------------------------------------------------------
# Triton kernel path (when enabled)
# ---------------------------------------------------------------------------


try:
    import triton  # noqa: F401
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


@pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON),
    reason="no CUDA/HIP + Triton",
)
def test_triton_forward_matches_fallback(monkeypatch) -> None:
    """With env var on vs off, forward outputs must match within bf16 noise."""
    torch.manual_seed(5)
    lin = nn.Linear(512, 256, bias=True).cuda().to(torch.bfloat16)
    mx = MXFP4Linear(lin, dtype=torch.bfloat16).cuda()
    x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)

    monkeypatch.delenv("MUD_PUPPY_MXFP4_TRITON", raising=False)
    y_pt = mx(x).detach().clone()

    monkeypatch.setenv("MUD_PUPPY_MXFP4_TRITON", "1")
    y_tr = mx(x).detach().clone()

    diff = (y_tr.float() - y_pt.float()).abs()
    ref = y_pt.float().abs()
    rel_ok = diff <= 1e-2 * ref
    abs_ok = diff <= 1e-2
    assert (rel_ok | abs_ok).all(), (
        f"triton/fallback diverge: max diff={diff.max().item():.4e}"
    )


@pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON),
    reason="no CUDA/HIP + Triton",
)
def test_triton_backward_matches_fallback(monkeypatch) -> None:
    """End-to-end: forward + backward through MXFP4Linear must produce
    matching grad_input between triton and pytorch paths. If this
    fails, training diverges over many steps."""
    torch.manual_seed(6)
    lin = nn.Linear(256, 128, bias=False).cuda().to(torch.bfloat16)
    mx = MXFP4Linear(lin, dtype=torch.bfloat16).cuda()
    x_base = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)

    def run(use_triton: bool):
        if use_triton:
            monkeypatch.setenv("MUD_PUPPY_MXFP4_TRITON", "1")
        else:
            monkeypatch.delenv("MUD_PUPPY_MXFP4_TRITON", raising=False)
        x = x_base.detach().clone().requires_grad_(True)
        y = mx(x)
        y.sum().backward()
        return y.detach().clone(), x.grad.detach().clone()

    y_pt, g_pt = run(False)
    y_tr, g_tr = run(True)
    assert (y_tr.float() - y_pt.float()).abs().max().item() < 5e-2
    diff = (g_tr.float() - g_pt.float()).abs()
    rel = g_pt.float().abs().clamp_min(1e-3)
    assert ((diff / rel) <= 2e-2).all() or (diff <= 5e-2).all(), (
        f"grad_input diverges between triton and pytorch: max diff={diff.max().item():.4e}"
    )
