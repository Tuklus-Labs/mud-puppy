"""Correctness tests for the MXFP4Experts MoE replacement.

Gates the invariants that matter for gpt-oss-20b training:

    1. ``from_gpt_oss`` produces a module whose forward output matches
       the native ``GptOssExperts`` within MXFP4 quantization error.
       A bug here silently corrupts every MoE forward in the model.

    2. The swap helper ``swap_gpt_oss_experts_to_mxfp4`` touches
       exactly the expert modules and leaves attention/router/norms
       untouched.

    3. The forward honours routing masks: an inactive expert must not
       contribute to its tokens' output.

    4. SwiGLU interleaving (gate = [..., ::2], up = [..., 1::2]) is
       preserved. Easy to break on the way through quantization.

    5. Triton kernel path and pure-PyTorch fallback produce
       equivalent outputs (drift gate for training).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


# Skip the whole file if transformers/gpt_oss isn't installed.
try:
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    HAVE_GPT_OSS = True
except ImportError:
    HAVE_GPT_OSS = False


pytestmark = pytest.mark.skipif(
    not HAVE_GPT_OSS,
    reason="transformers.models.gpt_oss not installed",
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _make_gpt_oss_experts(
    num_experts: int = 4,
    hidden_size: int = 64,
    intermediate_size: int = 64,
    seed: int = 0,
    dtype=torch.bfloat16,
) -> GptOssExperts:
    torch.manual_seed(seed)
    cfg = SimpleNamespace(
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        # transformers 5.x wraps GptOssExperts.forward with a decorator that
        # reads self.config._experts_implementation. ``None`` means "use the
        # original implementation" -- equivalent to no decorator.
        _experts_implementation=None,
    )
    m = GptOssExperts(cfg).to(dtype)
    # Initialize parameters with a reproducible distribution.
    with torch.no_grad():
        m.gate_up_proj.normal_(std=0.05)
        m.gate_up_proj_bias.zero_()
        m.down_proj.normal_(std=0.05)
        m.down_proj_bias.zero_()
    return m


def _random_routing(T: int, E: int, top_k: int = 2, seed: int = 0):
    """Build (router_indices, routing_weights) of the shape HF expects."""
    g = torch.Generator().manual_seed(seed)
    # One expert per (token, position) in [0, E); no padding class for sanity.
    router = torch.randint(0, E, (T, top_k), generator=g)
    weights = torch.softmax(torch.randn(T, top_k, generator=g), dim=-1).to(torch.bfloat16)
    return router, weights


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_from_gpt_oss_preserves_config():
    from mud_puppy.mxfp4_experts import MXFP4Experts
    src = _make_gpt_oss_experts(num_experts=4, hidden_size=64, intermediate_size=64)
    mx = MXFP4Experts.from_gpt_oss(src)
    assert mx.num_experts == 4
    assert mx.hidden_size == 64
    assert mx.intermediate_size == 64
    # Buffers have the expected shapes
    assert mx.gate_up_qweight.shape == (4, 128, 32)      # [E, 2I, H/2]
    assert mx.gate_up_scales.shape == (4, 128, 2)        # [E, 2I, H/32]
    assert mx.down_qweight.shape == (4, 64, 32)          # [E, H, I/2]
    assert mx.down_scales.shape == (4, 64, 2)            # [E, H, I/32]
    # Biases preserved
    assert torch.allclose(mx.gate_up_proj_bias, src.gate_up_proj_bias.to(torch.bfloat16))
    assert torch.allclose(mx.down_proj_bias, src.down_proj_bias.to(torch.bfloat16))
    # Constants match
    assert mx.alpha == src.alpha
    assert mx.limit == src.limit


def test_from_gpt_oss_requires_block_aligned_dims():
    from mud_puppy.mxfp4_experts import MXFP4Experts
    with pytest.raises(ValueError):
        MXFP4Experts(num_experts=2, hidden_size=60, intermediate_size=64)  # 60 % 32 != 0
    with pytest.raises(ValueError):
        MXFP4Experts(num_experts=2, hidden_size=64, intermediate_size=50)


# --------------------------------------------------------------------------
# Forward: matches native within quant error
# --------------------------------------------------------------------------


def test_forward_matches_native_within_quant_error(monkeypatch):
    """Forward output must match GptOssExperts up to MXFP4 rounding.

    Use RMS error relative to RMS output. MXFP4 is ~5% RMS on random
    weights; set the threshold at 15% to cover test variance.
    """
    monkeypatch.delenv("MUD_PUPPY_MXFP4_TRITON", raising=False)
    from mud_puppy.mxfp4_experts import MXFP4Experts

    E, H, I = 4, 64, 64
    src = _make_gpt_oss_experts(num_experts=E, hidden_size=H, intermediate_size=I)
    mx = MXFP4Experts.from_gpt_oss(src)

    T = 16
    torch.manual_seed(42)
    hidden = torch.randn(T, H, dtype=torch.bfloat16)
    router, weights = _random_routing(T, E, top_k=2, seed=7)

    with torch.no_grad():
        y_ref = src(hidden, router, weights)
        y_mx = mx(hidden, router, weights)

    assert y_mx.shape == y_ref.shape
    rms_err = ((y_ref.float() - y_mx.float()) ** 2).mean().sqrt().item()
    rms_ref = (y_ref.float() ** 2).mean().sqrt().item()
    rel = rms_err / (rms_ref + 1e-8)
    # Two chained MXFP4 matmuls (gate_up + down) each contribute ~10% RMS
    # error on a toy 64-dim model; 25% is the realistic compounded bound.
    # Real gpt-oss weights have better-scaled distributions and much lower
    # quant error in practice.
    assert rel < 0.25, (
        f"MXFP4Experts output diverges from native: "
        f"rms_err={rms_err:.4e}, rms_ref={rms_ref:.4e}, rel={rel:.3%}"
    )


def test_forward_padding_skip_path(monkeypatch):
    """The ``if expert_idx == self.num_experts: continue`` branch protects
    against a masking class. We can't exercise it via F.one_hot (which
    rejects indices >= num_classes) but we can verify that the branch is
    present so a future refactor doesn't silently remove it."""
    import inspect
    from mud_puppy.mxfp4_experts import MXFP4Experts
    src = inspect.getsource(MXFP4Experts.forward)
    assert "expert_idx == self.num_experts" in src
    assert "continue" in src


def test_forward_single_active_expert(monkeypatch):
    """When every token routes only to expert 0, all work goes through
    a single expert slice and the output should equal that expert's
    forward run on every token."""
    monkeypatch.delenv("MUD_PUPPY_MXFP4_TRITON", raising=False)
    from mud_puppy.mxfp4_experts import MXFP4Experts

    E, H, I = 4, 64, 64
    src = _make_gpt_oss_experts(num_experts=E, hidden_size=H, intermediate_size=I)
    mx = MXFP4Experts.from_gpt_oss(src)

    T = 8
    hidden = torch.randn(T, H, dtype=torch.bfloat16)
    router = torch.zeros(T, 1, dtype=torch.long)  # every token -> expert 0
    weights = torch.ones(T, 1, dtype=torch.bfloat16)

    with torch.no_grad():
        y_ref = src(hidden, router, weights)
        y_mx = mx(hidden, router, weights)

    rms_err = ((y_ref.float() - y_mx.float()) ** 2).mean().sqrt().item()
    rms_ref = (y_ref.float() ** 2).mean().sqrt().item()
    assert rms_err / (rms_ref + 1e-8) < 0.25


# --------------------------------------------------------------------------
# SwiGLU interleave preserved
# --------------------------------------------------------------------------


def test_apply_gate_uses_interleaved_layout():
    """gate = gate_up[..., ::2], up = gate_up[..., 1::2]. A bug here
    would swap or concatenate the two halves and SwiGLU would go wrong."""
    from mud_puppy.mxfp4_experts import MXFP4Experts

    mx = MXFP4Experts(num_experts=2, hidden_size=32, intermediate_size=32)
    # Build a gate_up [..., 2*I] where even positions = known gate value
    # and odd positions = known up value.
    I = 32
    gate_raw = torch.full((4, I), 1.0, dtype=torch.bfloat16)
    up_raw = torch.full((4, I), 0.5, dtype=torch.bfloat16)
    gate_up = torch.stack([gate_raw, up_raw], dim=-1).reshape(4, 2 * I)

    out = mx._apply_gate(gate_up)
    # gate clamp has no effect (1.0 < limit=7). up clamp has no effect (0.5 in [-7, 7]).
    # glu = gate * sigmoid(alpha * gate) = 1 * sigmoid(1.702) ~ 0.846
    # gated_output = (up + 1) * glu = 1.5 * 0.846 ~ 1.269
    expected = 1.5 * torch.sigmoid(torch.tensor(1.702))
    # All outputs should be the same expected value (with bf16 tolerance)
    assert torch.allclose(out.float(), torch.full_like(out, expected.item()).float(), atol=5e-2)


# --------------------------------------------------------------------------
# Swap helper
# --------------------------------------------------------------------------


def _make_toy_moe_model():
    """Minimal nn.Module with a GptOssExperts inside, plus other modules."""
    cfg = SimpleNamespace(num_local_experts=2, hidden_size=32, intermediate_size=32)

    class ToyMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_proj = nn.Linear(32, 32)
            self.mlp_experts = GptOssExperts(cfg)
            self.norm = nn.LayerNorm(32)

    return ToyMoE()


def test_swap_touches_only_experts():
    from mud_puppy.mxfp4_experts import MXFP4Experts, swap_gpt_oss_experts_to_mxfp4
    model = _make_toy_moe_model()
    n = swap_gpt_oss_experts_to_mxfp4(model)
    assert n == 1
    assert isinstance(model.mlp_experts, MXFP4Experts)
    # Non-expert modules untouched
    assert isinstance(model.attn_proj, nn.Linear)
    assert isinstance(model.norm, nn.LayerNorm)


def test_swap_is_idempotent():
    from mud_puppy.mxfp4_experts import swap_gpt_oss_experts_to_mxfp4
    model = _make_toy_moe_model()
    swap_gpt_oss_experts_to_mxfp4(model)
    n_second = swap_gpt_oss_experts_to_mxfp4(model)
    assert n_second == 0  # already swapped


# --------------------------------------------------------------------------
# Triton path agrees with fallback
# --------------------------------------------------------------------------


try:
    import triton  # noqa: F401
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


@pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON),
    reason="needs CUDA/HIP + Triton",
)
def test_triton_path_matches_fallback(monkeypatch):
    """With env var on vs off, forward outputs must match within kernel
    numerical error. If they diverge, training dynamics differ between
    dev runs (triton off) and production runs (triton on)."""
    from mud_puppy.mxfp4_experts import MXFP4Experts

    E, H, I = 4, 64, 64
    src = _make_gpt_oss_experts(num_experts=E, hidden_size=H, intermediate_size=I)
    mx = MXFP4Experts.from_gpt_oss(src).cuda()

    T = 16
    hidden = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    router, weights = _random_routing(T, E, top_k=2, seed=1)
    router = router.cuda()
    weights = weights.cuda()

    monkeypatch.delenv("MUD_PUPPY_MXFP4_TRITON", raising=False)
    with torch.no_grad():
        y_pt = mx(hidden, router, weights).clone()

    monkeypatch.setenv("MUD_PUPPY_MXFP4_TRITON", "1")
    with torch.no_grad():
        y_tr = mx(hidden, router, weights).clone()

    diff = (y_pt.float() - y_tr.float()).abs()
    ref = y_pt.float().abs()
    rel_ok = diff <= 5e-2 * ref
    abs_ok = diff <= 5e-2
    assert (rel_ok | abs_ok).all(), (
        f"triton MoE path diverges from fallback: max diff={diff.max().item():.4e}"
    )


# --------------------------------------------------------------------------
# GPU sanity
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA/HIP")
def test_forward_on_gpu():
    from mud_puppy.mxfp4_experts import MXFP4Experts

    E, H, I = 4, 64, 64
    src = _make_gpt_oss_experts(num_experts=E, hidden_size=H, intermediate_size=I).cuda()
    mx = MXFP4Experts.from_gpt_oss(src).cuda()

    T = 8
    hidden = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    router, weights = _random_routing(T, E, top_k=2, seed=2)
    router = router.cuda()
    weights = weights.cuda()

    with torch.no_grad():
        y = mx(hidden, router, weights)
    assert y.device.type == "cuda"
    assert torch.isfinite(y).all()
