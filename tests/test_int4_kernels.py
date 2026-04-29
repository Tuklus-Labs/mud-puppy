"""Correctness tests for the Triton INT4 fused matmul kernels.

Validates forward and backward numerics against the pure-PyTorch
reference path in ``bnb_rocm._dequantize_packed + F.linear``. These
kernels take the same packed-INT4 storage layout as the existing
``Linear4bit``; if they disagree with the reference, training
silently diverges. Every test here is a correctness gate.

Tolerances:
    * bf16 path: max abs diff <= 5e-2 (bf16 has ~3 decimal digits).
    * fp32 accumulator intermediate: <= 1e-5.

Run only on GPUs with Triton. Skipped on CPU.
"""

from __future__ import annotations

import pytest
import torch

try:
    import triton  # noqa: F401

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

from mud_puppy.bnb_rocm import Linear4bit, _dequantize_packed


requires_gpu_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON),
    reason="needs CUDA/HIP + Triton",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_linear4bit(in_features: int, out_features: int, dtype=torch.bfloat16, seed=0):
    """Build a Linear4bit from a random nn.Linear, return (module, original_linear)."""
    torch.manual_seed(seed)
    lin = torch.nn.Linear(in_features, out_features, bias=False)
    q = Linear4bit(lin, dtype=dtype)
    return q, lin


def _reference_forward(x, q: Linear4bit) -> torch.Tensor:
    """Pure-PyTorch forward using bnb_rocm's dequantize + F.linear path."""
    w = _dequantize_packed(q.qweight, q.scale, q.out_features, q.in_features, q.dtype)
    return torch.nn.functional.linear(x.to(q.dtype), w, q.bias)


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


@requires_gpu_triton
@pytest.mark.parametrize("shape", [
    # (batch, in, out) covering decode-ish, training-batch-ish, and an
    # attention proj shape from llama/mistral.
    (4, 128, 256),
    (16, 1024, 1024),
    (8, 4096, 11008),   # llama-7b MLP up_proj
    (1, 4096, 4096),    # batch=1 decode style
])
def test_triton_forward_matches_reference(shape):
    from mud_puppy.int4_kernels import triton_int4_matmul

    M, K, N = shape
    q, _ = _build_linear4bit(K, N, dtype=torch.bfloat16, seed=42)
    q = q.cuda()
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    ref = _reference_forward(x, q)
    out = triton_int4_matmul(x, q.qweight, q.scale)

    assert out.shape == ref.shape
    diff = (out.float() - ref.float()).abs()
    # bf16 rounding + fp32 accumulator differences. The naive path also
    # uses fp32 intermediates; main source of drift is the order of the
    # K-axis reduction across tiles.
    max_diff = diff.max().item()
    assert max_diff < 5e-2, (
        f"triton forward diverges from reference: max|Δ|={max_diff:.4e}, "
        f"shape={shape}"
    )


@requires_gpu_triton
def test_triton_forward_handles_odd_K():
    """K = 257 forces a padded nibble on the last byte. Verify correctness."""
    from mud_puppy.int4_kernels import triton_int4_matmul

    q, _ = _build_linear4bit(257, 128, dtype=torch.bfloat16, seed=7)
    q = q.cuda()
    x = torch.randn(4, 257, device="cuda", dtype=torch.bfloat16)
    ref = _reference_forward(x, q)
    out = triton_int4_matmul(x, q.qweight, q.scale)
    max_diff = (out.float() - ref.float()).abs().max().item()
    assert max_diff < 5e-2, f"odd-K forward diverges: max|Δ|={max_diff:.4e}"


@requires_gpu_triton
def test_triton_forward_no_nan():
    """Output must be fully finite for reasonable inputs."""
    from mud_puppy.int4_kernels import triton_int4_matmul

    q, _ = _build_linear4bit(512, 512, dtype=torch.bfloat16, seed=1)
    q = q.cuda()
    x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    out = triton_int4_matmul(x, q.qweight, q.scale)
    assert torch.isfinite(out).all(), "triton forward produced non-finite values"


# ---------------------------------------------------------------------------
# Backward correctness
# ---------------------------------------------------------------------------


@requires_gpu_triton
@pytest.mark.parametrize("shape", [
    (4, 128, 256),
    (16, 1024, 1024),
    (8, 4096, 11008),
])
def test_triton_backward_grad_input_matches_reference(shape):
    from mud_puppy.int4_kernels import triton_int4_grad_input

    M, K, N = shape
    q, _ = _build_linear4bit(K, N, dtype=torch.bfloat16, seed=42)
    q = q.cuda()

    # Reference grad_input = grad_output @ W
    w = _dequantize_packed(q.qweight, q.scale, q.out_features, q.in_features, q.dtype)
    grad_output = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    ref_gi = grad_output.to(torch.float32) @ w.to(torch.float32)

    gi = triton_int4_grad_input(grad_output, q.qweight, q.scale, in_features=K)

    assert gi.shape == (M, K)
    diff = (gi.float() - ref_gi).abs()
    max_diff = diff.max().item()
    assert max_diff < 5e-2, (
        f"triton backward grad_input diverges: max|Δ|={max_diff:.4e}, shape={shape}"
    )


# ---------------------------------------------------------------------------
# End-to-end: forward + backward through a tiny model
# ---------------------------------------------------------------------------


@requires_gpu_triton
def test_forward_plus_backward_end_to_end():
    """Full forward + backward through Linear4bit with triton forward and
    backward produces gradients that match the pure-PyTorch path to within
    bf16 tolerance. If this fails training will diverge."""
    from mud_puppy.int4_kernels import triton_int4_grad_input, triton_int4_matmul

    K, N, M = 512, 768, 8
    q, _ = _build_linear4bit(K, N, dtype=torch.bfloat16, seed=99)
    q = q.cuda()
    x_naive = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_triton = x_naive.detach().clone().requires_grad_(True)

    # Naive forward + backward
    out_naive = _reference_forward(x_naive, q)
    loss_naive = out_naive.sum()
    loss_naive.backward()
    gi_naive = x_naive.grad.clone()

    # Triton forward + backward (hand-composed; the autograd Function
    # wrapping is tested separately via the integration flag)
    out_triton = triton_int4_matmul(x_triton, q.qweight, q.scale)
    # For the backward portion we mirror what the autograd Function would do:
    grad_out = torch.ones_like(out_triton)
    gi_triton = triton_int4_grad_input(grad_out, q.qweight, q.scale, in_features=K)

    out_diff = (out_triton.float() - out_naive.float()).abs().max().item()
    assert out_diff < 5e-2, f"forward diverges: {out_diff:.4e}"

    gi_diff = (gi_triton.float() - gi_naive.float()).abs().max().item()
    assert gi_diff < 5e-2, f"grad_input diverges: {gi_diff:.4e}"


# ---------------------------------------------------------------------------
# Feature flag plumbing
# ---------------------------------------------------------------------------


def test_is_enabled_default_off(monkeypatch):
    """Kernel must be opt-in by default so it doesn't auto-enable on untested shapes."""
    from mud_puppy.int4_kernels import _is_enabled

    monkeypatch.delenv("MUD_PUPPY_INT4_TRITON", raising=False)
    assert _is_enabled() is False


def test_is_enabled_respects_env(monkeypatch):
    from mud_puppy.int4_kernels import _is_enabled

    # "1" turns it on only if Triton is available; we can't force Triton on
    # in the test process, so just exercise the parser.
    monkeypatch.setenv("MUD_PUPPY_INT4_TRITON", "1")
    # Result depends on TRITON_AVAILABLE; we don't assert True here because
    # CI may not have Triton. We do assert False on explicit off:
    monkeypatch.setenv("MUD_PUPPY_INT4_TRITON", "0")
    assert _is_enabled() is False
    monkeypatch.setenv("MUD_PUPPY_INT4_TRITON", "")
    assert _is_enabled() is False


# ---------------------------------------------------------------------------
# Linear4bit integration: triton path vs pytorch path through autograd
# ---------------------------------------------------------------------------


@requires_gpu_triton
def test_linear4bit_triton_forward_matches_pytorch(monkeypatch):
    """With MUD_PUPPY_INT4_TRITON on/off, the forward output must match.

    Catches bugs where the kernel plumbing is miswired (wrong tensor
    passed, flipped transpose, bias ordering) which would show up as a
    small but systematic output difference.
    """
    from mud_puppy.bnb_rocm import Linear4bit

    torch.manual_seed(5)
    lin = torch.nn.Linear(1024, 2048, bias=True).cuda()
    q = Linear4bit(lin, dtype=torch.bfloat16).cuda()

    x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)

    monkeypatch.delenv("MUD_PUPPY_INT4_TRITON", raising=False)
    out_pytorch = q(x).detach().clone()

    monkeypatch.setenv("MUD_PUPPY_INT4_TRITON", "1")
    out_triton = q(x).detach().clone()

    max_diff = (out_triton.float() - out_pytorch.float()).abs().max().item()
    assert max_diff < 5e-2, (
        f"Linear4bit triton/pytorch paths diverge: max|Δ|={max_diff:.4e}"
    )


# ---------------------------------------------------------------------------
# anvil-train config override
# ---------------------------------------------------------------------------


# A config from _FWD_CONFIGS that is known to compile and work on RDNA3
# shapes. The bypass path lives in the wrapper's ``if config is None``
# branch; this test makes sure the non-None branch reaches the same
# numerics as the autotuned path within the existing 5e-2 abs gate.
_ANVIL_INT4_CFG = {
    "BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64,
    "GROUP_M": 8, "num_warps": 4, "num_stages": 3,
    "speedup_vs_baseline": 1.27,
    "profiled_us": 19.0,
}


@requires_gpu_triton
def test_int4_forward_with_anvil_config():
    from mud_puppy.int4_kernels import triton_int4_matmul

    M, K, N = 16, 1024, 1024
    q, _ = _build_linear4bit(K, N, dtype=torch.bfloat16, seed=42)
    q = q.cuda()
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    ref = _reference_forward(x, q)
    out = triton_int4_matmul(x, q.qweight, q.scale, config=_ANVIL_INT4_CFG)

    assert out.shape == ref.shape
    diff = (out.float() - ref.float()).abs()
    max_diff = diff.max().item()
    assert max_diff < 5e-2, (
        f"int4 forward (anvil cfg) diverges: max|Δ|={max_diff:.4e}"
    )


@requires_gpu_triton
def test_int4_grad_input_with_anvil_config():
    from mud_puppy.int4_kernels import triton_int4_grad_input

    M, K, N = 16, 1024, 1024
    q, _ = _build_linear4bit(K, N, dtype=torch.bfloat16, seed=42)
    q = q.cuda()

    w = _dequantize_packed(q.qweight, q.scale, q.out_features, q.in_features, q.dtype)
    grad_output = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    ref_gi = grad_output.to(torch.float32) @ w.to(torch.float32)

    gi = triton_int4_grad_input(
        grad_output, q.qweight, q.scale, in_features=K, config=_ANVIL_INT4_CFG,
    )
    assert gi.shape == (M, K)
    max_diff = (gi.float() - ref_gi).abs().max().item()
    assert max_diff < 5e-2, (
        f"int4 grad_input (anvil cfg) diverges: max|Δ|={max_diff:.4e}"
    )


@requires_gpu_triton
def test_linear4bit_triton_backward_matches_pytorch(monkeypatch):
    """Gradients through Linear4bit must match between triton and pytorch paths."""
    from mud_puppy.bnb_rocm import Linear4bit

    torch.manual_seed(7)
    lin = torch.nn.Linear(512, 1024, bias=False).cuda()
    q = Linear4bit(lin, dtype=torch.bfloat16).cuda()

    # Same input for both runs -- we want to measure kernel disagreement,
    # not input variance.
    x_base = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)

    def run(use_triton: bool):
        if use_triton:
            monkeypatch.setenv("MUD_PUPPY_INT4_TRITON", "1")
        else:
            monkeypatch.delenv("MUD_PUPPY_INT4_TRITON", raising=False)
        x = x_base.detach().clone().requires_grad_(True)
        out = q(x)
        out.sum().backward()
        return out.detach().clone(), x.grad.detach().clone()

    out_pt, gi_pt = run(use_triton=False)
    out_tr, gi_tr = run(use_triton=True)

    out_diff = (out_tr.float() - out_pt.float()).abs().max().item()
    gi_diff = (gi_tr.float() - gi_pt.float()).abs().max().item()
    assert out_diff < 5e-2, f"fwd diverges: {out_diff:.4e}"
    assert gi_diff < 5e-2, f"bwd diverges: {gi_diff:.4e}"
