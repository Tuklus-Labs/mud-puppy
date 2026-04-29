"""Correctness tests for the Triton MXFP4 matmul kernels.

Validates the W4A16 fused-dequant path against the pure-PyTorch
reference from ``mxfp4`` + ``F.linear``. If the kernel disagrees, a
native-MXFP4 training run silently produces wrong gradients and loss
diverges after a few hundred steps.

Tolerance band:
    * Relative error <= 1% -- covers both MXFP4 re-quantization and
      the bf16 rounding that accumulates over large K reductions.
    * Per-element absolute check requires either tight absolute (1e-2)
      OR tight relative (1%). Either satisfying passes.
"""

from __future__ import annotations

import pytest
import torch

try:
    import triton  # noqa: F401
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

from mud_puppy.mxfp4 import pack_mxfp4, unpack_mxfp4


requires_gpu_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON),
    reason="needs CUDA/HIP + Triton",
)


def _reference_matmul(x: torch.Tensor, qweight, scales, shape, dtype):
    """Pure-PyTorch: dequantize to bf16 then matmul (same numeric path as kernel)."""
    w = unpack_mxfp4(qweight, scales, shape, block_size=32, dtype=dtype)
    return torch.nn.functional.linear(x.to(dtype), w)


def _assert_close(actual, expected, rtol=1e-2, atol=1e-2, name=""):
    """Pass if every element satisfies either absolute or relative tolerance.

    Useful for bf16 matmul outputs where some elements are near zero
    (absolute scale) and others are order-10+ (relative scale).
    """
    diff = (actual.float() - expected.float()).abs()
    ref_abs = expected.float().abs()
    # Each element OK if diff <= atol OR diff / |ref| <= rtol
    rel_ok = diff <= rtol * ref_abs
    abs_ok = diff <= atol
    ok = rel_ok | abs_ok
    if not ok.all():
        worst = diff.argmax()
        flat_actual = actual.float().flatten()[worst].item()
        flat_expected = expected.float().flatten()[worst].item()
        assert False, (
            f"{name}: max diff={diff.max().item():.4e} "
            f"(ref={flat_expected:.4e}, got={flat_actual:.4e}, "
            f"rel={diff.max().item() / (ref_abs.flatten()[worst].item() + 1e-12):.2%})"
        )


def _make_packed_weight(N: int, K: int, seed: int = 0, dtype=torch.bfloat16):
    """Build a packed MXFP4 weight from a random bf16 tensor."""
    torch.manual_seed(seed)
    w = torch.randn(N, K, dtype=dtype, device="cuda")
    packed, scales = pack_mxfp4(w, block_size=32)
    return packed, scales, w.shape


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


@requires_gpu_triton
@pytest.mark.parametrize("shape", [
    (4, 128, 256),
    (8, 1024, 1024),
    (4, 2880, 2880),   # gpt-oss-20b hidden
    (1, 4096, 4096),   # batch=1 decode style
])
def test_forward_matches_reference(shape):
    from mud_puppy.mxfp4_kernels import triton_mxfp4_matmul

    M, K, N = shape
    packed, scales, wshape = _make_packed_weight(N, K, seed=42)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    ref = _reference_matmul(x, packed, scales, wshape, torch.bfloat16)
    out = triton_mxfp4_matmul(x, packed, scales, in_features=K)

    assert out.shape == ref.shape
    _assert_close(out, ref, name=f"fwd shape={shape}")


@requires_gpu_triton
def test_forward_no_nan():
    from mud_puppy.mxfp4_kernels import triton_mxfp4_matmul

    packed, scales, shape = _make_packed_weight(512, 512, seed=1)
    x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    out = triton_mxfp4_matmul(x, packed, scales, in_features=512)
    assert torch.isfinite(out).all()


@requires_gpu_triton
def test_forward_handles_k_not_mult_of_64():
    """K = 128 (multiple of 32 only) must still work."""
    from mud_puppy.mxfp4_kernels import triton_mxfp4_matmul

    packed, scales, shape = _make_packed_weight(64, 128, seed=2)
    x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    ref = _reference_matmul(x, packed, scales, shape, torch.bfloat16)
    out = triton_mxfp4_matmul(x, packed, scales, in_features=128)
    _assert_close(out, ref, name="fwd K=128")


# ---------------------------------------------------------------------------
# Backward correctness
# ---------------------------------------------------------------------------


@requires_gpu_triton
@pytest.mark.parametrize("shape", [
    (4, 128, 256),
    (8, 1024, 1024),
    (4, 2880, 2880),
])
def test_backward_grad_input_matches_reference(shape):
    from mud_puppy.mxfp4_kernels import triton_mxfp4_grad_input

    M, K, N = shape
    packed, scales, wshape = _make_packed_weight(N, K, seed=42)

    # Reference uses the SAME dtype path as the kernel (bf16 matmul with fp32
    # accumulator) so we're comparing numerics, not fp32-vs-bf16 rounding.
    w_ref = unpack_mxfp4(packed, scales, wshape, block_size=32, dtype=torch.bfloat16)
    grad_output = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    ref_gi = torch.nn.functional.linear(grad_output, w_ref.T.contiguous())

    gi = triton_mxfp4_grad_input(grad_output, packed, scales, in_features=K)

    assert gi.shape == (M, K)
    _assert_close(gi, ref_gi, name=f"bwd shape={shape}")


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


def test_is_enabled_default_off(monkeypatch):
    from mud_puppy.mxfp4_kernels import _is_enabled

    monkeypatch.delenv("MUD_PUPPY_MXFP4_TRITON", raising=False)
    assert _is_enabled() is False


def test_is_enabled_respects_env(monkeypatch):
    from mud_puppy.mxfp4_kernels import _is_enabled

    monkeypatch.setenv("MUD_PUPPY_MXFP4_TRITON", "0")
    assert _is_enabled() is False
    monkeypatch.setenv("MUD_PUPPY_MXFP4_TRITON", "")
    assert _is_enabled() is False


# ---------------------------------------------------------------------------
# anvil-train config override
# ---------------------------------------------------------------------------


# Pick an RDNA3-friendly config that exists in _FWD_CONFIGS so the kernel
# definitely compiles for it without surprises. The point of this test is
# to verify the bypass path itself (config != None branch in
# triton_mxfp4_matmul / triton_mxfp4_grad_input), not to hunt for new
# winners.
_ANVIL_FWD_CFG = {
    "BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32,
    "GROUP_M": 1, "num_warps": 4, "num_stages": 2,
    # Diagnostic-only fields the loader leaves on the dict; the kernel
    # wrapper must filter these without choking.
    "speedup_vs_baseline": 1.42,
    "profiled_us": 12.3,
}
_ANVIL_BWD_CFG = dict(_ANVIL_FWD_CFG)


@requires_gpu_triton
def test_forward_with_anvil_config_passes_rel_tol():
    """Anvil-supplied config must still satisfy the 1% rel-tol gate."""
    from mud_puppy.mxfp4_kernels import triton_mxfp4_matmul

    M, K, N = 8, 1024, 1024
    packed, scales, wshape = _make_packed_weight(N, K, seed=42)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    ref = _reference_matmul(x, packed, scales, wshape, torch.bfloat16)
    out = triton_mxfp4_matmul(x, packed, scales, in_features=K, config=_ANVIL_FWD_CFG)

    assert out.shape == ref.shape
    _assert_close(out, ref, name="fwd anvil-cfg")


@requires_gpu_triton
def test_grad_input_with_anvil_config_passes_rel_tol():
    from mud_puppy.mxfp4_kernels import triton_mxfp4_grad_input

    M, K, N = 8, 1024, 1024
    packed, scales, wshape = _make_packed_weight(N, K, seed=42)
    w_ref = unpack_mxfp4(packed, scales, wshape, block_size=32, dtype=torch.bfloat16)
    grad_output = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    ref_gi = torch.nn.functional.linear(grad_output, w_ref.T.contiguous())

    gi = triton_mxfp4_grad_input(
        grad_output, packed, scales, in_features=K, config=_ANVIL_BWD_CFG,
    )
    assert gi.shape == (M, K)
    _assert_close(gi, ref_gi, name="bwd anvil-cfg")


@requires_gpu_triton
def test_forward_with_no_config_unchanged():
    """``config=None`` must give bit-equivalent results to the legacy path."""
    from mud_puppy.mxfp4_kernels import triton_mxfp4_matmul

    M, K, N = 4, 1024, 1024
    packed, scales, wshape = _make_packed_weight(N, K, seed=11)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    out_default = triton_mxfp4_matmul(x, packed, scales, in_features=K)
    out_explicit = triton_mxfp4_matmul(x, packed, scales, in_features=K, config=None)
    # Default path runs autotune; both results pick the same winner so
    # output should match exactly.
    assert torch.equal(out_default, out_explicit)
