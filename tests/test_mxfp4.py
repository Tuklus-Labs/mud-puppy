"""Correctness tests for the MXFP4 primitives in ``mud_puppy.mxfp4``.

Every quantized-training run downstream depends on these invariants.
If any of them break, training silently drifts off the MXFP4 grid and
the "deploy without conversion" pitch fails.

What we lock in:
    * E2M1 levels are the canonical 8 positive values (0, 0.5, ..., 6).
    * E8M0 round-trip: encode(2^k) == k + 127 for k in [-127, 127].
    * After ``quantize_mxfp4``, every output element lies on some
      ``level * 2^exp`` grid point.
    * Quantization error is bounded by the local grid spacing.
    * Stochastic rounding is unbiased in expectation (within MC noise).
    * ``pack_mxfp4 -> unpack_mxfp4`` reproduces the round-to-nearest
      quantized tensor bit-exactly.
    * Padding edge cases: block_size not dividing numel.
"""

from __future__ import annotations

import math

import pytest
import torch

from mud_puppy.mxfp4 import (
    E2M1_MAX,
    E2M1_POSITIVE_VALUES,
    _e2m1_levels,
    _e8m0_decode,
    _e8m0_encode,
    pack_mxfp4,
    quantize_mxfp4,
    quantize_mxfp4_with_scale,
    unpack_mxfp4,
)


# ---------------------------------------------------------------------------
# E2M1 level set
# ---------------------------------------------------------------------------


def test_e2m1_positive_values_match_spec() -> None:
    # OCP MX spec v1.0: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
    assert E2M1_POSITIVE_VALUES == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    assert E2M1_MAX == 6.0


def test_e2m1_levels_sorted_and_symmetric_around_zero() -> None:
    levels = _e2m1_levels(torch.device("cpu"), torch.float32)
    assert levels.numel() == 15  # 7 negative + 0 + 7 positive (no double-zero)
    assert torch.all(levels[1:] > levels[:-1]), "levels must be strictly increasing"
    # Symmetric: levels[-i] == -levels[i-1] for i in 1..7 around the zero midpoint.
    mid = (levels.numel() - 1) // 2
    assert levels[mid].item() == 0.0
    for i in range(1, mid + 1):
        assert levels[mid + i].item() == -levels[mid - i].item()


# ---------------------------------------------------------------------------
# E8M0 scale encoder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [-100, -50, -1, 0, 1, 50, 100])
def test_e8m0_roundtrip_powers_of_two(k: int) -> None:
    scale = torch.tensor([2.0 ** k], dtype=torch.float32)
    byte = _e8m0_encode(scale)
    assert byte.item() == k + 127
    back = _e8m0_decode(byte, torch.float32)
    assert torch.allclose(back, scale, rtol=1e-6)


def test_e8m0_ceils_to_next_power_of_two() -> None:
    """Encoder uses ceil so the decoded scale is always >= input scale.

    This guarantees the MXFP4 grid covers the block's dynamic range
    without saturation, which is required for unbiased stochastic
    rounding of values near the block max.
    """
    # 1.0 is a power of 2 -> exact. 1.4 and 1.6 both ceil to 2.0 (byte 128).
    # 2.0 is exact (byte 128). 2.01 ceils to 4.0 (byte 129).
    scale = torch.tensor([1.0, 1.4, 1.6, 2.0, 2.01], dtype=torch.float32)
    byte = _e8m0_encode(scale)
    assert byte.tolist() == [127, 128, 128, 128, 129]


def test_e8m0_clamps_overflow() -> None:
    # Very large scale should saturate at byte 254 (= 2^127, ~1.7e38).
    huge = torch.tensor([1e40], dtype=torch.float32)
    byte = _e8m0_encode(huge)
    assert byte.item() == 254
    # Very small scale should floor to byte 0 (= 2^-127).
    tiny = torch.tensor([1e-40], dtype=torch.float32)
    byte = _e8m0_encode(tiny)
    assert byte.item() == 0


def test_e8m0_handles_zero_scale() -> None:
    zero = torch.tensor([0.0], dtype=torch.float32)
    byte = _e8m0_encode(zero)
    # Zero is clamped to 2^-127 (byte 0), not NaN'd.
    assert byte.item() == 0


# ---------------------------------------------------------------------------
# Core quantize: grid membership
# ---------------------------------------------------------------------------


def test_quantize_output_on_grid() -> None:
    """Every output value must be level * 2^exp for some level in E2M1 and exp in [-127, 127].

    Uses the with-scale API for ground-truth block scales; inferring scale
    from the quantized output is wrong when no block value rounds to ±6.
    """
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32) * 3.0
    q, scale_bytes = quantize_mxfp4_with_scale(x, block_size=32)
    scales = _e8m0_decode(scale_bytes, torch.float32).tolist()

    for block_idx, block in enumerate(q.reshape(-1, 32)):
        scale = scales[block_idx]
        # Each value, divided by scale, must be in the ±E2M1 set (or zero).
        for v in block.tolist():
            normalized = abs(v) / scale
            close = any(abs(normalized - lvl) < 1e-4 for lvl in E2M1_POSITIVE_VALUES)
            assert close, f"block {block_idx}: {v} / {scale} = {normalized} not in E2M1 set"


def test_quantize_preserves_zero_blocks() -> None:
    x = torch.zeros(64, dtype=torch.float32)
    q = quantize_mxfp4(x, block_size=32)
    assert torch.all(q == 0.0)


def test_quantize_bounded_error() -> None:
    """Quantization error bounded by the largest E2M1 gap times the block scale,
    plus any saturation beyond ±6*scale.

    Largest gap is 2 (between 4 and 6), so post-scaling max interior error
    is 1.0 * scale. Saturation can add up to (|x_max| - 6*scale) on top.
    """
    torch.manual_seed(1)
    x = torch.randn(4096, dtype=torch.float32) * 2.5
    q, scale_bytes = quantize_mxfp4_with_scale(x, block_size=32)
    scales = _e8m0_decode(scale_bytes, torch.float32).tolist()

    err = (x - q).abs().reshape(-1, 32)
    x_blocks = x.reshape(-1, 32)

    for i in range(err.shape[0]):
        scale = scales[i]
        interior_bound = scale          # max interior gap is 2*scale; halfway = scale
        sat_overflow = (x_blocks[i].abs() - E2M1_MAX * scale).clamp_min(0).max().item()
        bound = interior_bound + sat_overflow + 1e-5
        assert err[i].max().item() <= bound, (
            f"block {i}: err {err[i].max():.4e} > bound {bound:.4e} "
            f"(scale={scale}, sat_overflow={sat_overflow})"
        )


def test_quantize_preserves_shape_and_dtype() -> None:
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        x = torch.randn(3, 128, dtype=dtype)
        q = quantize_mxfp4(x, block_size=32)
        assert q.shape == x.shape
        assert q.dtype == dtype


def test_quantize_handles_non_divisible_numel() -> None:
    # numel=100 -> 4 blocks of 32 = 128 padded.
    x = torch.randn(10, 10, dtype=torch.float32)
    q = quantize_mxfp4(x, block_size=32)
    assert q.shape == (10, 10)


# ---------------------------------------------------------------------------
# Stochastic rounding: unbiased
# ---------------------------------------------------------------------------


def test_stochastic_rounding_is_unbiased() -> None:
    """E[quantize_stochastic(x)] ≈ x over many trials within MC noise.

    Fails if rounding is biased (which would systematically drift weights
    during stochastic-rounding optimizer steps).
    """
    torch.manual_seed(42)
    # Pick values that sit deliberately between grid points.
    x = torch.tensor([0.1, 0.25, 0.37, 0.75, 1.3, 2.5, 3.5, 5.0], dtype=torch.float32)
    # 32-element block so we only exercise one scale.
    x_full = x.repeat(4)  # 32 elements
    assert x_full.numel() == 32

    n_trials = 2000
    accum = torch.zeros_like(x_full)
    for _ in range(n_trials):
        q = quantize_mxfp4(x_full, block_size=32, stochastic=True)
        accum += q
    mean = accum / n_trials

    # MC standard error is ~ scale / sqrt(n_trials). Block scale here is
    # 5.0/6.0 ≈ 1, rounded down to 2^-1 = 0.5 → wait, rounded to *nearest*
    # power of 2 of 0.833 = 1.0 (2^0). So scale = 1.0, values up to 6*1=6.
    # Typical grid gap ~0.5. SE ≈ 0.5 / sqrt(2000) ≈ 0.01.
    diff = (mean - x_full).abs().max().item()
    assert diff < 0.05, f"stochastic rounding biased: max|E[q]-x| = {diff}"


def test_stochastic_outputs_are_on_grid() -> None:
    """Every stochastic output lies on some level * 2^exp grid point.

    We recover the block scale by running the deterministic with-scale
    API on the same input (both paths use the same block-max based scale).
    """
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32)
    q_stoch = quantize_mxfp4(x, block_size=32, stochastic=True)
    _, scale_bytes = quantize_mxfp4_with_scale(x, block_size=32)
    scales = _e8m0_decode(scale_bytes, torch.float32).tolist()

    for block_idx, block in enumerate(q_stoch.reshape(-1, 32)):
        scale = scales[block_idx]
        for v in block.tolist():
            normalized = abs(v) / scale
            close = any(abs(normalized - lvl) < 1e-4 for lvl in E2M1_POSITIVE_VALUES)
            assert close, f"block {block_idx}: {v}/{scale}={normalized} off grid"


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------


def test_pack_unpack_bit_exact_roundtrip() -> None:
    """pack then unpack must reproduce the round-to-nearest quantized tensor exactly.

    This is a hard equality check because both paths go through the same
    grid lookup; any lossy step in between breaks the "deploy without
    reconversion" promise.
    """
    torch.manual_seed(5)
    x = torch.randn(64, 32, dtype=torch.float32)

    q_ref = quantize_mxfp4(x, block_size=32)  # full fake-quant result
    packed, scales = pack_mxfp4(x, block_size=32)
    q_recovered = unpack_mxfp4(packed, scales, x.shape, block_size=32, dtype=torch.float32)
    assert torch.equal(q_recovered, q_ref), (
        "pack/unpack round-trip diverges from quantize_mxfp4 RTN path"
    )


def test_pack_compression_ratio() -> None:
    """Packed tensor should be ~1/4 the size of bf16 source (plus small scale overhead).

    4-bit packed = 0.5 bytes/element + 1 scale byte per 32 elements
    = 0.5 + 1/32 ≈ 0.531 bytes/element.
    bf16 = 2 bytes/element. Ratio = 0.531 / 2 = 0.266, or ~3.77x compression.
    """
    x = torch.randn(8192, dtype=torch.bfloat16)
    packed, scales = pack_mxfp4(x, block_size=32)
    source_bytes = x.numel() * 2  # bf16
    quant_bytes = packed.numel() + scales.numel()
    ratio = source_bytes / quant_bytes
    assert 3.5 < ratio < 4.0, f"compression ratio {ratio} outside expected band"


def test_pack_shape_preserved_through_roundtrip() -> None:
    for shape in [(31,), (32,), (33,), (100,), (7, 13), (3, 8, 16)]:
        x = torch.randn(*shape, dtype=torch.float32)
        packed, scales = pack_mxfp4(x, block_size=32)
        recovered = unpack_mxfp4(packed, scales, x.shape, block_size=32, dtype=torch.float32)
        assert recovered.shape == x.shape


# ---------------------------------------------------------------------------
# Quantize-with-scale API
# ---------------------------------------------------------------------------


def test_quantize_with_scale_matches_quantize_rtn() -> None:
    torch.manual_seed(9)
    x = torch.randn(128, dtype=torch.float32)
    q = quantize_mxfp4(x, block_size=32)
    q2, scale_bytes = quantize_mxfp4_with_scale(x, block_size=32)
    assert torch.equal(q, q2)
    assert scale_bytes.shape == (4,)  # 128 / 32
    assert scale_bytes.dtype == torch.int32
    assert ((scale_bytes >= 0) & (scale_bytes <= 254)).all()


# ---------------------------------------------------------------------------
# Smoke: works on CUDA if available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA/HIP")
def test_quantize_on_gpu() -> None:
    x = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    q, scale_bytes = quantize_mxfp4_with_scale(x, block_size=32)
    assert q.device == x.device
    assert q.dtype == x.dtype
    # Ground-truth block scale from the with-scale API, not inferred from output.
    scale = _e8m0_decode(scale_bytes, torch.float32)[0].item()
    b = q.reshape(-1, 32)[0].float()
    for v in b.tolist():
        normalized = abs(v) / scale
        assert any(abs(normalized - lvl) < 1e-2 for lvl in E2M1_POSITIVE_VALUES)
