"""MXFP4 (Microscaling FP4) primitives for quantized training.

Implements the OCP Microscaling format at 4-bit precision. This is the
real MXFP4, not the misleadingly-named block-wise INT4 in
``mxfp4_rocm.py`` (which uses ``/7.0`` scaling and is really symmetric
INT4 with per-block fp16 scale).

Format (OCP MX spec v1.0):
    * Block size: 32 elements share one scale.
    * Element: E2M1 (1 sign bit, 2 exponent bits, 1 mantissa bit). The
      16 representable values are

          ±{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

    * Scale: E8M0 (8 bits, exponent only, bias 127). Each scale byte S
      represents 2^(S - 127). Range is 2^-127 .. 2^128, inclusive of 0.
      S = 255 is NaN and skipped; we clamp to [0, 254].

Why MXFP4 over symmetric INT4:
    * Wider dynamic range inside each block (values span 2^6.5 vs 2^3).
    * Outliers: a single 6x value doesn't crush the quantization grid for
      the other 31 elements (shared *exponent* rather than a linear max).
    * Hardware path: NVIDIA B200, AMD MI350/MI400 ship native MXFP4
      matmul; training on the same grid avoids deploy-time drift.

This module is the shared substrate for:
    * ``mxfp4_train.MXFP4QATLinear`` -- fake-quant forward for
      quantization-aware training with bf16 master weights.
    * Native packed MXFP4 storage (phase 4).

Correctness invariants:
    * ``dequantize(quantize_rtn(w))[i] in E2M1_VALUES * 2^e`` for every
      element ``i`` and some e in [-127, 128].
    * ``E[dequantize(quantize_stochastic(w))] == w`` (unbiased) in the
      limit, up to block-scale saturation effects.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# E2M1 value set
# ---------------------------------------------------------------------------

# The 8 non-negative E2M1 values, in increasing order.
# Computed as: (sign) * (1 + m/2) * 2^(e - 1) for e in {0,1,2,3}, m in {0,1};
# with the e=0 subnormals giving {0, 0.5}.
E2M1_POSITIVE_VALUES: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Max magnitude in E2M1 -- used to size the block scale.
E2M1_MAX: float = 6.0


def _e2m1_levels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the 16 E2M1 values sorted ascending as a 1D tensor.

    Cached per (device, dtype) by ``torch`` -- cheap to call repeatedly.
    """
    pos = torch.tensor(E2M1_POSITIVE_VALUES, dtype=dtype, device=device)
    # Negatives: reversed so the full tensor is strictly increasing from
    # -6 to +6 (with -0 and +0 collapsed since -0 == 0 in fp arithmetic).
    neg = -pos.flip(0)
    # Drop the duplicate zero so thresholds aren't double-counted.
    # Final layout: [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
    return torch.cat([neg[:-1], pos])


# ---------------------------------------------------------------------------
# E8M0 scale helpers
# ---------------------------------------------------------------------------


def _e8m0_encode(scale: torch.Tensor) -> torch.Tensor:
    """Encode a positive fp32 scale to E8M0 (power of 2 >= input).

    E8M0 stores an 8-bit exponent with bias 127. Callers pass
    ``block_max / 6`` and we want the smallest power of 2 >= that
    value -- i.e. ``ceil(log2(scale))`` -- so that after dividing the
    block by the decoded scale, no element lies outside ``[-6, 6]`` and
    thus no element saturates. Round-to-nearest would sometimes pick
    a scale smaller than block_max/6 and silently clip values near the
    block's dynamic-range edge, biasing stochastic rounding.

    Steps:
      1. Clean non-finite inputs: NaN -> 0 (byte 0), Inf -> 2^127 (byte 254).
         Without this, log2(inf) produces a CUDA int32 wrap to 0.
      2. Clamp to the smallest representable positive scale (2^-127).
      3. ``ceil(log2(scale))``.
      4. Clamp exponent to [-127, 127] (byte 0..254; 255 reserved NaN).

    Returns int32 in [0, 254].
    """
    TWO_POW_NEG_127 = 5.877471754111438e-39
    TWO_POW_127 = 1.7014118346046923e38
    scale = torch.where(torch.isnan(scale), torch.zeros_like(scale), scale)
    scale = torch.where(torch.isinf(scale), torch.full_like(scale, TWO_POW_127), scale)
    safe = scale.clamp_min(TWO_POW_NEG_127)
    exp = torch.log2(safe).ceil().to(torch.int32)
    exp = exp.clamp(min=-127, max=127)
    return exp + 127  # byte value


def _e8m0_decode(byte: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Decode an E8M0 byte (0..254) back to its scale value = 2^(byte - 127)."""
    exp = byte.to(torch.float32) - 127.0
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=byte.device), exp)
    return scale.to(dtype)


# ---------------------------------------------------------------------------
# Block-wise MXFP4 quantize / dequantize
# ---------------------------------------------------------------------------


def _reshape_to_blocks(x: torch.Tensor, block_size: int = 32) -> Tuple[torch.Tensor, torch.Size, int]:
    """Flatten x to (num_blocks, block_size), padding with zeros if needed.

    Returns (blocks, original_shape, padding_count). The padding is the
    number of zero elements appended at the end of the flattened tensor
    to make its length a multiple of ``block_size``.
    """
    original_shape = x.shape
    flat = x.reshape(-1)
    n = flat.numel()
    pad = (-n) % block_size  # non-negative pad count
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, block_size)
    return blocks, original_shape, pad


def _unblock(blocks: torch.Tensor, original_shape: torch.Size, pad: int) -> torch.Tensor:
    flat = blocks.reshape(-1)
    if pad:
        flat = flat[:-pad]
    return flat.reshape(original_shape)


def _round_to_nearest_level(x: torch.Tensor) -> torch.Tensor:
    """Round each element of x to the nearest E2M1 value.

    x is assumed to already be scaled so that its magnitudes lie in
    [0, 6.0] (anything beyond saturates to ±6). Returns values on the
    E2M1 grid in the same dtype as x.
    """
    levels = _e2m1_levels(x.device, x.dtype)
    # Use torch.bucketize to find insertion points, then pick the closer
    # of the two neighbors.
    idx = torch.bucketize(x, levels)
    idx = idx.clamp(min=1, max=len(levels) - 1)
    lo = levels[idx - 1]
    hi = levels[idx]
    pick_hi = (hi - x).abs() < (x - lo).abs()
    return torch.where(pick_hi, hi, lo)


def _round_stochastic_level(x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Round each element to an adjacent E2M1 level with probability
    proportional to how far it sits from each neighbor.

    If x is exactly between two levels, P(up) = 0.5. If x equals the
    lower level, P(up) = 0. Unbiased: E[round(x)] = x in the limit.
    """
    levels = _e2m1_levels(x.device, x.dtype)
    idx = torch.bucketize(x, levels)
    idx = idx.clamp(min=1, max=len(levels) - 1)
    lo = levels[idx - 1]
    hi = levels[idx]
    gap = (hi - lo).clamp_min(1e-30)  # guard against equal neighbors (shouldn't happen)
    frac_up = ((x - lo) / gap).clamp(0.0, 1.0)
    rnd = torch.rand_like(x) if generator is None else torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    pick_hi = rnd < frac_up
    return torch.where(pick_hi, hi, lo)


def quantize_mxfp4(
    x: torch.Tensor,
    block_size: int = 32,
    stochastic: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Fake-quantize a tensor to MXFP4 and return the dequantized result.

    The return value has the same shape and dtype as ``x``, but every
    element lies on an MXFP4 grid point: ``v = level * 2^(block_exp)``
    where level in E2M1_VALUES and block_exp in [-127, 127].

    If ``stochastic`` is True, ties are broken randomly per Rastegari-style
    stochastic rounding (unbiased in expectation). Use this inside the
    optimizer step to let weights drift between grid points.
    """
    if x.numel() == 0:
        return x

    orig_dtype = x.dtype
    work = x.to(torch.float32)
    blocks, shape, pad = _reshape_to_blocks(work, block_size)

    # Per-block scale: largest magnitude / E2M1_MAX, rounded to power of 2.
    block_max = blocks.abs().amax(dim=-1, keepdim=True)  # [num_blocks, 1]
    # Avoid log2(0); blocks that are all zero will get scale = 2^-127
    # (smallest representable) and produce all-zero outputs.
    scale_raw = block_max / E2M1_MAX
    scale_byte = _e8m0_encode(scale_raw.squeeze(-1))    # [num_blocks]
    scale = _e8m0_decode(scale_byte, torch.float32).unsqueeze(-1)  # [num_blocks, 1]

    # Scale into E2M1 domain, round to grid, scale back.
    normalized = blocks / scale
    normalized = normalized.clamp(-E2M1_MAX, E2M1_MAX)
    if stochastic:
        rounded = _round_stochastic_level(normalized, generator=generator)
    else:
        rounded = _round_to_nearest_level(normalized)
    out = rounded * scale

    return _unblock(out, shape, pad).to(orig_dtype)


def quantize_mxfp4_with_scale(
    x: torch.Tensor,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize and return (on_grid_tensor, scale_bytes).

    The on-grid tensor has the same shape and dtype as ``x``. The scale
    bytes are int32 values in [0, 254] shaped ``(num_blocks,)``. Together
    they fully describe the quantized tensor for round-trip storage.
    """
    if x.numel() == 0:
        empty_scales = torch.empty(0, dtype=torch.int32, device=x.device)
        return x, empty_scales

    orig_dtype = x.dtype
    work = x.to(torch.float32)
    blocks, shape, pad = _reshape_to_blocks(work, block_size)

    block_max = blocks.abs().amax(dim=-1, keepdim=True)
    scale_raw = block_max / E2M1_MAX
    scale_byte = _e8m0_encode(scale_raw.squeeze(-1))
    scale = _e8m0_decode(scale_byte, torch.float32).unsqueeze(-1)

    normalized = (blocks / scale).clamp(-E2M1_MAX, E2M1_MAX)
    rounded = _round_to_nearest_level(normalized)
    out = (rounded * scale)

    return _unblock(out, shape, pad).to(orig_dtype), scale_byte


# ---------------------------------------------------------------------------
# Packing: 32 E2M1 nibbles + 1 E8M0 byte per block
# ---------------------------------------------------------------------------


def _e2m1_to_index(on_grid: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Map on-grid values back to their 4-bit E2M1 index (0..15).

    Index layout (signed-magnitude, matches OCP spec):
        bit 3 = sign, bits 2..0 = magnitude index.
        magnitude index 0..7 -> {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    """
    # Normalize to unit scale so we can compare against the fixed level table.
    normalized = (on_grid.to(torch.float32) / scale)
    magnitudes = normalized.abs()

    # Round to nearest magnitude level. Use a small tolerance so fp-rounding
    # on the dequantize path doesn't shift indices.
    mag_levels = torch.tensor(E2M1_POSITIVE_VALUES, dtype=torch.float32, device=on_grid.device)
    # argmin over |magnitude - level|; expand-free via index_select-like trick.
    diff = (magnitudes.unsqueeze(-1) - mag_levels).abs()  # [..., 8]
    mag_idx = diff.argmin(dim=-1)  # [...]

    sign_bit = (normalized < 0).to(torch.int32) << 3
    return (sign_bit | mag_idx.to(torch.int32)).to(torch.uint8)


def pack_mxfp4(
    x: torch.Tensor,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` and return (packed_nibbles, scale_bytes).

    ``packed_nibbles`` is uint8 of length ``ceil(numel(x)/2)`` with two
    E2M1 values per byte (low nibble = even index, high nibble = odd).
    ``scale_bytes`` is uint8 length ``ceil(numel(x)/block_size)``.

    Together they are a complete, self-describing MXFP4 representation
    of ``x``; ``unpack_mxfp4`` with the original shape reproduces the
    round-to-nearest quantized tensor bit-exactly.
    """
    on_grid, scale_byte = quantize_mxfp4_with_scale(x, block_size=block_size)
    # Expand per-block scale so each element in the block sees its own
    # (identical) scale for the index lookup.
    scale = _e8m0_decode(scale_byte, torch.float32)
    # Flatten and align on block boundaries.
    flat_on_grid, shape, pad = _reshape_to_blocks(on_grid.to(torch.float32), block_size)
    # flat_on_grid: [num_blocks, block_size]; scale: [num_blocks]
    idx = _e2m1_to_index(flat_on_grid, scale.unsqueeze(-1))
    idx_flat = idx.reshape(-1)  # length = num_blocks * block_size (padded)

    # Pack two nibbles per byte.
    if idx_flat.numel() % 2:
        idx_flat = torch.cat([idx_flat, idx_flat.new_zeros(1)])
    low = idx_flat[0::2] & 0x0F
    high = idx_flat[1::2] & 0x0F
    packed = (low | (high << 4)).to(torch.uint8)
    return packed, scale_byte.to(torch.uint8)


def unpack_mxfp4(
    packed: torch.Tensor,
    scale_bytes: torch.Tensor,
    shape: torch.Size,
    block_size: int = 32,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Invert ``pack_mxfp4``: recover the dequantized tensor.

    ``shape`` is the tensor shape before packing (needed because packing
    may have added up to one byte of padding).
    """
    # Unpack nibbles.
    low = (packed & 0x0F).to(torch.int32)
    high = ((packed >> 4) & 0x0F).to(torch.int32)
    nibbles = torch.stack([low, high], dim=-1).reshape(-1)  # 2 nibbles per byte

    # Slice off padding at the tensor tail.
    total_elems = int(torch.tensor(shape).prod().item())
    padded_total = ((total_elems + block_size - 1) // block_size) * block_size
    nibbles = nibbles[:padded_total]

    # Decode nibble -> signed E2M1 value at unit scale.
    mag_levels = torch.tensor(E2M1_POSITIVE_VALUES, dtype=torch.float32, device=packed.device)
    sign = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    mag_idx = (nibbles & 0x7).clamp(max=7)
    magnitudes = mag_levels[mag_idx]
    unit = sign * magnitudes  # [padded_total], fp32 on unit scale

    # Apply per-block scale.
    blocks = unit.reshape(-1, block_size)
    scale = _e8m0_decode(scale_bytes.to(torch.int32), torch.float32).unsqueeze(-1)
    out = (blocks * scale).reshape(-1)[:total_elems]

    return out.reshape(shape).to(dtype)
