"""Triton INT4 fused dequant+GEMM for QLoRA training.

Backs ``Linear4bit.forward`` with a Triton kernel that dequantizes and
matmuls in a single pass, avoiding the allocation of a full bf16 weight
tensor per call. The PyTorch path in ``bnb_rocm._dequantize_packed``
materializes the full [out, in] bf16 weight on every forward AND every
backward -- on a 14B QLoRA run that's the dominant cost.

Storage layout (unchanged from ``bnb_rocm.Linear4bit``):

    qweight: uint8[N, ceil(K/2)]
        Row-major. Each byte holds two INT4 nibbles along the K axis:
        low nibble = even K index, high nibble = odd K index. Values
        are stored shifted by +8 so a nibble in [1, 15] represents
        a signed int in [-7, 7] after (nibble - 8).
    scale:   fp32[N, 1]   # per-row symmetric scale
    bias:    dtype[N]     # optional

Forward:  out = x @ dequant(qweight, scale).T  (+ bias)
Backward: grad_input = grad_output @ dequant(qweight, scale)

Both kernels dequant inline, never materializing the full matrix.

Calling convention keeps the existing ``Linear4bit`` contract -- no
changes to ``quantize_model_4bit`` or checkpoint format. Opt in via
``MUD_PUPPY_INT4_TRITON=1`` env var or ``config.int4_triton=True``;
default OFF until verified correct on the target model.

Correctness posture:
    - Tests compare the kernel output to the pure-PyTorch path
      (``_dequantize_packed`` then ``F.linear``) on random inputs with
      LLaMA-scale dimensions. Max elementwise abs diff must stay under
      1e-2 in bf16, 1e-3 in fp32 accumulator.
    - Backward is tested by running ``torch.autograd.gradcheck`` on a
      single layer at bf16 (loose tolerance) and checking the
      LoRA-gradient signal end-to-end over a few training steps.
    - A shape-mismatch in the kernel silently produces wrong gradients
      (training diverges over 10+ steps). The test assertion that
      catches this: ``assert torch.allclose(loss_triton, loss_naive,
      atol=1e-3)`` over a 20-step loss curve.

Design notes for future tuning:
    - kernel-anvil's ``sweep.generate_configs`` can tune the
      (BLOCK_M, BLOCK_N, BLOCK_K, num_warps) space. The cached config
      is keyed by (dtype, N, K) tuples in
      ``~/.cache/mud-puppy/kernels/<sha>.json`` (see ``kernel_cache.py``).
    - For batch_size=1 decode (M=1), a GEMV specialization wins by
      ~30%. mud-puppy training always has M>=1 (batch*seq), so we
      implement the M>=1 GEMM variant.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


# Triton is optional. If missing, callers fall back to the pure-PyTorch
# dequantize-then-matmul path in bnb_rocm.py.
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - covered by smoke-check in tests
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


if TRITON_AVAILABLE:

    # Autotune configs for the forward kernel. Covers:
    #   - small-M training (4, 16)
    #   - medium-M training (256, 512, 1024 = typical QLoRA B*S)
    # BLOCK_K must be even (nibble packing) and >= 32 for matrix cores.
    # GROUP_M controls L2 cache blocking: consecutive programs share the
    # same W slice when grouped. 4-8 is the sweet spot on RDNA3 (enough
    # L2 reuse without starving CU occupancy).
    _FWD_CONFIGS = [
        # No-swizzle fallbacks for shapes where L2 grouping hurts (small M or N)
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 1},  num_warps=4, num_stages=2),
        # Swizzled configs for larger shapes where L2 reuse pays off
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 8},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 8},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 4},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 4},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},  num_warps=8, num_stages=3),
    ]
    # Backward is symmetric to forward: grid is (M/BM, K/BK), inner over N.
    # Same config space; swizzle groups M tiles to share W reads.
    _BWD_CONFIGS = [
        # No-swizzle fallbacks
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 1},  num_warps=4, num_stages=2),
        # Swizzled configs
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 8},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 8},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 4},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 4},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},  num_warps=8, num_stages=3),
    ]


    @triton.autotune(configs=_FWD_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _int4_matmul_forward_kernel(
        # Pointers
        a_ptr, qb_ptr, scale_ptr, c_ptr,
        # Dimensions
        M, N, K,
        # Strides (element units, not bytes)
        stride_am, stride_ak,
        stride_qbn, stride_qbk,   # qb is [N, ceil(K/2)]
        stride_sn,                # scale is [N] -- broadcasts over K
        stride_cm, stride_cn,
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """C = A @ dequant(QB).T using tl.dot on dequantized tiles.

        A is [M, K] in a float dtype (bf16/fp16/fp32).
        QB[n, k_byte] holds two nibbles: low=even-K-index, high=odd-K-index.
        The kernel processes BLOCK_K columns of K per step, so BLOCK_K
        MUST be even (guaranteed by caller: 32/64/128).

        We load QB[N_tile, BLOCK_K/2] bytes then emit a [BLOCK_N, BLOCK_K]
        fp16 tile by extracting low+high nibbles and interleaving. That
        tile is then transposed to [BLOCK_K, BLOCK_N] so tl.dot(A, B)
        routes through RDNA3/CDNA matrix cores.

        Program-ID swizzling groups GROUP_M consecutive M tiles within
        each N tile so they share the same W slice in L2. With W read
        through the inner K loop, adjacent programs in scheduling order
        hit warm L2 lines.
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        m_mask = offs_m < M
        n_mask = offs_n < N

        # Per-row scale for this N tile; shape [BLOCK_N].
        scale_ptrs = scale_ptr + offs_n * stride_sn
        scale_vals = tl.load(scale_ptrs, mask=n_mask, other=0.0).to(tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # BLOCK_K MUST be even; process BLOCK_K/2 bytes per iteration which
        # expand to BLOCK_K K-columns after nibble extraction.
        BLOCK_K_HALF: tl.constexpr = BLOCK_K // 2
        offs_k_half = tl.arange(0, BLOCK_K_HALF)
        offs_k_full = tl.arange(0, BLOCK_K)

        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k_full         # [BLOCK_K] real K positions
            k_mask = k_idx < K
            k_byte_idx = k0 // 2 + offs_k_half  # [BLOCK_K_HALF] byte positions

            # Load A[M_tile, BLOCK_K]
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
            a_tile = tl.load(
                a_ptrs,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            # Load packed QB[N_tile, BLOCK_K_HALF] bytes
            qb_ptrs = (
                qb_ptr
                + offs_n[:, None] * stride_qbn
                + k_byte_idx[None, :] * stride_qbk
            )
            byte_k_mask = (k0 + offs_k_half * 2) < K  # at least low nibble valid
            qb = tl.load(
                qb_ptrs,
                mask=n_mask[:, None] & byte_k_mask[None, :],
                other=0,
            ).to(tl.int32)

            # Extract low and high nibbles; each is [BLOCK_N, BLOCK_K_HALF]
            low = (qb & 0x0F) - 8
            high = ((qb >> 4) & 0x0F) - 8

            # Interleave into [BLOCK_N, BLOCK_K]: even cols = low, odd = high.
            # Build by stacking on last axis then reshape.
            # shape: [BLOCK_N, BLOCK_K_HALF, 2] -> [BLOCK_N, BLOCK_K]
            b_int = tl.join(low, high).reshape(BLOCK_N, BLOCK_K)
            b_fp = b_int.to(tl.float32) * scale_vals[:, None]

            # Mask out-of-range K columns to zero so they don't contaminate the dot.
            b_fp = tl.where(k_mask[None, :], b_fp, 0.0)

            # tl.dot wants (M, K) @ (K, N); we have B as [N, K] -> transpose to [K, N].
            b_fp_t = tl.trans(b_fp)

            # Cast to A's dtype for the matmul; accumulator stays fp32.
            acc += tl.dot(a_tile, b_fp_t.to(a_tile.dtype), allow_tf32=False)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


    @triton.autotune(configs=_BWD_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _int4_matmul_backward_kernel(
        # Pointers
        go_ptr, qb_ptr, scale_ptr, gi_ptr,
        # Dimensions
        M, N, K,
        # Strides
        stride_gm, stride_gn,     # grad_output is [M, N]
        stride_qbn, stride_qbk,
        stride_sn,
        stride_im, stride_ik,     # grad_input is [M, K]
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """grad_input[M, K] = grad_output[M, N] @ W[M, N] with W dequanted inline.

        BLOCK_K must be even. Walks N in BLOCK_N chunks, loading grad_output
        [M, BLOCK_N] and W[BLOCK_N, BLOCK_K] (expanded from packed int4)
        and accumulating via tl.dot.

        Swizzle packs GROUP_M consecutive M tiles per K tile. The inner N
        loop reads W[:, BK slice of K]. Programs sharing a pid_k share
        the same W columns, so L2 hits amortize the otherwise-redundant
        47x DRAM reads that hurt the naive (M, K) grid order.
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_k = tl.cdiv(K, BLOCK_K)
        num_pid_in_group = GROUP_M * num_pid_k
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_k = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k_full = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        m_mask = offs_m < M
        k_mask = offs_k_full < K

        BLOCK_K_HALF: tl.constexpr = BLOCK_K // 2
        offs_k_half = pid_k * BLOCK_K_HALF + tl.arange(0, BLOCK_K_HALF)
        byte_k_mask = (offs_k_half * 2) < K

        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N

            # grad_output[M, BLOCK_N]
            go_ptrs = go_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
            go_tile = tl.load(
                go_ptrs,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            # Packed weights [BLOCK_N, BLOCK_K_HALF] bytes
            qb_ptrs = (
                qb_ptr
                + offs_n[:, None] * stride_qbn
                + offs_k_half[None, :] * stride_qbk
            )
            qb = tl.load(
                qb_ptrs,
                mask=n_mask[:, None] & byte_k_mask[None, :],
                other=0,
            ).to(tl.int32)
            low = (qb & 0x0F) - 8
            high = ((qb >> 4) & 0x0F) - 8

            w_int = tl.join(low, high).reshape(BLOCK_N, BLOCK_K)

            scale_ptrs = scale_ptr + offs_n * stride_sn
            scale_vals = tl.load(scale_ptrs, mask=n_mask, other=0.0).to(tl.float32)
            w_fp = w_int.to(tl.float32) * scale_vals[:, None]
            w_fp = tl.where(k_mask[None, :], w_fp, 0.0)

            # tl.dot(grad_output[M,N], W[N,K]) -> grad_input[M,K]
            acc += tl.dot(go_tile, w_fp.to(go_tile.dtype), allow_tf32=False)

        gi_ptrs = gi_ptr + offs_m[:, None] * stride_im + offs_k_full[None, :] * stride_ik
        tl.store(gi_ptrs, acc, mask=m_mask[:, None] & k_mask[None, :])


# ---------------------------------------------------------------------------
# High-level wrappers
# ---------------------------------------------------------------------------


# Default tile sizes. kernel-anvil's sweep can replace these per-shape,
# but these are a safe starting point for 7900 XTX / RDNA3.
_DEFAULT_FWD_CONFIG = {
    "BLOCK_M": 16,
    "BLOCK_N": 64,
    "BLOCK_K": 64,
    "num_warps": 4,
    "num_stages": 2,
}
_DEFAULT_BWD_CONFIG = {
    "BLOCK_M": 16,
    "BLOCK_N": 32,
    "BLOCK_K": 64,
    "num_warps": 4,
    "num_stages": 2,
}


def _is_enabled() -> bool:
    """Whether the Triton int4 path should be used.

    Off by default while the kernel is being validated. Enable via:
        export MUD_PUPPY_INT4_TRITON=1
    """
    val = os.environ.get("MUD_PUPPY_INT4_TRITON", "").lower()
    return TRITON_AVAILABLE and val in ("1", "true", "yes", "on")


def triton_int4_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """Compute ``out = x @ dequant(qweight, scale).T`` via Triton.

    x is [..., K] in a float dtype. qweight is [N, ceil(K/2)] uint8.
    scale is [N, 1] or [N] fp32. Output is [..., N] in x's dtype.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton_int4_matmul requires Triton")

    orig_shape = x.shape
    N = qweight.shape[0]
    K_packed = qweight.shape[1]
    # K can be odd (padded at quantization time to the next even value);
    # the caller stores the true K separately but we infer from input.
    K_from_x = orig_shape[-1]
    if K_from_x > K_packed * 2:
        raise ValueError(
            f"input K={K_from_x} exceeds packed capacity {K_packed*2}"
        )
    K = K_from_x

    x_2d = x.reshape(-1, K).contiguous()
    M = x_2d.shape[0]

    out_dtype = x.dtype
    # Accumulator in fp32; caller converts back.
    c = torch.empty(M, N, dtype=torch.float32, device=x.device)

    scale_1d = scale.reshape(-1).contiguous().to(torch.float32)

    # 1D grid because the kernel does its own (pid_m, pid_n) swizzling
    # for L2 cache reuse. Total programs = cdiv(M, BM) * cdiv(N, BN).
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    _int4_matmul_forward_kernel[grid](
        x_2d, qweight, scale_1d, c,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        qweight.stride(0), qweight.stride(1),
        scale_1d.stride(0),
        c.stride(0), c.stride(1),
    )

    return c.to(out_dtype).reshape(*orig_shape[:-1], N)


def triton_int4_grad_input(
    grad_output: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    in_features: int,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """Compute ``grad_input = grad_output @ dequant(qweight, scale)``.

    grad_output is [..., N], qweight is [N, ceil(K/2)] uint8, scale is
    [N]. ``in_features`` is the true K (unpadded). Returns [..., K].
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton_int4_grad_input requires Triton")

    orig_shape = grad_output.shape
    N = qweight.shape[0]
    K = in_features

    go_2d = grad_output.reshape(-1, N).contiguous()
    M = go_2d.shape[0]

    gi = torch.empty(M, K, dtype=torch.float32, device=grad_output.device)
    scale_1d = scale.reshape(-1).contiguous().to(torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(K, meta["BLOCK_K"]),
    )
    _int4_matmul_backward_kernel[grid](
        go_2d, qweight, scale_1d, gi,
        M, N, K,
        go_2d.stride(0), go_2d.stride(1),
        qweight.stride(0), qweight.stride(1),
        scale_1d.stride(0),
        gi.stride(0), gi.stride(1),
    )

    return gi.to(grad_output.dtype).reshape(*orig_shape[:-1], K)
