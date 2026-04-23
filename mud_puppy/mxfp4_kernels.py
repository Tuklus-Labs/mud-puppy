"""Triton fused dequant + matmul for MXFP4-packed weights (W4A16).

Also hosts the ``MXFP4Linear`` module that consumes these kernels.

Mirrors ``int4_kernels.py`` but with:
    * E2M1 nibble -> magnitude via a small LUT (8 fp32 values).
    * E8M0 block scale (one byte per 32-element MXFP4 block along K).
    * Fixed MXFP4 block alignment: ``BLOCK_K`` must be a multiple of 32,
      and we load ``BLOCK_K // 32`` scale bytes per iteration.

Storage layout (matches ``mxfp4.pack_mxfp4``):
    qweight: uint8[N, ceil(K/2)]         -- 2 E2M1 nibbles per byte.
    scales:  uint8[N, ceil(K/32)]        -- one E8M0 byte per K-block.
    bias:    optional dtype[N].

Forward:   C[M, N] = A[M, K] @ dequant(QW)[N, K].T  (+ bias)
Backward:  dA[M, K] = dC[M, N] @ dequant(QW)[N, K]

Both kernels dequantize inline; neither materializes the full [N, K]
weight in memory. Activations stay in bf16/fp16 (W4A16) throughout.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


# 8 positive E2M1 magnitudes. Loaded onto the device once and cached
# per-device below. Sign bit is applied separately in the kernel.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# Module-level cache: {device: LUT tensor on that device}
_LUT_CACHE: dict = {}


def _e2m1_lut(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = torch.tensor(
            E2M1_MAGNITUDES, dtype=torch.float32, device=device
        )
    return _LUT_CACHE[key]


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------


if TRITON_AVAILABLE:

    # Autotune configs: BLOCK_K must be a multiple of 32 (the MXFP4 block size).
    _FWD_CONFIGS = [
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 1},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 4},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},  num_warps=8, num_stages=3),
    ]
    _BWD_CONFIGS = list(_FWD_CONFIGS)  # Same tile space; swizzle semantics match.


    @triton.autotune(configs=_FWD_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _mxfp4_matmul_forward_kernel(
        a_ptr, qw_ptr, scales_ptr, lut_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_qwn, stride_qwk,
        stride_sn, stride_sk,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """C = A @ dequant_mxfp4(QW).T for W4A16.

        Constants:
            MXFP4 block size = 32 (OCP spec).
            BLOCK_K constraint: must be a multiple of 32.
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

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        BLOCK_K_HALF: tl.constexpr = BLOCK_K // 2
        BLOCKS_PER_TILE: tl.constexpr = BLOCK_K // 32   # MXFP4 blocks per K-tile

        offs_k_full = tl.arange(0, BLOCK_K)
        offs_k_half = tl.arange(0, BLOCK_K_HALF)
        offs_scale = tl.arange(0, BLOCKS_PER_TILE)

        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k_full
            k_mask = k_idx < K

            # Load A[BLOCK_M, BLOCK_K]
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
            a_tile = tl.load(
                a_ptrs,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            # Load packed nibbles QW[BLOCK_N, BLOCK_K_HALF]
            byte_k = k0 // 2 + offs_k_half
            byte_k_mask = (k0 + offs_k_half * 2) < K
            qw_ptrs = qw_ptr + offs_n[:, None] * stride_qwn + byte_k[None, :] * stride_qwk
            qw = tl.load(
                qw_ptrs,
                mask=n_mask[:, None] & byte_k_mask[None, :],
                other=0,
            ).to(tl.int32)

            # Extract low and high nibbles
            low = qw & 0x0F
            high = (qw >> 4) & 0x0F
            nibbles = tl.join(low, high).reshape(BLOCK_N, BLOCK_K)

            # Decode nibble: sign bit + magnitude via LUT gather.
            mag_idx = nibbles & 0x7                  # [BN, BK]
            sign_bit = (nibbles >> 3) & 0x1
            mag = tl.load(lut_ptr + mag_idx)         # fp32 [BN, BK]
            signed_val = tl.where(sign_bit == 1, -mag, mag)

            # Load E8M0 block scales [BLOCK_N, BLOCKS_PER_TILE]
            scale_k = k0 // 32 + offs_scale
            scale_k_mask = (k0 + offs_scale * 32) < K
            scale_ptrs = (
                scales_ptr
                + offs_n[:, None] * stride_sn
                + scale_k[None, :] * stride_sk
            )
            scale_bytes = tl.load(
                scale_ptrs,
                mask=n_mask[:, None] & scale_k_mask[None, :],
                other=127,  # decodes to scale=1.0 for masked positions (unused)
            ).to(tl.int32)
            block_scales = tl.exp2((scale_bytes - 127).to(tl.float32))  # [BN, BPT]

            # Broadcast scales across 32 K positions each.
            # [BN, BPT] -> [BN, BPT, 32] -> [BN, BPT*32] = [BN, BLOCK_K]
            scales_3d = block_scales.reshape(BLOCK_N, BLOCKS_PER_TILE, 1)
            scales_expanded = tl.broadcast_to(scales_3d, (BLOCK_N, BLOCKS_PER_TILE, 32))
            scale_per_k = scales_expanded.reshape(BLOCK_N, BLOCK_K)

            # Apply scale.
            w_fp = signed_val * scale_per_k
            # Zero out K positions beyond the real K boundary (for boundary tiles).
            w_fp = tl.where(k_mask[None, :], w_fp, 0.0)

            # Transpose for tl.dot(M,K) @ (K,N).
            w_fp_t = tl.trans(w_fp)
            acc += tl.dot(a_tile, w_fp_t.to(a_tile.dtype), allow_tf32=False)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


    @triton.autotune(configs=_BWD_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _mxfp4_matmul_backward_kernel(
        go_ptr, qw_ptr, scales_ptr, lut_ptr, gi_ptr,
        M, N, K,
        stride_gm, stride_gn,
        stride_qwn, stride_qwk,
        stride_sn, stride_sk,
        stride_im, stride_ik,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """grad_input[M, K] = grad_output[M, N] @ dequant_mxfp4(QW)[N, K].

        Same MXFP4 block-aware path as forward, but we now iterate N
        and produce one (M, K) tile per program.
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
        BLOCKS_PER_TILE: tl.constexpr = BLOCK_K // 32

        offs_k_half = pid_k * BLOCK_K_HALF + tl.arange(0, BLOCK_K_HALF)
        offs_scale = pid_k * BLOCKS_PER_TILE + tl.arange(0, BLOCKS_PER_TILE)
        byte_k_mask = (offs_k_half * 2) < K
        scale_k_mask = (offs_scale * 32) < K

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
            qw_ptrs = qw_ptr + offs_n[:, None] * stride_qwn + offs_k_half[None, :] * stride_qwk
            qw = tl.load(
                qw_ptrs,
                mask=n_mask[:, None] & byte_k_mask[None, :],
                other=0,
            ).to(tl.int32)
            low = qw & 0x0F
            high = (qw >> 4) & 0x0F
            nibbles = tl.join(low, high).reshape(BLOCK_N, BLOCK_K)

            mag_idx = nibbles & 0x7
            sign_bit = (nibbles >> 3) & 0x1
            mag = tl.load(lut_ptr + mag_idx)
            signed_val = tl.where(sign_bit == 1, -mag, mag)

            # E8M0 scales [BLOCK_N, BLOCKS_PER_TILE]
            scale_ptrs = scales_ptr + offs_n[:, None] * stride_sn + offs_scale[None, :] * stride_sk
            scale_bytes = tl.load(
                scale_ptrs,
                mask=n_mask[:, None] & scale_k_mask[None, :],
                other=127,
            ).to(tl.int32)
            block_scales = tl.exp2((scale_bytes - 127).to(tl.float32))

            scales_3d = block_scales.reshape(BLOCK_N, BLOCKS_PER_TILE, 1)
            scales_expanded = tl.broadcast_to(scales_3d, (BLOCK_N, BLOCKS_PER_TILE, 32))
            scale_per_k = scales_expanded.reshape(BLOCK_N, BLOCK_K)

            w_fp = signed_val * scale_per_k
            w_fp = tl.where(k_mask[None, :], w_fp, 0.0)

            acc += tl.dot(go_tile, w_fp.to(go_tile.dtype), allow_tf32=False)

        gi_ptrs = gi_ptr + offs_m[:, None] * stride_im + offs_k_full[None, :] * stride_ik
        tl.store(gi_ptrs, acc, mask=m_mask[:, None] & k_mask[None, :])


# ---------------------------------------------------------------------------
# High-level wrappers
# ---------------------------------------------------------------------------


def _is_enabled() -> bool:
    """Whether Triton MXFP4 matmul is active. Off by default."""
    val = os.environ.get("MUD_PUPPY_MXFP4_TRITON", "").lower()
    return TRITON_AVAILABLE and val in ("1", "true", "yes", "on")


def triton_mxfp4_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    """Compute ``out = x @ dequant_mxfp4(qweight, scales).T`` via Triton.

    Args:
        x: [..., K] activations (bf16/fp16/fp32).
        qweight: uint8[N, ceil(K/2)] packed E2M1 nibbles.
        scales: uint8[N, ceil(K/32)] E8M0 scale bytes.
        in_features: true K (unpadded). qweight/scales may be sized for
            a padded K; this parameter tells the kernel where to stop.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton_mxfp4_matmul requires Triton")

    orig_shape = x.shape
    N = qweight.shape[0]
    K = in_features
    x_2d = x.reshape(-1, K).contiguous()
    M = x_2d.shape[0]

    c = torch.empty(M, N, dtype=torch.float32, device=x.device)
    lut = _e2m1_lut(x.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    _mxfp4_matmul_forward_kernel[grid](
        x_2d, qweight, scales, lut, c,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        c.stride(0), c.stride(1),
    )

    return c.to(x.dtype).reshape(*orig_shape[:-1], N)


def triton_mxfp4_grad_input(
    grad_output: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    """Compute ``grad_input = grad_output @ dequant_mxfp4(qweight, scales)``.

    Args:
        grad_output: [..., N] upstream gradient.
        qweight: uint8[N, ceil(K/2)] packed nibbles.
        scales: uint8[N, ceil(K/32)] E8M0 bytes.
        in_features: true K.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton_mxfp4_grad_input requires Triton")

    orig_shape = grad_output.shape
    N = qweight.shape[0]
    K = in_features
    go_2d = grad_output.reshape(-1, N).contiguous()
    M = go_2d.shape[0]

    gi = torch.empty(M, K, dtype=torch.float32, device=grad_output.device)
    lut = _e2m1_lut(grad_output.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(K, meta["BLOCK_K"]),
    )
    _mxfp4_matmul_backward_kernel[grid](
        go_2d, qweight, scales, lut, gi,
        M, N, K,
        go_2d.stride(0), go_2d.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        gi.stride(0), gi.stride(1),
    )

    return gi.to(grad_output.dtype).reshape(*orig_shape[:-1], K)


# ---------------------------------------------------------------------------
# Packed-storage Linear module
# ---------------------------------------------------------------------------


class _MXFP4LinearFn(torch.autograd.Function):
    """Custom autograd for MXFP4-packed linear.

    Saves only the packed qweight + scales across the autograd graph
    (not the dequantized bf16 weight). For a 20B MoE model that's the
    difference between a training run fitting in 15 GB and OOMing at 40 GB.

    Uses the Triton fused kernel when ``MUD_PUPPY_MXFP4_TRITON=1`` is
    set and the kernel is available. Otherwise falls back to
    ``unpack_mxfp4`` + ``F.linear`` -- slower but correct and
    non-kernel-dependent for CI.
    """

    @staticmethod
    def forward(ctx, input, qweight, scales, bias, in_features, out_features, dtype):
        output = None
        used_triton = False
        try:
            if _is_enabled():
                output = triton_mxfp4_matmul(
                    input.contiguous(), qweight, scales, in_features=in_features,
                )
                if bias is not None:
                    output = output + bias
                used_triton = True
        except Exception as exc:  # pragma: no cover - fallback path
            log.warning("mxfp4 triton forward failed (%s); falling back to pytorch", exc)
            output = None

        if output is None:
            from .mxfp4 import unpack_mxfp4
            # Unpack may return a padded-K weight if in_features was padded
            # at quantize time; slice back to true in_features.
            padded_k = qweight.shape[-1] * 2
            w = unpack_mxfp4(qweight, scales, (out_features, padded_k), block_size=32, dtype=dtype)
            w = w[:, :in_features].contiguous()
            output = F.linear(input, w, bias)

        ctx.save_for_backward(qweight, scales)
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.dtype = dtype
        ctx.has_bias = bias is not None
        ctx.used_triton = used_triton
        return output

    @staticmethod
    def backward(ctx, grad_output):
        qweight, scales = ctx.saved_tensors
        grad_input = None
        if ctx.used_triton:
            try:
                grad_input = triton_mxfp4_grad_input(
                    grad_output.contiguous(), qweight, scales, in_features=ctx.in_features,
                )
            except Exception as exc:  # pragma: no cover
                log.warning("mxfp4 triton backward failed (%s); falling back", exc)
                grad_input = None

        if grad_input is None:
            from .mxfp4 import unpack_mxfp4
            padded_k = qweight.shape[-1] * 2
            w = unpack_mxfp4(
                qweight, scales,
                (ctx.out_features, padded_k),
                block_size=32, dtype=ctx.dtype,
            )
            w = w[:, :ctx.in_features].contiguous()
            grad_input = grad_output @ w

        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.reshape(-1, ctx.out_features).sum(0)

        # No grads for qweight, scales, in_features, out_features, dtype
        return grad_input, None, None, grad_bias, None, None, None


class MXFP4Linear(nn.Linear):
    """Frozen 4-bit linear layer with native MXFP4 packed storage.

    Inherits from ``nn.Linear`` so PEFT/LoRA recognizes it via isinstance.
    Stores weights as packed E2M1 nibbles + E8M0 block scales (~3.76x
    compression vs bf16). On-the-fly dequantize + matmul via the Triton
    kernels in this module.

    VRAM per layer: out * ceil(in/2) bytes (packed) + out * ceil(in/32) bytes (scales).
    vs nn.Linear: out * in * 2 bytes (bf16).
    Savings: ~3.76x.

    This is the *inference-and-LoRA-target* module. Weights don't update
    during training; use ``MXFP4QATLinear`` (mxfp4_train.py) for the
    QAT-with-bf16-master path.
    """

    BLOCK_SIZE: int = 32

    def __init__(self, linear: nn.Linear, dtype: torch.dtype = torch.bfloat16):
        nn.Module.__init__(self)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.dtype = dtype

        # Pad K to a multiple of BLOCK_SIZE (32) so every row has whole blocks.
        pad = (-self.in_features) % self.BLOCK_SIZE
        if pad:
            w = F.pad(linear.weight.detach(), (0, pad))
        else:
            w = linear.weight.detach()
        self._padded_in = self.in_features + pad

        from .mxfp4 import pack_mxfp4
        packed, scale_bytes = pack_mxfp4(w.to(torch.float32), block_size=self.BLOCK_SIZE)
        self.register_buffer("qweight", packed)
        self.register_buffer("scales", scale_bytes)

        if linear.bias is not None:
            self.register_parameter(
                "bias", nn.Parameter(linear.bias.detach().to(dtype), requires_grad=False)
            )
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self):
        """Dequantize on access. PEFT may read this for dtype/device info."""
        return self._dequantize()

    @weight.setter
    def weight(self, value):
        # No-op: PEFT's _prepare_model_for_kbit_training may try to assign.
        pass

    def _dequantize(self) -> torch.Tensor:
        from .mxfp4 import unpack_mxfp4
        w = unpack_mxfp4(
            self.qweight, self.scales,
            (self.out_features, self._padded_in),
            block_size=self.BLOCK_SIZE, dtype=self.dtype,
        )
        return w[:, :self.in_features].contiguous()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _MXFP4LinearFn.apply(
            input.to(self.dtype),
            self.qweight,
            self.scales,
            self.bias,
            self.in_features,
            self.out_features,
            self.dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quant=mxfp4_packed"
        )


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


def quantize_model_mxfp4(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    skip_modules: Optional[List[str]] = None,
    min_size: int = 1024,
) -> nn.Module:
    """Replace ``nn.Linear`` layers with :class:`MXFP4Linear`.

    Matches the signature of ``bnb_rocm.quantize_model_4bit`` so the CLI
    can route to either path based on ``config.quant_format``. Default
    skip list protects embeddings and output projections where 4-bit
    quant typically costs more perplexity than it saves memory.
    """
    skip_modules = skip_modules or [
        "lm_head", "embed_tokens", "word_embeddings",
        "wte", "wpe", "score", "classifier",
    ]
    for name, module in list(model.named_modules()):
        if any(s in name.lower() for s in skip_modules):
            continue
        if isinstance(module, nn.Linear) and not isinstance(module, MXFP4Linear):
            if module.weight.numel() < min_size:
                continue
            q = MXFP4Linear(module, dtype=dtype)
            _set_module(model, name, q)
    return model
