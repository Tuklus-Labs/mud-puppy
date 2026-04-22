"""Real GPTQ post-training quantization for ROCm (and CUDA).

Implements the algorithm from Frantar et al. 2023:
  "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
  https://arxiv.org/abs/2210.17323

This replaces the previous reference implementation which had no Hessian-based
calibration, no actorder, and no group-wise scales.

Memory layout
-------------
For a linear layer with weight W of shape [out_features, in_features]:

  - int4 packed weights: ceil(in_features / 2) * out_features bytes
    (two 4-bit values per uint8 byte, low nibble first)
  - scales: out_features * n_groups * 4 bytes (float32)
    where n_groups = ceil(in_features / group_size)
  - optional zero_points: same shape as scales (float32)
  - optional permutation: in_features * 4 bytes (int64)

Total vs fp32: (in*out)/2 bytes weights + (in/group_size)*out*4 bytes scales.
For group_size=128 and typical in=out=4096: ~8.5MB vs 64MB fp32 (7.5x reduction).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bnb_rocm import _set_module

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTQQuantizer:
    """Configuration for the GPTQ algorithm.

    Attributes
    ----------
    bits:
        Number of quantization bits. Typically 4.
    group_size:
        Number of input columns sharing one scale (and zero_point). 128 is the
        standard used by auto-gptq. Use -1 for per-channel (whole row = 1 group).
    actorder:
        If True, reorder columns by descending Hessian diagonal before
        quantizing (desc_act in auto-gptq). Typically improves perplexity.
    damp_percent:
        Fraction of mean(diag(H)) added to H diagonal for numerical stability.
    desc_act:
        Alias for actorder (kept for compatibility with auto-gptq naming).
    """

    bits: int = 4
    group_size: int = 128
    actorder: bool = True
    damp_percent: float = 0.01
    desc_act: bool = True   # mirrors actorder when set

    def __post_init__(self):
        # desc_act defaults to True at field level; only override actorder if the user
        # explicitly set desc_act to False (i.e. it differs from its default and actorder
        # was not also explicitly set). The simplest correct rule: actorder wins because
        # it is the primary field. desc_act is kept as an alias that mirrors actorder.
        self.desc_act = self.actorder

    @property
    def qmin(self) -> int:
        return -(2 ** (self.bits - 1))

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1


# ---------------------------------------------------------------------------
# Hessian computation
# ---------------------------------------------------------------------------

def _compute_hessian(
    X: torch.Tensor,
    damp_percent: float = 0.01,
) -> torch.Tensor:
    """Compute Hessian H = 2 X^T X / n + damp * I.

    Parameters
    ----------
    X:
        Activation matrix of shape [n_samples, in_features].
    damp_percent:
        Fraction of mean(diag(H_undamped)) added to diagonal.

    Returns
    -------
    H:
        Hessian matrix [in_features, in_features], float32.
    """
    X = X.to(torch.float32)
    n = X.shape[0]
    if n == 0:
        in_f = X.shape[1]
        return torch.eye(in_f, dtype=torch.float32, device=X.device)

    # H = 2 * X^T X / n  (factor 2 from squared-error Hessian)
    H = (2.0 / n) * (X.T @ X)

    # Damping: add damp_percent * mean(diag(H)) to diagonal
    diag_mean = H.diag().mean()
    damp = damp_percent * diag_mean
    H.diagonal().add_(damp)

    return H


# ---------------------------------------------------------------------------
# int4 packing / unpacking
# ---------------------------------------------------------------------------

def pack_int4(w: torch.Tensor) -> torch.Tensor:
    """Pack an int8 tensor (values in [-8, 7]) into uint8, two nibbles per byte.

    The low nibble stores the even column, the high nibble stores the odd column.
    If in_features is odd, the last column is stored in the low nibble of the
    last byte with the high nibble set to 0.

    Parameters
    ----------
    w:
        Weight tensor [out_features, in_features], dtype int8, values in [-8, 7].

    Returns
    -------
    packed:
        Tensor [out_features, ceil(in_features / 2)], dtype uint8.
    """
    if w.numel() == 0:
        # C8: empty tensor -- return an appropriately-shaped empty uint8 tensor.
        out_f = w.shape[0] if w.ndim >= 1 else 0
        return torch.empty((out_f, 0), dtype=torch.uint8, device=w.device)
    out_f, in_f = w.shape
    pad = in_f % 2
    if pad:
        # Pad with zero in the high nibble position
        w = F.pad(w.float(), (0, 1)).to(torch.int8)

    # Shift values from [-8,7] to [0,15] so nibbles are unsigned
    w_u = (w.to(torch.int32) + 8).to(torch.uint8)
    even = w_u[:, 0::2]   # low nibble
    odd = w_u[:, 1::2]    # high nibble
    packed = (even & 0x0F) | ((odd & 0x0F) << 4)
    return packed.to(torch.uint8)


def unpack_int4(
    packed: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Unpack a uint8 tensor back to int8 values in [-8, 7].

    Parameters
    ----------
    packed:
        Tensor [out_features, ceil(in_features / 2)], dtype uint8.
    out_features:
        Original out_features (rows).
    in_features:
        Original in_features (columns, before packing).

    Returns
    -------
    w:
        Tensor [out_features, in_features], dtype int8, values in [-8, 7].
    """
    even = (packed & 0x0F).to(torch.int32) - 8
    odd = ((packed >> 4) & 0x0F).to(torch.int32) - 8
    # Interleave: [even0, odd0, even1, odd1, ...]
    combined = torch.stack([even, odd], dim=2).reshape(out_features, -1)
    # Trim to original in_features
    return combined[:, :in_features].to(torch.int8)


# ---------------------------------------------------------------------------
# Core GPTQ per-layer quantizer
# ---------------------------------------------------------------------------

def _gptq_quantize_layer(
    W: torch.Tensor,
    X: torch.Tensor,
    quantizer: GPTQQuantizer,
    block_size: int = 128,
) -> Dict[str, Any]:
    """Quantize a single weight matrix using the GPTQ algorithm.

    Parameters
    ----------
    W:
        Weight matrix [out_features, in_features], float32.
    X:
        Calibration activations [n_samples, in_features], float32.
    quantizer:
        GPTQQuantizer config.
    block_size:
        Number of columns processed per Cholesky-error-propagation block.

    Returns
    -------
    dict with keys:
        qweight:       int8 quantized weights [out, in], values in [qmin, qmax]
        qweight_packed: uint8 packed weights [out, ceil(in/2)]
        scales:        float32 scales [out, n_groups]
        zero_points:   float32 zero points [out, n_groups] (symmetric -> 0)
        weight_dequant: float32 dequantized weight (for MSE computation)
        perm:          int64 permutation (only present if actorder=True)
    """
    W = W.to(torch.float32).clone()
    out_f, in_f = W.shape
    device = W.device

    # C7: reject NaN/Inf in weights before starting. A corrupt weight matrix
    # would silently produce a garbage quantized model.
    if not torch.isfinite(W).all():
        raise RuntimeError(
            "GPTQ: layer weights contain non-finite values (NaN/Inf). "
            "Check that the model loaded correctly and that no prior "
            "quantization or training step corrupted the weights."
        )

    H = _compute_hessian(X, damp_percent=quantizer.damp_percent)
    H = H.to(device)

    # C7: validate the Hessian. A zero or NaN diagonal means the calibration
    # data did not activate this layer -- quantizing with a degenerate Hessian
    # silently produces the wrong scales.
    if not torch.isfinite(H).all():
        raise RuntimeError(
            "GPTQ calibration produced a Hessian with NaN/Inf values. "
            "Ensure calibration activations are finite (no NaN inputs)."
        )
    if H.diag().max() == 0:
        raise RuntimeError(
            "GPTQ calibration produced a zero Hessian diagonal. "
            "This usually means the calibration data never activated this "
            "layer. Provide representative calibration data."
        )

    # --- actorder: reorder columns by descending H diagonal ---
    perm: Optional[torch.Tensor] = None
    if quantizer.actorder:
        perm = torch.argsort(H.diag(), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]

    # --- Cholesky decomposition of H for stable inverse ---
    # H = L L^T  =>  H^{-1} = L^{-T} L^{-1}
    # We use the inverse Cholesky factor directly.
    try:
        L = torch.linalg.cholesky(H)
        # Hinv via triangular solve: solve L Hinv_half = I  => Hinv_half = L^{-1}
        # Then Hinv = Hinv_half^T @ Hinv_half  (but we only need column-wise access)
        # Faster: just store the full Hinv using cholesky_solve
        I = torch.eye(in_f, device=device, dtype=torch.float32)
        Hinv = torch.cholesky_solve(I, L)
    except torch.linalg.LinAlgError:
        # Fallback: direct inverse with extra damping
        log.warning("Cholesky failed; falling back to direct inverse with extra damping")
        diag_mean = H.diag().mean()
        H.diagonal().add_(diag_mean * 0.1)
        Hinv = torch.linalg.inv(H)

    # --- Group size ---
    group_size = quantizer.group_size if quantizer.group_size > 0 else in_f
    n_groups = math.ceil(in_f / group_size)

    # --- Storage for quantized values ---
    W_q = torch.zeros_like(W, dtype=torch.int8)   # quantized ints
    W_dq = torch.zeros_like(W)                     # dequantized floats
    scales = torch.zeros(out_f, n_groups, device=device, dtype=torch.float32)
    zero_points = torch.zeros(out_f, n_groups, device=device, dtype=torch.float32)

    # Work on a mutable copy to propagate error
    W_work = W.clone()

    # --- Block-wise GPTQ loop ---
    qmin = quantizer.qmin
    qmax = quantizer.qmax

    for block_start in range(0, in_f, block_size):
        block_end = min(block_start + block_size, in_f)

        # Slice into current block
        W_block = W_work[:, block_start:block_end].clone()
        Hinv_block = Hinv[block_start:block_end, block_start:block_end]

        W_q_block = torch.zeros_like(W_block, dtype=torch.int8)
        W_dq_block = torch.zeros_like(W_block)
        err_block = torch.zeros_like(W_block)

        for j in range(block_end - block_start):
            col = block_start + j  # global column index

            # Compute scale for this column's group
            g = col // group_size
            g_start = g * group_size
            g_end = min(g_start + group_size, in_f)

            # Per-group scale: symmetric, use current group's weight range
            # We only recompute scale at the start of each group
            if col % group_size == 0:
                w_group = W_work[:, g_start:g_end]
                abs_max = w_group.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
                group_scale = abs_max / qmax
                scales[:, g] = group_scale.squeeze(1)

            s = scales[:, g].unsqueeze(1)  # [out_f, 1]

            # Quantize column j
            w_col = W_work[:, col]
            w_q_col = torch.clamp((w_col.unsqueeze(1) / s).round(), qmin, qmax).squeeze(1)
            w_q_col = w_q_col.to(torch.int8)
            w_dq_col = w_q_col.float() * s.squeeze(1)

            W_q_block[:, j] = w_q_col
            W_dq_block[:, j] = w_dq_col

            # Error: difference after quantization, scaled by Hinv diagonal
            h_jj = Hinv_block[j, j].clamp(min=1e-8)
            err_col = (w_col - w_dq_col) / h_jj  # [out_f]
            err_block[:, j] = err_col

            # Propagate error to remaining columns in this block
            if j + 1 < block_end - block_start:
                # W_work[:, col+1:block_end] -= err_col outer Hinv_block[j, j+1:]
                W_work[:, col + 1 : block_end] -= (
                    err_col.unsqueeze(1) * Hinv_block[j, j + 1 :].unsqueeze(0)
                )

        # After processing the block, propagate error to all remaining columns
        if block_end < in_f:
            # err_block: [out_f, block_size], Hinv[block_start:block_end, block_end:]: [block_size, rest]
            W_work[:, block_end:] -= err_block @ Hinv[block_start:block_end, block_end:]

        W_q[:, block_start:block_end] = W_q_block
        W_dq[:, block_start:block_end] = W_dq_block

    result: Dict[str, Any] = {
        "qweight": W_q,
        "qweight_packed": pack_int4(W_q),
        "scales": scales,
        "zero_points": zero_points,
        "weight_dequant": W_dq,
    }
    if perm is not None:
        result["perm"] = perm

    return result


# ---------------------------------------------------------------------------
# Observer: captures activations during calibration
# ---------------------------------------------------------------------------

class GPTQObserver:
    """Forward hook that accumulates the Hessian across calibration batches.

    Attach this as a forward hook on an nn.Linear module. After the calibration
    pass, call get_hessian() to retrieve H = 2 X^T X / n.

    Usage
    -----
    observer = GPTQObserver()
    hook = module.register_forward_hook(observer)
    # ... run calibration data through model ...
    hook.remove()
    H = observer.get_hessian()
    """

    def __init__(self):
        self.H: Optional[torch.Tensor] = None  # Accumulated 2*X^T*X (unnormalized)
        self.n_samples: int = 0

    def __call__(self, module: nn.Module, args, output):
        """Forward hook: args[0] is the input tensor to the linear layer."""
        # Handle tuple inputs (e.g. from attention layers)
        x = args[0].detach().to(torch.float32)

        # Flatten batch and sequence dims: [batch, seq, in_f] -> [batch*seq, in_f]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() > 3:
            x = x.reshape(-1, x.shape[-1])

        n = x.shape[0]

        # Accumulate: running sum of X^T X (unnormalized)
        # H_acc = 2 * sum_i(x_i^T x_i) = 2 * X^T X
        batch_H = x.T @ x  # [in_f, in_f]

        if self.H is None:
            self.H = 2.0 * batch_H
        else:
            self.H = self.H + 2.0 * batch_H

        self.n_samples += n

    def get_hessian(self, damp_percent: float = 0.01) -> torch.Tensor:
        """Return the normalized and damped Hessian H = 2 X^T X / n + damp * I."""
        if self.H is None or self.n_samples == 0:
            raise RuntimeError("No activations captured. Run calibration data first.")

        H = self.H / self.n_samples

        # Damping
        diag_mean = H.diag().mean()
        damp = damp_percent * diag_mean
        H = H.clone()
        H.diagonal().add_(damp)
        return H

    def get_activations_matrix(self) -> torch.Tensor:
        """Return a pseudo-activation matrix for use with _compute_hessian.

        Since we accumulate X^T X directly, we return a synthetic X-like
        structure derived from the accumulated H. For _gptq_quantize_layer,
        which calls _compute_hessian internally, we instead pass the H directly
        via _gptq_quantize_layer_from_H.
        """
        raise NotImplementedError("Use get_hessian() directly.")


# ---------------------------------------------------------------------------
# GPTQLinear: inference layer with packed int4 weights
# ---------------------------------------------------------------------------

class GPTQLinear(nn.Module):
    """Linear layer with Hessian-calibrated int4 packed weights.

    Forward pass dequantizes on-the-fly (scales are small, so this is cheap).
    Supports group-size quantization and actorder permutation.

    Parameters
    ----------
    in_features:
        Width of input tensor.
    out_features:
        Width of output tensor.
    bias:
        Optional bias tensor.
    quantizer:
        GPTQQuantizer config used during calibration (needed to decode).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[torch.Tensor],
        quantizer: GPTQQuantizer,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantizer = quantizer
        self.group_size = quantizer.group_size if quantizer.group_size > 0 else in_features
        self.n_groups = math.ceil(in_features / self.group_size)

        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

        # Placeholder buffers; populated by from_quantized()
        packed_cols = math.ceil(in_features / 2)
        self.register_buffer(
            "qweight_packed",
            torch.zeros(out_features, packed_cols, dtype=torch.uint8),
        )
        self.register_buffer(
            "scales",
            torch.ones(out_features, self.n_groups, dtype=torch.float32),
        )
        self.register_buffer(
            "zero_points",
            torch.zeros(out_features, self.n_groups, dtype=torch.float32),
        )
        self.register_buffer("perm", None)  # type: ignore[arg-type]

    @classmethod
    def from_quantized(
        cls,
        original_linear: nn.Linear,
        result: Dict[str, Any],
        quantizer: GPTQQuantizer,
    ) -> "GPTQLinear":
        """Construct a GPTQLinear from a _gptq_quantize_layer result dict."""
        bias = original_linear.bias
        layer = cls(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            bias=bias,
            quantizer=quantizer,
        )
        layer.qweight_packed = result["qweight_packed"]
        layer.scales = result["scales"]
        layer.zero_points = result["zero_points"]
        if "perm" in result and result["perm"] is not None:
            layer.perm = result["perm"]
        return layer

    def _dequant_weight(self) -> torch.Tensor:
        """Dequantize packed int4 weights to float32."""
        W_q = unpack_int4(self.qweight_packed, self.out_features, self.in_features)
        W_q_f = W_q.to(torch.float32)  # [out_f, in_f]

        # Apply per-group scales
        # scales: [out_f, n_groups], broadcast over columns in each group
        W_dq = torch.zeros_like(W_q_f)
        for g in range(self.n_groups):
            col_start = g * self.group_size
            col_end = min(col_start + self.group_size, self.in_features)
            s = self.scales[:, g].unsqueeze(1)  # [out_f, 1]
            z = self.zero_points[:, g].unsqueeze(1)
            W_dq[:, col_start:col_end] = W_q_f[:, col_start:col_end] * s + z

        # Undo actorder permutation
        if self.perm is not None:
            inv_perm = torch.argsort(self.perm)
            W_dq = W_dq[:, inv_perm]

        return W_dq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        W_dq = self._dequant_weight().to(orig_dtype)
        out = F.linear(x, W_dq, self.bias)
        return out


# ---------------------------------------------------------------------------
# GPTQ quantize layer using pre-computed Hessian
# ---------------------------------------------------------------------------

def _gptq_quantize_layer_from_H(
    W: torch.Tensor,
    H: torch.Tensor,
    quantizer: GPTQQuantizer,
    block_size: int = 128,
) -> Dict[str, Any]:
    """Quantize a weight matrix given pre-computed Hessian H (no damping applied yet).

    This is used by quantize_model_gptq after GPTQObserver has collected H.
    H must already be damped (i.e. returned by observer.get_hessian()).
    """
    W = W.to(torch.float32).clone()
    out_f, in_f = W.shape
    device = W.device
    H = H.to(device)

    # --- actorder: reorder columns by descending H diagonal ---
    perm: Optional[torch.Tensor] = None
    if quantizer.actorder:
        perm = torch.argsort(H.diag(), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]

    # --- Cholesky inverse of H ---
    try:
        L = torch.linalg.cholesky(H)
        I = torch.eye(in_f, device=device, dtype=torch.float32)
        Hinv = torch.cholesky_solve(I, L)
    except torch.linalg.LinAlgError:
        log.warning("Cholesky failed in _gptq_quantize_layer_from_H; using direct inverse")
        Hinv = torch.linalg.pinv(H)

    group_size = quantizer.group_size if quantizer.group_size > 0 else in_f
    n_groups = math.ceil(in_f / group_size)

    W_q = torch.zeros_like(W, dtype=torch.int8)
    W_dq = torch.zeros_like(W)
    scales = torch.zeros(out_f, n_groups, device=device, dtype=torch.float32)
    zero_points = torch.zeros(out_f, n_groups, device=device, dtype=torch.float32)
    W_work = W.clone()

    qmin = quantizer.qmin
    qmax = quantizer.qmax

    for block_start in range(0, in_f, block_size):
        block_end = min(block_start + block_size, in_f)
        W_block = W_work[:, block_start:block_end].clone()
        Hinv_block = Hinv[block_start:block_end, block_start:block_end]

        W_q_block = torch.zeros_like(W_block, dtype=torch.int8)
        W_dq_block = torch.zeros_like(W_block)
        err_block = torch.zeros_like(W_block)

        for j in range(block_end - block_start):
            col = block_start + j
            g = col // group_size

            if col % group_size == 0:
                g_start = g * group_size
                g_end = min(g_start + group_size, in_f)
                w_group = W_work[:, g_start:g_end]
                abs_max = w_group.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
                group_scale = abs_max / qmax
                scales[:, g] = group_scale.squeeze(1)

            s = scales[:, g].unsqueeze(1)
            w_col = W_work[:, col]
            w_q_col = torch.clamp((w_col.unsqueeze(1) / s).round(), qmin, qmax).squeeze(1)
            w_q_col = w_q_col.to(torch.int8)
            w_dq_col = w_q_col.float() * s.squeeze(1)

            W_q_block[:, j] = w_q_col
            W_dq_block[:, j] = w_dq_col

            h_jj = Hinv_block[j, j].clamp(min=1e-8)
            err_col = (w_col - w_dq_col) / h_jj
            err_block[:, j] = err_col

            if j + 1 < block_end - block_start:
                W_work[:, col + 1 : block_end] -= (
                    err_col.unsqueeze(1) * Hinv_block[j, j + 1 :].unsqueeze(0)
                )

        if block_end < in_f:
            W_work[:, block_end:] -= err_block @ Hinv[block_start:block_end, block_end:]

        W_q[:, block_start:block_end] = W_q_block
        W_dq[:, block_start:block_end] = W_dq_block

    result: Dict[str, Any] = {
        "qweight": W_q,
        "qweight_packed": pack_int4(W_q),
        "scales": scales,
        "zero_points": zero_points,
        "weight_dequant": W_dq,
    }
    if perm is not None:
        result["perm"] = perm

    return result


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def quantize_model_gptq(
    model: nn.Module,
    calibration_data: List[torch.Tensor],
    bits: int = 4,
    group_size: int = 128,
    actorder: bool = True,
    damp_percent: float = 0.01,
    skip_modules: Optional[List[str]] = None,
    min_size: int = 1024,
    block_size: int = 128,
) -> nn.Module:
    """Quantize all eligible nn.Linear layers using real GPTQ.

    Parameters
    ----------
    model:
        Loaded fp32/bf16 model. Will be modified in-place.
    calibration_data:
        List of input tensors, each shape [batch, in_features] or
        [batch, seq, in_features]. Typically 128 batches of the training set.
    bits:
        Quantization bits (4 is standard).
    group_size:
        Column group size for separate scales (128 is standard).
    actorder:
        If True, reorder columns by Hessian diagonal (improves perplexity).
    damp_percent:
        Damping fraction for Hessian stability.
    skip_modules:
        List of substring patterns for module names to skip.
    min_size:
        Skip layers with fewer than this many weight elements.
    block_size:
        Column block size for GPTQ error propagation loop (128 is standard).

    Returns
    -------
    model:
        Model with nn.Linear layers replaced by GPTQLinear.
    """
    skip_modules = skip_modules or [
        "lm_head",
        "embed_tokens",
        "word_embeddings",
        "wte",
        "wpe",
        "score",
        "classifier",
    ]

    quantizer = GPTQQuantizer(
        bits=bits,
        group_size=group_size,
        actorder=actorder,
        damp_percent=damp_percent,
    )

    # Collect eligible layers and attach observers
    observers: Dict[str, GPTQObserver] = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(skip in name.lower() for skip in skip_modules):
            continue
        if module.weight.numel() < min_size:
            continue

        obs = GPTQObserver()
        hook = module.register_forward_hook(obs)
        observers[name] = obs
        hooks.append(hook)

    if not observers:
        log.warning("quantize_model_gptq: no eligible layers found")
        return model

    log.info("quantize_model_gptq: collecting activations for %d layers", len(observers))

    # Run calibration data through model
    model.eval()
    with torch.no_grad():
        for batch in calibration_data:
            # calibration_data items are [batch, in_features] or model-specific inputs
            # For a raw linear-model test (ToyModel), the input goes directly to model
            try:
                model(batch)
            except Exception as e:
                # A silently-swallowed failure leaves observers with zero
                # samples, producing an unquantized model with no diagnostic.
                # Raise immediately so the caller gets a clear error with
                # actionable context (batch shape/dtype/device mismatch).
                raise RuntimeError(
                    f"GPTQ calibration forward failed on batch: {e}. "
                    "Check batch shape/dtype/device matches model expectations."
                ) from e

    # Remove hooks
    for hook in hooks:
        hook.remove()

    log.info("quantize_model_gptq: quantizing layers")

    # Quantize each layer using collected Hessians
    for name, module in list(model.named_modules()):
        if name not in observers:
            continue

        obs = observers[name]
        if obs.n_samples == 0:
            log.warning("Layer %s: no calibration samples collected, skipping", name)
            continue

        H = obs.get_hessian(damp_percent=damp_percent)
        W = module.weight.detach().float()

        result = _gptq_quantize_layer_from_H(W, H, quantizer, block_size=block_size)
        gptq_lin = GPTQLinear.from_quantized(module, result, quantizer)
        _set_module(model, name, gptq_lin)
        log.info("Quantized layer %s (%dx%d)", name, module.out_features, module.in_features)

    return model


def save_quantized(model: nn.Module, path: str):
    """Save quantized model weights.

    Uses save_pretrained if available, falls back to torch.save for non-HF models.
    """
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
    else:
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), f"{path}/model.pt")


def load_quantized(
    model_cls,
    path: str,
    trust_remote_code: bool = False,
    **kwargs,
) -> nn.Module:
    """Load a GPTQ-quantized model saved with save_quantized.

    This assumes the model was saved via save_pretrained after quantize_model_gptq.
    The GPTQLinear layers will be reconstructed from the saved state_dict.
    """
    if hasattr(model_cls, "from_pretrained"):
        model = model_cls.from_pretrained(
            path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    else:
        model = model_cls(path, **kwargs)
        state = torch.load(f"{path}/model.pt", weights_only=True)
        model.load_state_dict(state)
    return model
