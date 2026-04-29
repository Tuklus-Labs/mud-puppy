"""Runner adapters for ``kernel-anvil`` against the mud-puppy training kernels.

``kernel-anvil`` discovers candidate Triton configs by importing a
runner module (via ``importlib.util.spec_from_file_location``) and
calling its ``setup() -> inputs``, ``reference(inputs) -> tensor``, and
``run(inputs, **config) -> tensor`` functions. See
``kernel_anvil/cli.py::_load_runner`` and
``kernel_anvil/examples/simple_gemv.py`` for the canonical example.

This module exposes four runner *namespaces* -- one per op the autotune
JSON v1 contract names:

  * :data:`mxfp4_fwd_runner`
  * :data:`mxfp4_grad_input_runner`
  * :data:`int4_fwd_runner`
  * :data:`int4_grad_input_runner`

Each is a small object with ``setup``, ``reference``, ``run``,
``BASELINE_CONFIG``, and ``DATA_BYTES`` attributes that satisfy the
runner contract. Anvil's ``_load_runner`` is happy with attributes on
modules; for callers that prefer a real module, the four convenience
functions :func:`make_mxfp4_fwd_runner_module` etc. build a fresh
``types.ModuleType`` with the required attributes wired up. Either
form works.

Shape passing
~~~~~~~~~~~~~

The example runner declares its (N, K) at module top level. We can't do
that here because anvil sweeps multiple shapes per session and module
imports are cached. Instead, the runner reads the target ``(M, N, K)``
from environment variables -- ``ANVIL_TRAIN_M``, ``ANVIL_TRAIN_N``,
``ANVIL_TRAIN_K`` -- which kernel-anvil's training pipeline sets before
calling ``setup()``. The values are also overridable via module-level
globals (``M``, ``N``, ``K``) for inline experiments.

Defaults are a small sanity-check shape (M=64, N=128, K=128) so a bare
``runner.setup(); runner.reference(inputs)`` call doesn't blow up if
the env vars are unset.
"""

from __future__ import annotations

import os
import types
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .mxfp4 import pack_mxfp4, unpack_mxfp4


# ---------------------------------------------------------------------------
# Shape resolution
# ---------------------------------------------------------------------------


# Defaults are intentionally tiny so a no-env smoke test runs in <1s.
_DEFAULT_M = 64
_DEFAULT_N = 128
_DEFAULT_K = 128


def _resolve_shape(
    default_m: int = _DEFAULT_M,
    default_n: int = _DEFAULT_N,
    default_k: int = _DEFAULT_K,
) -> tuple[int, int, int]:
    """Resolve (M, N, K) for this runner invocation.

    Order of precedence (highest first):
        1. Environment variables ``ANVIL_TRAIN_M/N/K`` (set by the
           kernel-anvil training pipeline before calling setup()).
        2. Module-level globals on the runner module if present.
        3. Defaults baked into this module.

    Module globals are resolved by the caller (the per-runner setup
    closure), not here. The env vars are the canonical wire format.
    """
    def _ienv(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    return (
        _ienv("ANVIL_TRAIN_M", default_m),
        _ienv("ANVIL_TRAIN_N", default_n),
        _ienv("ANVIL_TRAIN_K", default_k),
    )


def _device() -> torch.device:
    """Pick the device to run on. CUDA/HIP is required for Triton."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError(
        "kernel-anvil training runners require a CUDA/HIP device; got CPU only."
    )


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _make_random_input(M: int, K: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    torch.manual_seed(0xA1)
    return torch.randn(M, K, device=_device(), dtype=dtype)


def _make_random_grad_output(M: int, N: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    torch.manual_seed(0xB2)
    return torch.randn(M, N, device=_device(), dtype=dtype)


def _make_mxfp4_weight(N: int, K: int):
    """Build a packed MXFP4 (qweight, scales) pair for an N x K weight."""
    torch.manual_seed(0xC3)
    w = torch.randn(N, K, device=_device(), dtype=torch.bfloat16)
    # MXFP4 block size is 32; pad K up if needed.
    pad = (-K) % 32
    if pad:
        w = F.pad(w, (0, pad))
    packed, scales = pack_mxfp4(w.to(torch.float32), block_size=32)
    return packed, scales, w.shape  # shape is (N, padded_K)


def _make_int4_weight(N: int, K: int):
    """Build a packed INT4 (qweight, scale) pair for an N x K weight."""
    torch.manual_seed(0xD4)
    w = torch.randn(N, K, device=_device(), dtype=torch.float32)
    max_val = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = max_val / 7.0
    qw = torch.clamp((w / scale).round(), -7, 7).to(torch.int8) + 8
    if K % 2:
        qw = F.pad(qw, (0, 1), value=8)
    packed = (qw[:, 0::2] | (qw[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale  # scale is fp32 [N, 1]


# ---------------------------------------------------------------------------
# Runner objects
# ---------------------------------------------------------------------------


@dataclass
class _Runner:
    """Lightweight container that satisfies the kernel-anvil runner contract.

    Attributes are looked up by name (``runner.setup``, ``runner.run``,
    ``runner.reference``, ``runner.BASELINE_CONFIG``, ``runner.DATA_BYTES``).
    """

    setup: Callable[[], dict]
    reference: Callable[[dict], torch.Tensor]
    run: Callable[..., torch.Tensor]
    BASELINE_CONFIG: dict
    DATA_BYTES: int


# Baseline configs match the most common RDNA3 winner from each kernel's
# autotune list. kernel-anvil uses these as the comparison point for
# computing speedup_vs_baseline.
_BASELINE_FWD = {
    "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64,
    "GROUP_M": 8, "num_warps": 8, "num_stages": 3,
}
_BASELINE_BWD = {
    "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64,
    "GROUP_M": 8, "num_warps": 8, "num_stages": 3,
}


# ---------- MXFP4 forward ----------

def _mxfp4_fwd_setup() -> dict:
    M, N, K = _resolve_shape()
    qweight, scales, wshape = _make_mxfp4_weight(N, K)
    x = _make_random_input(M, K, dtype=torch.bfloat16)
    return {
        "x": x, "qweight": qweight, "scales": scales,
        "in_features": K, "M": M, "N": N, "K": K,
        "weight_shape": wshape,
    }


def _mxfp4_fwd_reference(inputs: dict) -> torch.Tensor:
    """Pure-PyTorch dequant + matmul -- the correctness ground truth."""
    w = unpack_mxfp4(
        inputs["qweight"], inputs["scales"], inputs["weight_shape"],
        block_size=32, dtype=torch.bfloat16,
    )
    # Slice padded weight back to the true K
    w = w[:, : inputs["in_features"]].contiguous()
    return F.linear(inputs["x"], w)


def _mxfp4_fwd_run(inputs: dict, **config) -> torch.Tensor:
    from .mxfp4_kernels import triton_mxfp4_matmul
    cfg = {k: v for k, v in config.items() if not k.startswith("_")}
    return triton_mxfp4_matmul(
        inputs["x"], inputs["qweight"], inputs["scales"],
        in_features=inputs["in_features"],
        config=cfg if cfg else None,
    )


def _mxfp4_fwd_data_bytes() -> int:
    M, N, K = _resolve_shape()
    # x: M*K*2 (bf16); qweight: N*K/2 (4bit); scales: N*K/32; out: M*N*2.
    return M * K * 2 + (N * K) // 2 + (N * K) // 32 + M * N * 2


# ---------- MXFP4 grad-input ----------

def _mxfp4_grad_setup() -> dict:
    M, N, K = _resolve_shape()
    qweight, scales, wshape = _make_mxfp4_weight(N, K)
    grad_output = _make_random_grad_output(M, N, dtype=torch.bfloat16)
    return {
        "grad_output": grad_output, "qweight": qweight, "scales": scales,
        "in_features": K, "M": M, "N": N, "K": K,
        "weight_shape": wshape,
    }


def _mxfp4_grad_reference(inputs: dict) -> torch.Tensor:
    w = unpack_mxfp4(
        inputs["qweight"], inputs["scales"], inputs["weight_shape"],
        block_size=32, dtype=torch.bfloat16,
    )
    w = w[:, : inputs["in_features"]].contiguous()
    # grad_input = grad_output @ W
    return inputs["grad_output"] @ w


def _mxfp4_grad_run(inputs: dict, **config) -> torch.Tensor:
    from .mxfp4_kernels import triton_mxfp4_grad_input
    cfg = {k: v for k, v in config.items() if not k.startswith("_")}
    return triton_mxfp4_grad_input(
        inputs["grad_output"], inputs["qweight"], inputs["scales"],
        in_features=inputs["in_features"],
        config=cfg if cfg else None,
    )


def _mxfp4_grad_data_bytes() -> int:
    M, N, K = _resolve_shape()
    return M * N * 2 + (N * K) // 2 + (N * K) // 32 + M * K * 2


# ---------- INT4 forward ----------

def _int4_fwd_setup() -> dict:
    M, N, K = _resolve_shape()
    qweight, scale = _make_int4_weight(N, K)
    x = _make_random_input(M, K, dtype=torch.bfloat16)
    return {
        "x": x, "qweight": qweight, "scale": scale,
        "in_features": K, "M": M, "N": N, "K": K,
    }


def _int4_fwd_reference(inputs: dict) -> torch.Tensor:
    # Mirror bnb_rocm._dequantize_packed -> F.linear path.
    qweight = inputs["qweight"]
    scale = inputs["scale"]
    K = inputs["in_features"]
    out_features = qweight.shape[0]
    low = (qweight & 0x0F).to(torch.bfloat16) - 8.0
    high = ((qweight >> 4) & 0x0F).to(torch.bfloat16) - 8.0
    w = torch.stack([low, high], dim=-1).reshape(out_features, -1)
    w = w[:, :K]
    w = w * scale.to(torch.bfloat16)
    return F.linear(inputs["x"], w)


def _int4_fwd_run(inputs: dict, **config) -> torch.Tensor:
    from .int4_kernels import triton_int4_matmul
    cfg = {k: v for k, v in config.items() if not k.startswith("_")}
    return triton_int4_matmul(
        inputs["x"], inputs["qweight"], inputs["scale"],
        config=cfg if cfg else None,
    )


def _int4_fwd_data_bytes() -> int:
    M, N, K = _resolve_shape()
    # x + qweight (4bit) + scale (fp32 per-row) + out
    return M * K * 2 + (N * K) // 2 + N * 4 + M * N * 2


# ---------- INT4 grad-input ----------

def _int4_grad_setup() -> dict:
    M, N, K = _resolve_shape()
    qweight, scale = _make_int4_weight(N, K)
    grad_output = _make_random_grad_output(M, N, dtype=torch.bfloat16)
    return {
        "grad_output": grad_output, "qweight": qweight, "scale": scale,
        "in_features": K, "M": M, "N": N, "K": K,
    }


def _int4_grad_reference(inputs: dict) -> torch.Tensor:
    qweight = inputs["qweight"]
    scale = inputs["scale"]
    K = inputs["in_features"]
    out_features = qweight.shape[0]
    low = (qweight & 0x0F).to(torch.bfloat16) - 8.0
    high = ((qweight >> 4) & 0x0F).to(torch.bfloat16) - 8.0
    w = torch.stack([low, high], dim=-1).reshape(out_features, -1)
    w = w[:, :K]
    w = w * scale.to(torch.bfloat16)
    # grad_input = grad_output @ W
    return inputs["grad_output"] @ w


def _int4_grad_run(inputs: dict, **config) -> torch.Tensor:
    from .int4_kernels import triton_int4_grad_input
    cfg = {k: v for k, v in config.items() if not k.startswith("_")}
    return triton_int4_grad_input(
        inputs["grad_output"], inputs["qweight"], inputs["scale"],
        in_features=inputs["in_features"],
        config=cfg if cfg else None,
    )


def _int4_grad_data_bytes() -> int:
    M, N, K = _resolve_shape()
    return M * N * 2 + (N * K) // 2 + N * 4 + M * K * 2


# ---------------------------------------------------------------------------
# Public runner instances + module factories
# ---------------------------------------------------------------------------


# Precomputed DATA_BYTES uses the env var values at the time the module
# is imported. The reported number is a hint only; correctness and
# benchmark timing don't depend on it being exact.

mxfp4_fwd_runner = _Runner(
    setup=_mxfp4_fwd_setup,
    reference=_mxfp4_fwd_reference,
    run=_mxfp4_fwd_run,
    BASELINE_CONFIG=dict(_BASELINE_FWD),
    DATA_BYTES=_mxfp4_fwd_data_bytes(),
)

mxfp4_grad_input_runner = _Runner(
    setup=_mxfp4_grad_setup,
    reference=_mxfp4_grad_reference,
    run=_mxfp4_grad_run,
    BASELINE_CONFIG=dict(_BASELINE_BWD),
    DATA_BYTES=_mxfp4_grad_data_bytes(),
)

int4_fwd_runner = _Runner(
    setup=_int4_fwd_setup,
    reference=_int4_fwd_reference,
    run=_int4_fwd_run,
    BASELINE_CONFIG=dict(_BASELINE_FWD),
    DATA_BYTES=_int4_fwd_data_bytes(),
)

int4_grad_input_runner = _Runner(
    setup=_int4_grad_setup,
    reference=_int4_grad_reference,
    run=_int4_grad_run,
    BASELINE_CONFIG=dict(_BASELINE_BWD),
    DATA_BYTES=_int4_grad_data_bytes(),
)


def _make_module(name: str, runner: _Runner) -> types.ModuleType:
    """Wrap a ``_Runner`` as a real module so ``_load_runner`` is happy.

    kernel-anvil's ``_load_runner`` reads ``setup`` / ``reference`` /
    ``run`` / ``BASELINE_CONFIG`` / ``DATA_BYTES`` attributes off the
    imported module. Both module-attribute lookup and dataclass-attribute
    lookup work, but a real module gives us a stable ``__name__`` for
    error logs.
    """
    mod = types.ModuleType(name)
    mod.setup = runner.setup
    mod.reference = runner.reference
    mod.run = runner.run
    mod.BASELINE_CONFIG = dict(runner.BASELINE_CONFIG)
    mod.DATA_BYTES = runner.DATA_BYTES
    return mod


def make_mxfp4_fwd_runner_module() -> types.ModuleType:
    return _make_module("mud_puppy.anvil_runner.mxfp4_fwd", mxfp4_fwd_runner)


def make_mxfp4_grad_input_runner_module() -> types.ModuleType:
    return _make_module("mud_puppy.anvil_runner.mxfp4_grad_input", mxfp4_grad_input_runner)


def make_int4_fwd_runner_module() -> types.ModuleType:
    return _make_module("mud_puppy.anvil_runner.int4_fwd", int4_fwd_runner)


def make_int4_grad_input_runner_module() -> types.ModuleType:
    return _make_module("mud_puppy.anvil_runner.int4_grad_input", int4_grad_input_runner)


# Mapping from op name (matches the anvil-train JSON v1 op keys) to runner.
RUNNERS: dict[str, _Runner] = {
    "mxfp4_fwd": mxfp4_fwd_runner,
    "mxfp4_grad_input": mxfp4_grad_input_runner,
    "int4_fwd": int4_fwd_runner,
    "int4_grad_input": int4_grad_input_runner,
}


# Top-level setup/run/reference shims so this whole module can also be
# fed to ``_load_runner`` directly. The op is selected via the
# ``ANVIL_TRAIN_OP`` env var (defaults to ``mxfp4_fwd``). This is the
# easiest mode for a kernel-anvil CLI invocation:
# ``train-optimize --runner mud_puppy/anvil_runner.py``.

def _selected_runner() -> _Runner:
    op = os.environ.get("ANVIL_TRAIN_OP", "mxfp4_fwd")
    runner = RUNNERS.get(op)
    if runner is None:
        raise ValueError(
            f"ANVIL_TRAIN_OP={op!r} is not a known runner; "
            f"expected one of {list(RUNNERS)}"
        )
    return runner


def setup() -> dict:
    return _selected_runner().setup()


def reference(inputs: dict) -> torch.Tensor:
    return _selected_runner().reference(inputs)


def run(inputs: dict, **config) -> torch.Tensor:
    return _selected_runner().run(inputs, **config)


# These two are looked up at import time by anvil; they reflect whichever
# op was selected when this module was first imported. Re-importing
# (clearing sys.modules) picks them up fresh.
BASELINE_CONFIG: dict = dict(_BASELINE_FWD)
DATA_BYTES: int = _mxfp4_fwd_data_bytes()
try:
    _selected = _selected_runner()
    BASELINE_CONFIG = dict(_selected.BASELINE_CONFIG)
    DATA_BYTES = _selected.DATA_BYTES
except Exception:
    # Default to mxfp4_fwd if the env var is unset or invalid; the
    # selected_runner() call above will surface the real op at runtime.
    pass
