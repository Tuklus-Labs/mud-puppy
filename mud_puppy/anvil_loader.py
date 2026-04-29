"""Loader for ``anvil-train`` JSON v1 kernel-config caches.

``kernel-anvil`` runs an offline sweep against the mud-puppy training
kernels and writes the winning Triton config per (op, M-bucket, N-bucket,
K-bucket) cell into a JSON file under
``~/.cache/anvil-train/<gpu_gfx>/<model_basename>-<quant>-b<B>s<S>.json``.
At training time, this module loads that JSON, validates it against the
current kernel source (sha256 of the kernel module), and exposes a
``get_kernel_config(op, M, N, K)`` lookup that callers in the kernel
wrappers can use to bypass ``@triton.autotune``.

JSON contract (v1, frozen by docs/plans/2026-04-28-kernel-anvil-training-impl.md):

.. code-block:: json

    {
      "schema": "anvil-train/v1",
      "gpu": "gfx1100",
      "rocm_version": "7.1",
      "torch_version": "2.10.0+rocm7.1",
      "triton_version": "3.6.0",
      "kernel_hash": "sha256:<hex>",
      "model": "Qwen3-8B",
      "batch": 1,
      "seq": 4096,
      "ops": {
        "mxfp4_fwd": {
          "<m_bucket>,<n_bucket>,<k_bucket>": {
            "BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
            "GROUP_M": 8, "num_warps": 8, "num_stages": 4
          }
        }
      }
    }

Bucket scheme: 5 buckets per axis (0..4), boundaries
``(1024, 4096, 8192, 16384)`` on each of M, N, K. ``bucket_index`` returns
0 for values <= 1024, 1 for 1025..4096, 2 for 4097..8192, 3 for
8193..16384, 4 for >16384. Lookups that miss a cell return ``None`` so
the caller falls through to the existing ``@triton.autotune`` path.

Cache invalidation: any change to ``kernel_hash`` (computed over the
content of ``mxfp4_kernels.py`` and ``int4_kernels.py``) means the JSON
was tuned for a different kernel and is silently dropped with a warning.
The same applies for unparseable JSON or wrong schema strings -- the
loader never raises, it just degrades to ``None``.

The loader memoizes parsed JSONs keyed by absolute path, so repeated
``load_for_model(...)`` calls in a long training run don't re-parse.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bucket scheme (3D, 5x5x5 cells)
# ---------------------------------------------------------------------------

# Boundaries are inclusive upper bounds. A value <= 1024 is bucket 0;
# 1025..4096 is bucket 1; 4097..8192 is bucket 2; 8193..16384 is bucket 3;
# anything larger is bucket 4. Same boundaries on all three axes (M, N, K).
BUCKET_BOUNDARIES = (1024, 4096, 8192, 16384)
NUM_BUCKETS = 5

# Op names that anvil-train JSON keys are valid for. Loader is permissive
# and accepts any op name, but the wrappers only consult these four.
VALID_OPS = (
    "mxfp4_fwd",
    "mxfp4_grad_input",
    "int4_fwd",
    "int4_grad_input",
)

# Schema version this loader understands. Anything else is rejected with
# a warning so a v2 JSON in a v1 loader doesn't silently misinterpret.
SCHEMA_V1 = "anvil-train/v1"


def bucket_index(value: int) -> int:
    """Return the 0..4 bucket index for ``value`` along any axis.

    See ``BUCKET_BOUNDARIES``. Negative or zero values bucket to 0; very
    large values cap at 4. Non-integer inputs are coerced to int.
    """
    v = int(value)
    for i, boundary in enumerate(BUCKET_BOUNDARIES):
        if v <= boundary:
            return i
    return NUM_BUCKETS - 1


def _bucket_key(M: int, N: int, K: int) -> str:
    """Build the ``"m,n,k"`` string used as the per-cell JSON key."""
    return f"{bucket_index(M)},{bucket_index(N)},{bucket_index(K)}"


# ---------------------------------------------------------------------------
# Kernel hash
# ---------------------------------------------------------------------------

# Files whose content participates in the kernel hash. If any of these
# files change, the cached JSON is considered stale. We hash the source
# bytes directly so the loader does not need to import the kernel module
# (avoids a circular import in CI environments without Triton).
_KERNEL_SOURCE_FILES = ("mxfp4_kernels.py", "int4_kernels.py")


def _kernel_source_dir() -> Path:
    """Resolve ``mud_puppy/`` from this file's location.

    ``__file__`` lives in ``.../mud_puppy/anvil_loader.py``; the parent
    directory holds the kernel source files we hash.
    """
    return Path(__file__).resolve().parent


def compute_kernel_hash() -> str:
    """Compute the canonical sha256 of the kernel source files.

    Returns a string like ``"sha256:<hex>"`` so that string compares
    against the JSON's ``kernel_hash`` field (which uses the same
    prefix) work directly. If a kernel file is missing for any reason
    (truncated install, dev override), the missing file is hashed as
    its absolute path string so two missing-file states aren't
    accidentally considered equivalent.
    """
    h = hashlib.sha256()
    src_dir = _kernel_source_dir()
    for name in _KERNEL_SOURCE_FILES:
        p = src_dir / name
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        try:
            h.update(p.read_bytes())
        except OSError:
            # Hash the path itself so two missing files don't collide.
            h.update(b"<missing:")
            h.update(str(p).encode("utf-8"))
            h.update(b">")
        h.update(b"\x00")
    return f"sha256:{h.hexdigest()}"


# ---------------------------------------------------------------------------
# AnvilTrainConfig
# ---------------------------------------------------------------------------


# Per-process cache: absolute path -> AnvilTrainConfig (or None for invalid).
# Keeps the parsed JSON in memory for the lifetime of the process so a
# long training run doesn't re-read the same file thousands of times.
_LOAD_CACHE: dict[str, Optional["AnvilTrainConfig"]] = {}


class AnvilTrainConfig:
    """Parsed anvil-train JSON v1 with bucketed config lookup.

    Construct via ``AnvilTrainConfig.load(path)`` -- it handles caching,
    hash validation, and malformed-JSON fallback. The constructor itself
    takes the already-validated dict so unit tests can inject fixtures
    without going through the filesystem.

    The object is read-only after construction. ``get_kernel_config`` is
    the only hot-path method; everything else is metadata for diagnostics.
    """

    def __init__(self, payload: dict, source_path: Optional[str] = None):
        self._payload = payload
        self.source_path = source_path
        # Convenience accessors (all optional with sensible defaults).
        self.schema: str = payload.get("schema", "")
        self.gpu: str = payload.get("gpu", "")
        self.rocm_version: str = payload.get("rocm_version", "")
        self.torch_version: str = payload.get("torch_version", "")
        self.triton_version: str = payload.get("triton_version", "")
        self.kernel_hash: str = payload.get("kernel_hash", "")
        self.model: str = payload.get("model", "")
        self.batch: int = int(payload.get("batch", 0) or 0)
        self.seq: int = int(payload.get("seq", 0) or 0)
        # ops: {op_name: {"m,n,k": {...config...}}}
        ops_raw = payload.get("ops", {})
        if not isinstance(ops_raw, dict):
            ops_raw = {}
        self._ops: dict[str, dict[str, dict]] = {
            str(k): (v if isinstance(v, dict) else {}) for k, v in ops_raw.items()
        }

    # ------------------------------------------------------------------
    # Construction / parsing
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | os.PathLike) -> Optional["AnvilTrainConfig"]:
        """Load and validate an anvil-train JSON v1 file.

        Returns ``None`` (and logs a warning) on any of:
          * file does not exist
          * file is not valid JSON
          * top-level ``schema`` is not ``anvil-train/v1``
          * ``kernel_hash`` does not match the current kernel source

        On success, the result is memoized so repeated calls with the
        same absolute path do not re-read or re-parse.
        """
        abs_path = str(Path(path).resolve())
        if abs_path in _LOAD_CACHE:
            return _LOAD_CACHE[abs_path]

        result = cls._load_uncached(abs_path)
        _LOAD_CACHE[abs_path] = result
        return result

    @classmethod
    def _load_uncached(cls, abs_path: str) -> Optional["AnvilTrainConfig"]:
        p = Path(abs_path)
        if not p.exists():
            log.debug("anvil-train: cache miss (no file at %s)", abs_path)
            return None
        try:
            with p.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "anvil-train: failed to parse %s (%s); falling back to "
                "@triton.autotune defaults",
                abs_path, exc,
            )
            return None

        if not isinstance(payload, dict):
            log.warning(
                "anvil-train: %s top-level is not a JSON object; ignoring",
                abs_path,
            )
            return None

        schema = payload.get("schema", "")
        if schema != SCHEMA_V1:
            log.warning(
                "anvil-train: %s schema=%r unsupported (expected %r); "
                "falling back to @triton.autotune defaults",
                abs_path, schema, SCHEMA_V1,
            )
            return None

        # Validate kernel_hash against current source. If absent or
        # mismatched, treat as cache miss -- the JSON was tuned for a
        # different kernel build.
        expected_hash = compute_kernel_hash()
        actual_hash = payload.get("kernel_hash", "")
        if actual_hash != expected_hash:
            log.warning(
                "anvil-train: %s kernel_hash mismatch (cache=%s, current=%s); "
                "configs will not be applied. Re-run kernel-anvil train-optimize.",
                abs_path, actual_hash, expected_hash,
            )
            return None

        return cls(payload, source_path=abs_path)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_kernel_config(
        self, op: str, M: int, N: int, K: int,
    ) -> Optional[dict]:
        """Look up the per-cell config for (op, M, N, K).

        Returns the raw config dict (with BLOCK_M/N/K, GROUP_M, num_warps,
        num_stages keys) or ``None`` if either the op or the cell is
        absent. Diagnostic-only fields like ``speedup_vs_baseline`` and
        ``profiled_us`` are kept in the dict; the kernel wrappers ignore
        anything they don't recognize.

        ``M``, ``N``, ``K`` may be any int-like values; they are cast to
        int before bucketing. Lookup is case-sensitive on op names.
        """
        op_table = self._ops.get(op)
        if not op_table:
            return None
        cell_key = _bucket_key(M, N, K)
        cfg = op_table.get(cell_key)
        if cfg is None or not isinstance(cfg, dict):
            return None
        return cfg

    def num_configs(self) -> int:
        """Total cells across all ops. Useful for the startup log line."""
        return sum(len(table) for table in self._ops.values())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AnvilTrainConfig(model={self.model!r}, gpu={self.gpu!r}, "
            f"batch={self.batch}, seq={self.seq}, "
            f"ops={list(self._ops)}, cells={self.num_configs()}, "
            f"path={self.source_path!r})"
        )


# ---------------------------------------------------------------------------
# Convenience: cache-path resolution
# ---------------------------------------------------------------------------


def _gpu_gfx_target() -> str:
    """Return the ``gfxNNNN`` target for the current device, or ``""``.

    Uses ``mud_puppy.arch`` so we don't string-match gfx targets
    elsewhere. On CPU / no-GPU systems, returns the empty string and the
    caller falls back to no-cache behaviour.
    """
    try:
        from . import arch  # local import; avoid hard dep at module load
        info = arch.get_arch()
        return info.gfx_target or ""
    except Exception:  # pragma: no cover - defensive
        return ""


def cache_path_for_model(
    model_id: str, batch: int, seq: int, quant: str,
    gpu_gfx: Optional[str] = None,
) -> Path:
    """Build the canonical cache path for a (model, B, S, quant) tuple.

    Mirrors ``~/.cache/anvil-train/<gpu>/<model_basename>-<quant>-b<B>s<S>.json``.
    ``model_id`` may be a HuggingFace repo id (with slashes) or a local
    path; only the trailing component is used as the file basename so
    siblings on disk stay readable.
    """
    gfx = gpu_gfx if gpu_gfx is not None else _gpu_gfx_target()
    # Normalize: strip path separators and anything after a colon (HF revision).
    base = os.path.basename(str(model_id).rstrip("/"))
    base = base.split(":", 1)[0] or "unknown-model"
    fname = f"{base}-{quant}-b{int(batch)}s{int(seq)}.json"
    cache_dir = Path(os.path.expanduser("~/.cache/anvil-train"))
    if gfx:
        cache_dir = cache_dir / gfx
    return cache_dir / fname


def load_for_model(
    model_id: str,
    batch: int,
    seq: int,
    quant: str,
    gpu_gfx: Optional[str] = None,
) -> Optional[AnvilTrainConfig]:
    """Look up the anvil-train JSON for ``(model, batch, seq, quant)``.

    Returns ``None`` if the cache file does not exist, fails to parse,
    or has a mismatched kernel hash. ``quant`` should be one of
    ``"mxfp4"`` or ``"int4"`` -- it's used as the filename slug only.

    Callers should call this once at module/layer construction time and
    stash the returned object on the layer; per-step lookups go through
    :meth:`AnvilTrainConfig.get_kernel_config`.
    """
    path = cache_path_for_model(model_id, batch, seq, quant, gpu_gfx=gpu_gfx)
    return AnvilTrainConfig.load(path)


# ---------------------------------------------------------------------------
# Test / development helpers
# ---------------------------------------------------------------------------


def clear_cache() -> None:
    """Drop the in-process load cache. Used by tests to avoid leaks."""
    _LOAD_CACHE.clear()


# ---------------------------------------------------------------------------
# Top-level integration entrypoint
# ---------------------------------------------------------------------------


def apply_to_model(
    model,  # nn.Module
    model_id: str,
    batch: int,
    seq: int,
    quant: str,
    gpu_gfx: Optional[str] = None,
) -> int:
    """One-call integration helper for trainers.

    Loads the cached anvil-train JSON for ``(model_id, batch, seq, quant)``
    if any, then walks ``model`` populating per-layer config slots on
    every :class:`MXFP4Linear` and :class:`Linear4bit`. Logs a single
    startup line ``"mud-puppy: applied N anvil-train configs for <model>"``
    when at least one slot is populated, and ``0`` when no JSON was
    found (caller can branch on the return value).

    Returns the count of (layer, op) slots actually populated. ``0``
    means the kernels stay on the existing ``@triton.autotune`` path.

    The lookup is done with ``M = batch * seq`` because anvil-train tunes
    per (model, B, S) tuple.
    """
    cfg = load_for_model(model_id, batch, seq, quant, gpu_gfx=gpu_gfx)
    if cfg is None:
        log.info(
            "mud-puppy: no anvil-train config for %s (b=%d s=%d quant=%s); "
            "using @triton.autotune defaults",
            model_id, batch, seq, quant,
        )
        return 0

    M = int(batch) * int(seq)
    total = 0
    if quant == "mxfp4":
        from . import mxfp4_kernels
        total += mxfp4_kernels.apply_anvil_configs(model, cfg, M)
    elif quant == "int4":
        from . import bnb_rocm
        total += bnb_rocm.apply_anvil_configs(model, cfg, M)
    else:
        # Unknown quant slug -- try both layer types so callers using a
        # mixed quant scheme still pick up whatever cells matched.
        from . import mxfp4_kernels, bnb_rocm
        total += mxfp4_kernels.apply_anvil_configs(model, cfg, M)
        total += bnb_rocm.apply_anvil_configs(model, cfg, M)

    if total > 0:
        log.info(
            "mud-puppy: applied %d anvil-train configs for %s "
            "(b=%d s=%d quant=%s, source=%s)",
            total, model_id, batch, seq, quant, cfg.source_path,
        )
    return total
