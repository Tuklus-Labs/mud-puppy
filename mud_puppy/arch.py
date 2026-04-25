"""GPU architecture detection and capability queries.

Central helper for kernels, precision selectors, and device-specific
fallbacks. Distinguishes RDNA3/4 (desktop) from CDNA2/3 (datacenter) so
Triton autotune tables can target the right tile sizes and call sites
can branch on hardware FP8 / MFMA availability without open-coding
string matches across the codebase.

All queries are cached per-device on first call. The probe does NOT
require the GPU to be "warm" -- it only inspects reported properties
and, for the FP8 hardware probe, performs a single small scaled_mm.

Hardware matrix
~~~~~~~~~~~~~~~

+-------+------+----------------+-----------+--------+--------+
| Arch  | gfx  | Examples       | Wavefront | Matrix | HW FP8 |
+=======+======+================+===========+========+========+
| RDNA2 | 1030 | 6900 XT        | 32        | none   | no     |
| RDNA3 | 1100 | 7900 XTX/XT    | 32        | WMMA   | no     |
| RDNA4 | 1200 | 9070 XT        | 32        | WMMA   | yes    |
| CDNA1 | 908  | MI100          | 64        | MFMA   | no     |
| CDNA2 | 90a  | MI210, MI250X  | 64        | MFMA   | no     |
| CDNA3 | 940+ | MI300X, MI300A | 64        | MFMA   | yes    |
+-------+------+----------------+-----------+--------+--------+

On CUDA or CPU the detector returns an ``UNKNOWN`` arch; callers that
special-case AMD should check ``is_cdna()`` / ``is_rdna()`` rather than
"is_amd" so the code does the right thing on mixed or future hardware.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

log = logging.getLogger(__name__)


class ArchFamily(Enum):
    """Top-level GPU family.

    The distinction that matters most in practice is CDNA (datacenter,
    wavefront=64, MFMA, HBM) vs RDNA (desktop, wavefront=32, WMMA,
    GDDR). NVIDIA and unknown devices are separate so AMD-specific
    fallbacks can short-circuit cleanly.
    """

    UNKNOWN = "unknown"
    # RDNA1 (gfx101x, 5700 XT etc.) predates WMMA; queries for matrix
    # cores must answer False for RDNA1 the same as for RDNA2. Kept as a
    # distinct family so ``family == RDNA2`` doesn't lie for a 5700 XT.
    RDNA1 = "rdna1"
    RDNA2 = "rdna2"
    RDNA3 = "rdna3"
    RDNA4 = "rdna4"
    # APU variants such as Renoir/Cezanne (gfx90c) look superficially
    # like CDNA ("gfx9xx") but are integrated RDNA-lineage graphics with
    # 32-wide wavefronts and no matrix cores / HBM / hw-fp8.
    APU = "apu"
    CDNA1 = "cdna1"
    CDNA2 = "cdna2"
    CDNA3 = "cdna3"
    NVIDIA = "nvidia"


@dataclass(frozen=True)
class ArchInfo:
    """Everything kernels and precision selectors need to know.

    The booleans are derived from the family so call sites should prefer
    reading ``has_hw_fp8`` over ``family == CDNA3 or RDNA4``.
    """

    family: ArchFamily
    gfx_target: str                 # e.g. "gfx1100", "gfx942"; "" if not AMD
    device_name: str                # raw torch device name, for logging
    wavefront_size: int             # 32 on RDNA, 64 on CDNA, 32 on NVIDIA (warp)
    has_matrix_cores: bool          # WMMA or MFMA available
    has_hw_fp8: bool                # native FP8 via _scaled_mm fast path
    has_hbm: bool                   # influences memory-bound kernel tuning
    cu_count: int                   # compute unit / SM count; 0 if unknown
    mem_bw_gbs_hint: float          # rough peak DRAM bandwidth in GB/s

    @property
    def is_cdna(self) -> bool:
        return self.family in (ArchFamily.CDNA1, ArchFamily.CDNA2, ArchFamily.CDNA3)

    @property
    def is_rdna(self) -> bool:
        return self.family in (
            ArchFamily.RDNA1,
            ArchFamily.RDNA2,
            ArchFamily.RDNA3,
            ArchFamily.RDNA4,
        )

    @property
    def is_amd(self) -> bool:
        return self.is_cdna or self.is_rdna

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.family.value}({self.gfx_target or self.device_name}, "
            f"wave={self.wavefront_size}, mfma={self.has_matrix_cores}, "
            f"hw_fp8={self.has_hw_fp8}, hbm={self.has_hbm}, cu={self.cu_count})"
        )


# Cache keyed by device index. A system can mix GPUs (uncommon for us,
# but possible on a rental droplet), so we detect per-device.
_CACHE: dict[int, ArchInfo] = {}

# Unknown/default info for CPU or no-GPU runs. Safe zeros so callers can
# read any field without branching on "is this even a GPU".
_UNKNOWN = ArchInfo(
    family=ArchFamily.UNKNOWN,
    gfx_target="",
    device_name="cpu",
    wavefront_size=32,
    has_matrix_cores=False,
    has_hw_fp8=False,
    has_hbm=False,
    cu_count=0,
    mem_bw_gbs_hint=0.0,
)


def _gfx_to_family(gfx: str) -> ArchFamily:
    """Map a ``gfx<n>`` string to an :class:`ArchFamily`.

    Handles the common targets mud-puppy cares about; unknown strings
    return UNKNOWN so the caller falls back to portable paths.
    """
    g = gfx.lower()
    if not g.startswith("gfx"):
        return ArchFamily.UNKNOWN
    # APU codes (Renoir/Cezanne and relatives): gfx90c is an integrated
    # RDNA-lineage part with wavefront 32, no MFMA, no HBM. It MUST NOT
    # be classified as CDNA (would give it matrix cores + HBM falsely).
    # We match on the exact literal string so numeric parsing below
    # doesn't accidentally put it in the CDNA2 bucket (n=90).
    if g == "gfx90c":
        return ArchFamily.APU
    # Strip any trailing feature letters (e.g. gfx90a, gfx942 variants).
    # We match the numeric prefix.
    num_part = "".join(ch for ch in g[3:] if ch.isdigit())
    if not num_part:
        return ArchFamily.UNKNOWN
    try:
        n = int(num_part)
    except ValueError:
        return ArchFamily.UNKNOWN

    # RDNA generations.
    if 1010 <= n <= 1019:
        # RDNA1 (gfx101x, Navi 10/14, e.g. 5700 XT). No matrix cores.
        return ArchFamily.RDNA1
    if 1030 <= n <= 1039:
        return ArchFamily.RDNA2
    if 1100 <= n <= 1199:
        return ArchFamily.RDNA3
    if 1200 <= n <= 1299:
        return ArchFamily.RDNA4
    # CDNA generations. Note: gfx90a parses to n=90 because isdigit
    # strips the 'a' suffix; handle that case explicitly and require
    # the literal suffixed string so we don't accidentally match gfx90c
    # (which is handled above as an APU).
    if n == 908:
        return ArchFamily.CDNA1
    if g == "gfx90a":
        return ArchFamily.CDNA2
    if 940 <= n <= 949:
        return ArchFamily.CDNA3
    return ArchFamily.UNKNOWN


def _mem_bw_hint(family: ArchFamily) -> float:
    """Rough peak DRAM bandwidth in GB/s.

    Used as a tie-breaker when picking kernel configs: memory-bound
    kernels benefit from larger BLOCK_K on HBM parts because latency
    hiding is cheaper. Values are approximate; kernel autotune still
    gets the last word.
    """
    return {
        ArchFamily.RDNA1: 448.0,    # 5700 XT (GDDR6)
        ArchFamily.RDNA2: 512.0,
        ArchFamily.RDNA3: 960.0,    # 7900 XTX
        ArchFamily.RDNA4: 900.0,    # 9070 XT est.
        ArchFamily.APU: 50.0,       # DDR4/DDR5 system mem; rough hint
        ArchFamily.CDNA1: 1200.0,   # MI100
        ArchFamily.CDNA2: 1640.0,   # MI250X
        ArchFamily.CDNA3: 5300.0,   # MI300X (HBM3)
        ArchFamily.NVIDIA: 2000.0,  # generic; actual depends on SM gen
        ArchFamily.UNKNOWN: 0.0,
    }[family]


def _detect_amd(device_idx: int) -> Optional[ArchInfo]:
    """Build an :class:`ArchInfo` for an AMD device, or None if not AMD."""
    if getattr(torch.version, "hip", None) is None:
        return None
    try:
        props = torch.cuda.get_device_properties(device_idx)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("get_device_properties(%d) failed: %s", device_idx, exc)
        return None

    # PyTorch ROCm exposes the gfx target via ``gcnArchName``. Older
    # builds called it ``gcn_arch_name``; check both.
    gfx = (
        getattr(props, "gcnArchName", None)
        or getattr(props, "gcn_arch_name", None)
        or ""
    )
    gfx = gfx.strip()
    device_name = getattr(props, "name", "") or "amd-gpu"

    # gcnArchName may include feature suffixes like "gfx90a:sramecc+:xnack-".
    # Keep only the first token.
    gfx_clean = gfx.split(":", 1)[0] if gfx else ""
    family = _gfx_to_family(gfx_clean)

    wavefront = 64 if family in (
        ArchFamily.CDNA1,
        ArchFamily.CDNA2,
        ArchFamily.CDNA3,
    ) else 32

    # RDNA1, RDNA2, APU, and UNKNOWN all lack matrix cores (WMMA/MFMA).
    has_matrix = family not in (
        ArchFamily.RDNA1,
        ArchFamily.RDNA2,
        ArchFamily.APU,
        ArchFamily.UNKNOWN,
    )
    has_hbm = family in (ArchFamily.CDNA1, ArchFamily.CDNA2, ArchFamily.CDNA3)
    cu = int(getattr(props, "multi_processor_count", 0) or 0)
    has_hw_fp8 = family in (ArchFamily.CDNA3, ArchFamily.RDNA4)

    return ArchInfo(
        family=family,
        gfx_target=gfx_clean,
        device_name=device_name,
        wavefront_size=wavefront,
        has_matrix_cores=has_matrix,
        has_hw_fp8=has_hw_fp8,
        has_hbm=has_hbm,
        cu_count=cu,
        mem_bw_gbs_hint=_mem_bw_hint(family),
    )


def _detect_nvidia(device_idx: int) -> Optional[ArchInfo]:
    """Build an :class:`ArchInfo` for an NVIDIA device, or None if not CUDA."""
    if getattr(torch.version, "hip", None) is not None:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(device_idx)
    except Exception:  # pragma: no cover
        return None
    # Hopper (SM90) and Blackwell (SM100) ship hardware FP8.
    major = int(getattr(props, "major", 0) or 0)
    has_hw_fp8 = major >= 9
    cu = int(getattr(props, "multi_processor_count", 0) or 0)
    return ArchInfo(
        family=ArchFamily.NVIDIA,
        gfx_target=f"sm_{major}{getattr(props, 'minor', 0)}",
        device_name=getattr(props, "name", "nvidia-gpu"),
        wavefront_size=32,
        has_matrix_cores=major >= 7,  # Volta+ has tensor cores
        has_hw_fp8=has_hw_fp8,
        has_hbm=True,  # most datacenter NVIDIA is HBM; rough hint anyway
        cu_count=cu,
        mem_bw_gbs_hint=_mem_bw_hint(ArchFamily.NVIDIA),
    )


def get_arch(device: Optional[torch.device | int | str] = None) -> ArchInfo:
    """Return cached :class:`ArchInfo` for the selected device.

    ``device`` may be a ``torch.device``, an int index, or ``None`` (uses
    the current CUDA device if any). CPU / no-GPU systems return the
    UNKNOWN singleton.
    """
    # Allow overriding detection for tests and for droplets where the
    # reported gcnArchName is weirdly blank.
    override = os.environ.get("MUD_PUPPY_ARCH_OVERRIDE", "").strip().lower()
    if override:
        return _from_override(override)

    if not torch.cuda.is_available():
        return _UNKNOWN

    if device is None:
        idx = torch.cuda.current_device()
    elif isinstance(device, int):
        idx = device
    elif isinstance(device, str):
        idx = torch.device(device).index or 0
    else:
        idx = device.index if device.index is not None else 0

    if idx in _CACHE:
        return _CACHE[idx]

    info = _detect_amd(idx) or _detect_nvidia(idx) or _UNKNOWN
    _CACHE[idx] = info
    log.info("mud-puppy arch detected (dev=%d): %s", idx, info)
    return info


def _from_override(token: str) -> ArchInfo:
    """Build an ArchInfo from ``MUD_PUPPY_ARCH_OVERRIDE`` (test helper).

    Accepts a family name (``rdna3``, ``cdna3``) or a gfx string
    (``gfx942``). Everything else is derived from the family.
    """
    # Try gfx first so gfx1100 -> RDNA3 etc.
    if token.startswith("gfx"):
        family = _gfx_to_family(token)
        gfx = token
    else:
        try:
            family = ArchFamily(token)
        except ValueError:
            family = ArchFamily.UNKNOWN
        gfx = {
            ArchFamily.RDNA1: "gfx1010",
            ArchFamily.RDNA2: "gfx1030",
            ArchFamily.RDNA3: "gfx1100",
            ArchFamily.RDNA4: "gfx1200",
            ArchFamily.APU: "gfx90c",
            ArchFamily.CDNA1: "gfx908",
            ArchFamily.CDNA2: "gfx90a",
            ArchFamily.CDNA3: "gfx942",
            ArchFamily.NVIDIA: "sm_90",
            ArchFamily.UNKNOWN: "",
        }[family]

    is_cdna = family in (ArchFamily.CDNA1, ArchFamily.CDNA2, ArchFamily.CDNA3)
    return ArchInfo(
        family=family,
        gfx_target=gfx,
        device_name=f"override:{token}",
        wavefront_size=64 if is_cdna else 32,
        has_matrix_cores=family not in (
            ArchFamily.RDNA1,
            ArchFamily.RDNA2,
            ArchFamily.APU,
            ArchFamily.UNKNOWN,
        ),
        has_hw_fp8=family in (ArchFamily.CDNA3, ArchFamily.RDNA4)
        or (family == ArchFamily.NVIDIA),
        has_hbm=is_cdna or family == ArchFamily.NVIDIA,
        cu_count=0,
        mem_bw_gbs_hint=_mem_bw_hint(family),
    )


def clear_cache() -> None:
    """Drop the per-device cache (used in tests)."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# FP8 hardware probe
# ---------------------------------------------------------------------------

# The arch table above says "CDNA3 and RDNA4 have hardware FP8." That is
# the answer for kernel tuning. But a *runnable* FP8 path also requires
# the installed PyTorch build to actually dispatch ``torch._scaled_mm``
# to the native kernel -- which can be missing on older ROCm releases
# even when the hardware supports it. ``is_fp8_runnable`` does the
# runtime probe; kernel tuning should use ``info.has_hw_fp8``.

_FP8_PROBE_CACHE: dict[int, bool] = {}


def is_fp8_runnable(device: Optional[torch.device | int | str] = None) -> bool:
    """True iff ``torch._scaled_mm`` works on this device.

    Runtime probe: attempts a 16x32 @ 32x16 scaled_mm in FP8 E4M3 and
    returns False on any exception. Cached per device to keep the cost
    to ~2 ms on first call.
    """
    info = get_arch(device)
    if not info.has_hw_fp8:
        return False
    if not (hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn")):
        return False
    if not torch.cuda.is_available():
        return False

    idx = torch.cuda.current_device() if device is None else (
        device if isinstance(device, int) else torch.device(device).index or 0
    )
    if idx in _FP8_PROBE_CACHE:
        return _FP8_PROBE_CACHE[idx]
    try:
        a = torch.zeros(16, 32, device=f"cuda:{idx}", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        b = torch.zeros(32, 16, device=f"cuda:{idx}", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        sa = torch.tensor(1.0, device=f"cuda:{idx}")
        sb = torch.tensor(1.0, device=f"cuda:{idx}")
        torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        ok = True
    except Exception as exc:  # pragma: no cover - hardware-dependent
        log.debug("FP8 runtime probe failed on dev %d: %s", idx, exc)
        ok = False
    _FP8_PROBE_CACHE[idx] = ok
    return ok


# ---------------------------------------------------------------------------
# Kernel tuning hints
# ---------------------------------------------------------------------------

def recommended_num_warps(info: ArchInfo, block_m: int, block_n: int) -> int:
    """Suggest a ``num_warps`` for a given tile size.

    CDNA wavefronts are 64-wide, so a tile that wants 256 threads uses
    num_warps=4 (vs num_warps=8 on RDNA3 for the same thread count).
    Triton's autotune usually finds this on its own, but the hint is
    useful for seed configs.
    """
    threads = max(block_m, block_n)  # rough occupancy floor
    warps = max(2, threads // info.wavefront_size)
    # Keep to powers of two that Triton accepts.
    for w in (2, 4, 8, 16):
        if warps <= w:
            return w
    return 16


def recommended_num_stages(info: ArchInfo) -> int:
    """Suggest a ``num_stages`` for pipelined kernels.

    HBM parts (CDNA, datacenter NVIDIA) benefit from deeper pipelining
    because DRAM latency is higher relative to compute. GDDR parts want
    2 stages to leave register budget for larger tiles.
    """
    if info.has_hbm:
        return 3
    return 2
