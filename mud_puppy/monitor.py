"""GPU telemetry for the mud-puppy training monitor.

Reads GPU stats (VRAM, utilization, temperature, power) and returns them
as a flat dict.  Works on both ROCm (AMD) and CUDA (NVIDIA) systems, with
graceful degradation when a metric is unavailable.

Priority order:
  1. ``torch.cuda`` for VRAM (works on both CUDA and ROCm via HIP)
  2. ``rocm-smi`` subprocess for util / temp / power (AMD)
  3. ``nvidia-smi`` subprocess for everything (NVIDIA)
  4. Zeros for any metric that cannot be read.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Dict

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TELEMETRY_KEYS = ("vram_used", "vram_total", "gpu_util", "temperature", "power_draw")


def get_gpu_telemetry(device_index: int = 0) -> Dict[str, float]:
    """Return a snapshot of GPU telemetry.

    Returns a dict with the following keys (all ``float``):

    ========== =================================================
    Key        Meaning
    ========== =================================================
    vram_used  VRAM in use (GB)
    vram_total Total VRAM (GB)
    gpu_util   GPU utilisation (0-100 %)
    temperature Junction / die temperature (Celsius)
    power_draw Board power draw (Watts)
    ========== =================================================

    Any metric that cannot be read is returned as ``0.0``.
    """
    result: Dict[str, float] = {k: 0.0 for k in TELEMETRY_KEYS}

    # --- VRAM via torch.cuda (works for both CUDA and ROCm/HIP) -----------
    _read_torch_vram(result, device_index)

    # --- ROCm path --------------------------------------------------------
    if _read_rocm_smi(result, device_index):
        return result

    # --- NVIDIA path (fallback) -------------------------------------------
    _read_nvidia_smi(result, device_index)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_torch_vram(out: Dict[str, float], dev: int) -> bool:
    """Populate vram_used / vram_total from ``torch.cuda``."""
    try:
        import torch  # noqa: delayed import -- torch is heavy

        if not torch.cuda.is_available():
            return False

        dev = min(dev, torch.cuda.device_count() - 1)
        if dev < 0:
            return False

        out["vram_used"] = torch.cuda.memory_allocated(dev) / (1024 ** 3)
        out["vram_total"] = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
        return True
    except Exception as exc:
        log.debug("torch VRAM read failed: %s", exc)
        return False


def _read_rocm_smi(out: Dict[str, float], dev: int) -> bool:
    """Populate util / temp / power (and VRAM if torch missed) from rocm-smi.

    Returns True if rocm-smi ran successfully (even if some fields are
    missing), so the caller knows not to fall through to nvidia-smi.
    """
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showuse", "--showtemp", "--showpower",
             "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode != 0:
            return False

        data = json.loads(proc.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        log.debug("rocm-smi failed: %s", exc)
        return False

    # Pick the right card.  rocm-smi keys are "card0", "card1", etc.
    card_key = f"card{dev}"
    card = data.get(card_key)
    if card is None:
        # Fall back to the first card if the requested index doesn't exist.
        if data:
            card = next(iter(data.values()))
        else:
            return True  # rocm-smi ran but returned nothing useful

    # --- GPU utilisation ---------------------------------------------------
    out["gpu_util"] = _float(card.get("GPU use (%)"))

    # --- Temperature (prefer junction, fall back to edge) ------------------
    temp = card.get("Temperature (Sensor junction) (C)")
    if temp is None:
        temp = card.get("Temperature (Sensor edge) (C)")
    out["temperature"] = _float(temp)

    # --- Power (key name varies between discrete and integrated GPUs) ------
    for key, val in card.items():
        if "Power (W)" in key:
            out["power_draw"] = _float(val)
            break

    # --- VRAM (bytes -> GB) -- only overwrite if torch didn't already ------
    if out["vram_total"] == 0.0:
        total_b = card.get("VRAM Total Memory (B)")
        used_b = card.get("VRAM Total Used Memory (B)")
        if total_b is not None:
            out["vram_total"] = _float(total_b) / (1024 ** 3)
        if used_b is not None:
            out["vram_used"] = _float(used_b) / (1024 ** 3)

    return True


def _read_nvidia_smi(out: Dict[str, float], dev: int) -> bool:
    """Populate all metrics from nvidia-smi (CUDA fallback)."""
    query = (
        "memory.used,memory.total,utilization.gpu,"
        "temperature.gpu,power.draw"
    )
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             f"--id={dev}",
             f"--query-gpu={query}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode != 0:
            return False

        parts = [p.strip() for p in proc.stdout.strip().split(",")]
        if len(parts) < 5:
            return False

        mem_used, mem_total, util, temp, power = parts[:5]

        # nvidia-smi reports memory in MiB.
        if out["vram_used"] == 0.0:
            out["vram_used"] = _float(mem_used) / 1024
        if out["vram_total"] == 0.0:
            out["vram_total"] = _float(mem_total) / 1024
        out["gpu_util"] = _float(util)
        out["temperature"] = _float(temp)
        out["power_draw"] = _float(power)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log.debug("nvidia-smi failed: %s", exc)
        return False


def _float(val) -> float:
    """Safely cast to float, returning 0.0 on failure."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
