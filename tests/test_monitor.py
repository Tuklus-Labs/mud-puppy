"""Tests for mud_puppy.monitor -- GPU telemetry."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Load monitor.py directly so we don't trigger mud_puppy/__init__.py
# (which pulls in torch/transformers/etc. that the test venv may lack).
_MONITOR_PATH = Path(__file__).resolve().parent.parent / "mud_puppy" / "monitor.py"
_spec = importlib.util.spec_from_file_location("mud_puppy.monitor", _MONITOR_PATH)
_monitor = importlib.util.module_from_spec(_spec)
sys.modules["mud_puppy.monitor"] = _monitor
_spec.loader.exec_module(_monitor)

get_gpu_telemetry = _monitor.get_gpu_telemetry
TELEMETRY_KEYS = _monitor.TELEMETRY_KEYS


# ------------------------------------------------------------------
# Required tests (from task spec)
# ------------------------------------------------------------------

def test_get_gpu_telemetry_returns_dict():
    """get_gpu_telemetry() returns a dict with all 5 expected keys."""
    result = get_gpu_telemetry()
    assert isinstance(result, dict)
    for key in TELEMETRY_KEYS:
        assert key in result, f"Missing key: {key}"


def test_get_gpu_telemetry_values_are_numeric():
    """All values in the telemetry dict are int or float."""
    result = get_gpu_telemetry()
    for key, value in result.items():
        assert isinstance(value, (int, float)), (
            f"Key {key!r} has non-numeric value {value!r} (type {type(value).__name__})"
        )


# ------------------------------------------------------------------
# Fallback / edge-case tests
# ------------------------------------------------------------------

def test_returns_zeros_when_no_gpu():
    """When torch.cuda is unavailable and both SMI tools are missing,
    every value should be 0.0."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": mock_torch}), \
         patch("subprocess.run", side_effect=FileNotFoundError):
        result = get_gpu_telemetry()

    assert all(v == 0.0 for v in result.values()), result


def test_rocm_smi_parsing():
    """Verify that _read_rocm_smi correctly parses rocm-smi JSON output."""
    fake_json = {
        "card0": {
            "Temperature (Sensor junction) (C)": "78.0",
            "Temperature (Sensor edge) (C)": "62.0",
            "Average Graphics Package Power (W)": "250.0",
            "GPU use (%)": "95",
            "VRAM Total Memory (B)": str(24 * 1024**3),
            "VRAM Total Used Memory (B)": str(12 * 1024**3),
        }
    }

    out = {k: 0.0 for k in TELEMETRY_KEYS}
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = __import__("json").dumps(fake_json)

    with patch("subprocess.run", return_value=mock_proc):
        ok = _monitor._read_rocm_smi(out, 0)

    assert ok is True
    assert out["gpu_util"] == 95.0
    assert out["temperature"] == 78.0  # junction preferred over edge
    assert out["power_draw"] == 250.0
    assert abs(out["vram_total"] - 24.0) < 0.01
    assert abs(out["vram_used"] - 12.0) < 0.01


def test_nvidia_smi_parsing():
    """Verify that _read_nvidia_smi correctly parses nvidia-smi CSV output."""
    out = {k: 0.0 for k in TELEMETRY_KEYS}
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "8192, 24576, 87, 72, 310.5\n"

    with patch("subprocess.run", return_value=mock_proc):
        ok = _monitor._read_nvidia_smi(out, 0)

    assert ok is True
    assert abs(out["vram_used"] - 8.0) < 0.01       # 8192 MiB -> 8 GB
    assert abs(out["vram_total"] - 24.0) < 0.01      # 24576 MiB -> 24 GB
    assert out["gpu_util"] == 87.0
    assert out["temperature"] == 72.0
    assert out["power_draw"] == 310.5
