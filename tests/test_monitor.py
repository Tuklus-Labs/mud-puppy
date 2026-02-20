"""Tests for mud_puppy.monitor -- GPU telemetry + WebSocket server + callback."""

import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace
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


# ------------------------------------------------------------------
# Task 2: MonitorServer + MonitorCallback tests
# ------------------------------------------------------------------

MonitorServer = _monitor.MonitorServer
MonitorCallback = _monitor.MonitorCallback


def test_monitor_server_starts_and_stops():
    """MonitorServer lifecycle: start sets is_running, stop clears it."""
    server = MonitorServer(port=15980)  # high port to avoid conflicts
    assert not server.is_running()

    server.start()
    try:
        assert server.is_running()
    finally:
        server.stop()

    assert not server.is_running()


def test_monitor_callback_stores_metrics():
    """on_log populates metrics_history with the expected fields."""
    model = MagicMock()
    model.named_parameters.return_value = []
    cb = MonitorCallback(model=model, config_data={"lr": 1e-4})
    # Simulate on_train_begin to set _start_time
    state = SimpleNamespace(global_step=0, max_steps=100, epoch=0.0)
    cb.on_train_begin(args=None, state=state, control=None)

    # Simulate a log event at step 10
    state.global_step = 10
    state.epoch = 0.5
    logs = {"loss": 0.42, "learning_rate": 5e-5, "grad_norm": 1.2}
    cb.on_log(args=None, state=state, control=None, logs=logs)

    assert len(cb.metrics_history) == 1
    m = cb.metrics_history[0]
    assert m["type"] == "metrics"
    assert m["step"] == 10
    assert m["max_steps"] == 100
    assert m["loss"] == 0.42
    assert m["lr"] == 5e-5
    assert m["grad_norm"] == 1.2
    assert "eta_seconds" in m
    assert "steps_per_sec" in m


def test_monitor_callback_computes_eta():
    """ETA computation returns a plausible estimate from elapsed time."""
    model = MagicMock()
    model.named_parameters.return_value = []
    cb = MonitorCallback(model=model, config_data={})

    # Fake start time: 10 seconds ago
    cb._start_time = time.time() - 10.0

    # Simulate state: 50 of 100 steps done in ~10 seconds
    state = SimpleNamespace(global_step=50, max_steps=100)
    eta = cb._compute_eta(state)

    # 50 remaining steps at 0.2 sec/step = ~10 seconds
    assert 5.0 < eta < 15.0, f"ETA {eta} seems off for 50/100 steps in 10s"


def test_monitor_callback_collects_lora_norms():
    """LoRA norms are extracted from model params containing 'lora_' in name."""
    # Build a mock model with named_parameters that include lora_ params
    # (pure MagicMock -- no torch needed)

    lora_a = MagicMock()
    lora_a.requires_grad = True
    lora_a.data = MagicMock()
    lora_a.data.norm.return_value = MagicMock(item=MagicMock(return_value=0.5))

    lora_b = MagicMock()
    lora_b.requires_grad = True
    lora_b.data = MagicMock()
    lora_b.data.norm.return_value = MagicMock(item=MagicMock(return_value=1.2))

    regular = MagicMock()
    regular.requires_grad = True

    model = MagicMock()
    model.named_parameters.return_value = [
        ("layer.0.lora_A", lora_a),
        ("layer.0.lora_B", lora_b),
        ("layer.0.weight", regular),  # not a lora param
    ]

    # Capture emitted messages via a mock server
    mock_server = MagicMock()
    cb = MonitorCallback(
        model=model,
        config_data={},
        server=mock_server,
        lora_norm_interval=50,
    )
    cb._start_time = time.time() - 10.0

    # Trigger on_log at step 50 (divisible by lora_norm_interval)
    state = SimpleNamespace(global_step=50, max_steps=200, epoch=1.0)
    cb.on_log(args=None, state=state, control=None, logs={"loss": 0.3})

    # Find the lora_norms broadcast call
    calls = mock_server.broadcast.call_args_list
    lora_msgs = [c.args[0] for c in calls if c.args[0].get("type") == "lora_norms"]

    assert len(lora_msgs) == 1, f"Expected 1 lora_norms message, got {len(lora_msgs)}"
    norms = lora_msgs[0]["norms"]
    assert "layer.0.lora_A" in norms
    assert "layer.0.lora_B" in norms
    assert "layer.0.weight" not in norms
    assert norms["layer.0.lora_A"] == 0.5
    assert norms["layer.0.lora_B"] == 1.2


# ------------------------------------------------------------------
# Task 3: CLI parser integration tests
# ------------------------------------------------------------------

def test_cli_parser_has_monitor_flags():
    # Load cli.py directly to avoid mud_puppy/__init__.py pulling in torch
    _CLI_PATH = Path(__file__).resolve().parent.parent / "mud_puppy" / "cli.py"

    # Stub out heavy imports that cli.py references at module level
    fake_config = MagicMock()
    fake_trainer = MagicMock()
    with patch.dict("sys.modules", {
        "mud_puppy": MagicMock(),
        "mud_puppy.config": fake_config,
        "mud_puppy.trainer": fake_trainer,
    }):
        _cli_spec = importlib.util.spec_from_file_location("mud_puppy.cli", _CLI_PATH)
        _cli_mod = importlib.util.module_from_spec(_cli_spec)
        _cli_spec.loader.exec_module(_cli_mod)

    build_parser = _cli_mod.build_parser
    parser = build_parser()
    args = parser.parse_args(["model.bin", "data.jsonl", "--monitor", "--monitor-port", "5981"])
    assert args.monitor is True
    assert args.monitor_port == 5981
    args2 = parser.parse_args(["model.bin", "data.jsonl", "--monitor-tui"])
    assert args2.monitor_tui is True


# ------------------------------------------------------------------
# Task 4: TUI Monitor (Rich) tests
# ------------------------------------------------------------------

_TUI_PATH = Path(__file__).resolve().parent.parent / "mud_puppy" / "tui.py"
_tui_spec = importlib.util.spec_from_file_location("mud_puppy.tui", _TUI_PATH)
_tui_mod = importlib.util.module_from_spec(_tui_spec)
sys.modules["mud_puppy.tui"] = _tui_mod
_tui_spec.loader.exec_module(_tui_mod)

TUIMonitor = _tui_mod.TUIMonitor
sparkline = _tui_mod.sparkline
_format_eta = _tui_mod._format_eta
_format_lr = _tui_mod._format_lr


def test_tui_monitor_handles_metrics():
    """TUIMonitor processes config/metrics/gpu messages and stores latest_metrics."""
    tui = TUIMonitor(live=False)

    # Initially empty
    assert tui.config_data is None
    assert tui.latest_metrics is None
    assert tui.latest_gpu is None
    assert tui.loss_history == []

    # Config message
    tui.update({"type": "config", "method": "lora", "model": "llama-3-8b"})
    assert tui.config_data is not None
    assert tui.config_data["method"] == "lora"
    assert tui.config_data["model"] == "llama-3-8b"

    # Metrics message
    tui.update({
        "type": "metrics",
        "step": 10,
        "max_steps": 100,
        "epoch": 0.5,
        "loss": 0.42,
        "lr": 5e-5,
        "grad_norm": 1.2,
        "eta_seconds": 90,
    })
    assert tui.latest_metrics is not None
    assert tui.latest_metrics["step"] == 10
    assert tui.latest_metrics["loss"] == 0.42
    assert len(tui.loss_history) == 1
    assert tui.loss_history[0] == 0.42

    # Second metrics message
    tui.update({
        "type": "metrics",
        "step": 20,
        "max_steps": 100,
        "epoch": 1.0,
        "loss": 0.35,
        "lr": 4e-5,
        "grad_norm": 0.9,
        "eta_seconds": 60,
    })
    assert tui.latest_metrics["step"] == 20
    assert tui.latest_metrics["loss"] == 0.35
    assert len(tui.loss_history) == 2
    assert tui.loss_history[1] == 0.35

    # GPU message
    tui.update({
        "type": "gpu",
        "vram_used": 12.5,
        "vram_total": 24.0,
        "gpu_util": 95.0,
        "temperature": 72.0,
        "power_draw": 280.0,
    })
    assert tui.latest_gpu is not None
    assert tui.latest_gpu["vram_used"] == 12.5
    assert tui.latest_gpu["gpu_util"] == 95.0


def test_tui_sparkline():
    """sparkline() renders correctly with known values."""
    # All same values -> all midpoint chars
    result = sparkline([5.0, 5.0, 5.0, 5.0, 5.0], width=5)
    assert len(result) == 5
    # All identical values map to index 4 (midpoint)
    assert all(c == _tui_mod._SPARK_CHARS[4] for c in result)

    # Ascending 0-8 -> each maps to one character in order
    vals = list(range(9))
    result = sparkline([float(v) for v in vals], width=9)
    assert len(result) == 9
    # First char should be space (index 0), last should be full block (index 8)
    assert result[0] == _tui_mod._SPARK_CHARS[0]
    assert result[-1] == _tui_mod._SPARK_CHARS[8]

    # Padding: fewer values than width
    result = sparkline([1.0, 2.0, 3.0], width=10)
    assert len(result) == 10
    # First 7 chars should be spaces (padding)
    assert result[:7] == "       "

    # Truncation: more values than width
    result = sparkline([float(i) for i in range(50)], width=10)
    assert len(result) == 10

    # Empty list -> all spaces
    result = sparkline([], width=5)
    assert result == "     "


def test_tui_sparkline_two_values():
    """Sparkline with min/max pair maps correctly."""
    result = sparkline([0.0, 1.0], width=5)
    assert len(result) == 5
    # 3 padding spaces + space char + full block
    assert result[3] == _tui_mod._SPARK_CHARS[0]  # min -> index 0
    assert result[4] == _tui_mod._SPARK_CHARS[8]  # max -> index 8


def test_format_eta():
    """_format_eta produces correct time strings."""
    assert _format_eta(0) == "--:--"
    assert _format_eta(-5) == "--:--"
    assert _format_eta(90) == "01:30"
    assert _format_eta(3661) == "1:01:01"
    assert _format_eta(60) == "01:00"
    assert _format_eta(5) == "00:05"


def test_format_lr():
    """_format_lr produces scientific notation."""
    assert _format_lr(5e-5) == "5.00e-05"
    assert _format_lr(1e-3) == "1.00e-03"
    assert _format_lr(0) == "0"


def test_tui_complete_message():
    """Complete message triggers _render_complete without crashing."""
    tui = TUIMonitor(live=False)
    # Should not raise
    tui.update({
        "type": "complete",
        "total_time": 300.0,
        "total_steps": 500,
        "best_loss": 0.25,
    })
    # After complete, latest_metrics should still be None (complete doesn't set it)
    assert tui.latest_metrics is None


def test_tui_start_stop_no_live():
    """start/stop with live=False are no-ops and don't raise."""
    tui = TUIMonitor(live=False)
    tui.start()
    tui.stop()
    # Should be safe to call multiple times
    tui.stop()
