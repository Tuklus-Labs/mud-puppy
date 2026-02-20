# Mud-Puppy Training Monitor -- Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Real-time Tempest Vector-Neon training dashboard for mud-puppy, served via WebSocket from inside the trainer process.

**Architecture:** MonitorCallback (HF TrainerCallback) pushes metrics via WebSocket to a self-contained HTML dashboard. aiohttp runs in a background thread. Rich TUI fallback for headless. Three new files: `monitor.py`, `tui.py`, `static/dashboard.html`.

**Tech Stack:** aiohttp (WebSocket server), vanilla JS + Canvas2D (charts), Rich (TUI), subprocess rocm-smi (GPU telemetry).

**Design doc:** `docs/plans/2026-02-20-training-monitor-design.md`

---

### Task 1: GPU Telemetry Module

The GPU telemetry function is a dependency for both the web monitor and TUI. Build it first as a standalone function in monitor.py.

**Files:**
- Create: `mud_puppy/monitor.py`
- Create: `tests/test_monitor.py`

**Step 1: Write the failing test**

```python
# tests/test_monitor.py
import pytest
from mud_puppy.monitor import get_gpu_telemetry


def test_get_gpu_telemetry_returns_dict():
    """GPU telemetry returns a dict with expected keys, even without a GPU."""
    result = get_gpu_telemetry()
    assert isinstance(result, dict)
    assert "vram_used" in result
    assert "vram_total" in result
    assert "gpu_util" in result
    assert "temperature" in result
    assert "power_draw" in result


def test_get_gpu_telemetry_values_are_numeric():
    result = get_gpu_telemetry()
    for key in ("vram_used", "vram_total", "gpu_util", "temperature", "power_draw"):
        assert isinstance(result[key], (int, float)), f"{key} should be numeric"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/aegis/Projects/mud-puppy && venv/bin/python -m pytest tests/test_monitor.py -v`
Expected: FAIL (monitor module doesn't exist yet)

**Step 3: Write minimal implementation**

```python
# mud_puppy/monitor.py
"""Real-time training monitor for mud-puppy.

Provides a WebSocket-based dashboard server and HuggingFace TrainerCallback
for pushing training metrics to connected browsers.
"""

import json
import subprocess
import time


def get_gpu_telemetry() -> dict:
    """Read GPU stats via torch.cuda or rocm-smi subprocess fallback.

    Returns dict with keys: vram_used, vram_total, gpu_util, temperature, power_draw.
    All values are floats. Returns zeros if no GPU is available.
    """
    result = {
        "vram_used": 0.0,
        "vram_total": 0.0,
        "gpu_util": 0.0,
        "temperature": 0.0,
        "power_draw": 0.0,
    }

    # Try torch.cuda first (works on both CUDA and ROCm)
    try:
        import torch
        if torch.cuda.is_available():
            result["vram_used"] = torch.cuda.memory_allocated() / 1e9
            result["vram_total"] = torch.cuda.get_device_properties(0).total_mem / 1e9
            # torch doesn't expose utilization directly, try rocm-smi
    except Exception:
        pass

    # rocm-smi for utilization, temp, power (ROCm systems)
    try:
        out = subprocess.run(
            ["rocm-smi", "--showuse", "--showtemp", "--showpower", "--json"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0:
            data = json.loads(out.stdout)
            # rocm-smi JSON format varies by version, handle both
            for key, card in data.items():
                if not isinstance(card, dict):
                    continue
                # GPU utilization
                for field in ("GPU use (%)", "GPU Usage (%)", "GPU_USE_PERCENT"):
                    if field in card:
                        val = str(card[field]).replace("%", "").strip()
                        try:
                            result["gpu_util"] = float(val)
                        except ValueError:
                            pass
                        break
                # Temperature
                for field in ("Temperature (Sensor edge) (C)", "TEMPERATURE_EDGE", "Temperature"):
                    if field in card:
                        val = str(card[field]).replace("C", "").strip()
                        try:
                            result["temperature"] = float(val)
                        except ValueError:
                            pass
                        break
                # Power
                for field in ("Average Graphics Package Power (W)", "AVERAGE_POWER", "Power"):
                    if field in card:
                        val = str(card[field]).replace("W", "").strip()
                        try:
                            result["power_draw"] = float(val)
                        except ValueError:
                            pass
                        break
                break  # Only first GPU
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    # nvidia-smi fallback for CUDA systems
    if result["gpu_util"] == 0.0:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if out.returncode == 0:
                parts = out.stdout.strip().split(",")
                if len(parts) >= 3:
                    result["gpu_util"] = float(parts[0].strip())
                    result["temperature"] = float(parts[1].strip())
                    result["power_draw"] = float(parts[2].strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

    return result
```

**Step 4: Run test to verify it passes**

Run: `cd /home/aegis/Projects/mud-puppy && venv/bin/python -m pytest tests/test_monitor.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
cd /home/aegis/Projects/mud-puppy
git add mud_puppy/monitor.py tests/test_monitor.py
git commit -m "feat: add GPU telemetry function for training monitor"
```

---

### Task 2: MonitorCallback + WebSocket Server

The core backend: aiohttp server running in a daemon thread, MonitorCallback pushing metrics over WebSocket.

**Files:**
- Modify: `mud_puppy/monitor.py` (append to existing)
- Modify: `tests/test_monitor.py` (append tests)

**Step 1: Write the failing tests**

Append to `tests/test_monitor.py`:

```python
import asyncio
import threading
from unittest.mock import MagicMock, patch


def test_monitor_server_starts_and_stops():
    """MonitorServer can start on a port and shut down cleanly."""
    from mud_puppy.monitor import MonitorServer

    server = MonitorServer(port=15980)  # high port for testing
    server.start()
    assert server.is_running()
    server.stop()
    assert not server.is_running()


def test_monitor_callback_stores_metrics():
    """MonitorCallback on_log stores metrics in its history."""
    from mud_puppy.monitor import MonitorCallback

    cb = MonitorCallback(model=None, config_data={})
    # Simulate HF Trainer on_log call
    mock_state = MagicMock()
    mock_state.global_step = 10
    mock_state.epoch = 0.5
    mock_state.max_steps = 100
    logs = {"loss": 1.23, "learning_rate": 2e-5}

    cb.on_log(None, mock_state, None, logs=logs)
    assert len(cb.metrics_history) == 1
    assert cb.metrics_history[0]["data"]["loss"] == 1.23


def test_monitor_callback_computes_eta():
    """MonitorCallback calculates ETA from step rate."""
    from mud_puppy.monitor import MonitorCallback

    cb = MonitorCallback(model=None, config_data={})
    mock_state = MagicMock()
    mock_state.global_step = 50
    mock_state.epoch = 0.5
    mock_state.max_steps = 100

    cb._train_start_time = time.time() - 100  # 100 seconds elapsed
    eta = cb._compute_eta(mock_state)
    # 50 steps in 100s = 2s/step, 50 remaining = ~100s
    assert 80 <= eta <= 120


def test_monitor_callback_collects_lora_norms():
    """MonitorCallback extracts LoRA weight norms from model."""
    import torch
    import torch.nn as nn
    from mud_puppy.monitor import MonitorCallback

    # Create a fake model with lora-named parameters
    model = nn.Module()
    model.register_parameter("layers.0.lora_A", nn.Parameter(torch.randn(8, 64)))
    model.register_parameter("layers.0.lora_B", nn.Parameter(torch.randn(64, 8)))
    model.register_parameter("layers.1.lora_A", nn.Parameter(torch.randn(8, 64)))
    model.register_parameter("layers.1.lora_B", nn.Parameter(torch.randn(64, 8)))

    cb = MonitorCallback(model=model, config_data={})
    norms = cb._collect_lora_norms()
    assert "layers.0" in norms
    assert "A" in norms["layers.0"]
    assert "B" in norms["layers.0"]
    assert isinstance(norms["layers.0"]["A"], float)
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/aegis/Projects/mud-puppy && venv/bin/python -m pytest tests/test_monitor.py -v`
Expected: FAIL (MonitorServer, MonitorCallback not defined)

**Step 3: Write implementation**

Append to `mud_puppy/monitor.py`:

```python
import asyncio
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

try:
    from aiohttp import web
except ImportError:
    web = None

from transformers import TrainerCallback


class MonitorServer:
    """aiohttp WebSocket server running in a background daemon thread.

    Serves the dashboard HTML and accepts WebSocket connections.
    Broadcasts JSON messages to all connected clients.
    """

    def __init__(self, port: int = 5980):
        self.port = port
        self._clients: list = []
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner: Optional[web.AppRunner] = None
        self._running = False
        self._browser_opened = False

    def is_running(self) -> bool:
        return self._running

    async def _ws_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.append(ws)

        # Open browser on first connection
        if not self._browser_opened:
            self._browser_opened = True

        try:
            async for msg in ws:
                pass  # We only send, never receive
        finally:
            self._clients.remove(ws)
        return ws

    async def _index_handler(self, request):
        html_path = Path(__file__).parent / "static" / "dashboard.html"
        if html_path.exists():
            return web.FileResponse(html_path)
        return web.Response(text="Dashboard HTML not found", status=404)

    async def _start_app(self):
        if web is None:
            raise RuntimeError("aiohttp is required for the web monitor. Install with: pip install aiohttp")

        app = web.Application()
        app.router.add_get("/", self._index_handler)
        app.router.add_get("/ws", self._ws_handler)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        self._running = True

    async def _stop_app(self):
        # Close all WebSocket connections
        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()

        if self._runner:
            await self._runner.cleanup()
        self._running = False

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_app())
        self._loop.run_forever()
        self._loop.run_until_complete(self._stop_app())
        self._loop.close()

    def start(self):
        """Start the server in a background daemon thread."""
        if self._running:
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Wait for server to be ready
        deadline = time.time() + 5
        while not self._running and time.time() < deadline:
            time.sleep(0.05)

    def stop(self):
        """Stop the server and close all connections."""
        if self._loop and self._running:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
            self._running = False

    def broadcast(self, message: dict):
        """Send a JSON message to all connected WebSocket clients."""
        if not self._clients or not self._loop:
            return
        data = json.dumps(message)

        async def _send():
            dead = []
            for ws in self._clients:
                try:
                    await ws.send_str(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.remove(ws)

        asyncio.run_coroutine_threadsafe(_send(), self._loop)


class MonitorCallback(TrainerCallback):
    """HuggingFace TrainerCallback that pushes metrics to MonitorServer and/or TUI.

    Args:
        model: The model being trained (for LoRA norm extraction).
        config_data: Dict of training config metadata to send on_train_begin.
        server: Optional MonitorServer instance for web dashboard.
        tui: Optional TUI monitor instance.
        lora_norm_interval: Collect LoRA norms every N steps (0 = disabled).
    """

    def __init__(
        self,
        model,
        config_data: dict,
        server: Optional[MonitorServer] = None,
        tui=None,
        lora_norm_interval: int = 50,
    ):
        self.model = model
        self.config_data = config_data
        self.server = server
        self.tui = tui
        self.lora_norm_interval = lora_norm_interval
        self.metrics_history: list = []
        self._train_start_time: Optional[float] = None
        self._last_step_time: Optional[float] = None
        self._last_step: int = 0
        self._tokens_seen: int = 0

    def _emit(self, message: dict):
        """Send message to all active outputs (server + tui)."""
        if self.server:
            self.server.broadcast(message)
        if self.tui:
            self.tui.update(message)

    def _compute_eta(self, state) -> float:
        """Compute estimated seconds remaining."""
        if self._train_start_time is None or state.global_step == 0:
            return 0.0
        elapsed = time.time() - self._train_start_time
        steps_done = state.global_step
        steps_remaining = state.max_steps - steps_done
        if steps_done == 0:
            return 0.0
        secs_per_step = elapsed / steps_done
        return secs_per_step * steps_remaining

    def _collect_lora_norms(self) -> dict:
        """Extract per-layer LoRA weight norms from the model.

        Returns dict like {"layers.0": {"A": 0.12, "B": 0.08}, ...}.
        Groups parameters by their layer prefix (everything before 'lora_A' or 'lora_B').
        """
        norms = {}
        if self.model is None:
            return norms

        for name, param in self.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                # Extract layer prefix: "layers.0.self_attn.q_proj.lora_A.weight" -> "layers.0"
                parts = name.split(".")
                lora_idx = next(i for i, p in enumerate(parts) if p.startswith("lora_"))
                # Use first two dotted segments as layer key (e.g. "layers.0")
                layer_key = ".".join(parts[:min(lora_idx, 2)])

                ab = "A" if "lora_A" in name else "B"

                if layer_key not in norms:
                    norms[layer_key] = {}
                norms[layer_key][ab] = param.data.norm().item()

        return norms

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start_time = time.time()
        self._emit({
            "type": "config",
            "timestamp": time.time(),
            "data": self.config_data,
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        now = time.time()
        eta = self._compute_eta(state)

        # Compute steps/sec
        steps_per_sec = 0.0
        if self._last_step_time and state.global_step > self._last_step:
            dt = now - self._last_step_time
            ds = state.global_step - self._last_step
            if dt > 0:
                steps_per_sec = ds / dt
        self._last_step_time = now
        self._last_step = state.global_step

        msg = {
            "type": "metrics",
            "timestamp": now,
            "data": {
                "step": state.global_step,
                "max_steps": state.max_steps,
                "epoch": round(state.epoch, 3) if state.epoch else 0,
                "loss": logs.get("loss", logs.get("train_loss")),
                "learning_rate": logs.get("learning_rate"),
                "grad_norm": logs.get("grad_norm"),
                "eta_seconds": round(eta, 1),
                "steps_per_sec": round(steps_per_sec, 2),
            },
        }
        self.metrics_history.append(msg)
        self._emit(msg)

        # LoRA norms at interval
        if (
            self.lora_norm_interval > 0
            and state.global_step > 0
            and state.global_step % self.lora_norm_interval == 0
        ):
            norms = self._collect_lora_norms()
            if norms:
                self._emit({
                    "type": "lora_norms",
                    "timestamp": now,
                    "data": {"step": state.global_step, "layers": norms},
                })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        self._emit({
            "type": "eval",
            "timestamp": time.time(),
            "data": {
                "step": state.global_step,
                "eval_loss": metrics.get("eval_loss"),
                **{k: v for k, v in metrics.items() if k != "eval_loss"},
            },
        })

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self._train_start_time if self._train_start_time else 0
        best_loss = None
        if self.metrics_history:
            losses = [m["data"].get("loss") for m in self.metrics_history if m["data"].get("loss") is not None]
            if losses:
                best_loss = min(losses)

        self._emit({
            "type": "complete",
            "timestamp": time.time(),
            "data": {
                "total_time": round(elapsed, 1),
                "total_steps": state.global_step,
                "best_loss": best_loss,
            },
        })


def start_gpu_telemetry(server: MonitorServer, interval: float = 1.0):
    """Start a background thread that broadcasts GPU telemetry at interval."""

    def _telemetry_loop():
        while server.is_running():
            telemetry = get_gpu_telemetry()
            server.broadcast({
                "type": "gpu",
                "timestamp": time.time(),
                "data": telemetry,
            })
            time.sleep(interval)

    t = threading.Thread(target=_telemetry_loop, daemon=True)
    t.start()
    return t
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/aegis/Projects/mud-puppy && venv/bin/python -m pytest tests/test_monitor.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
cd /home/aegis/Projects/mud-puppy
git add mud_puppy/monitor.py tests/test_monitor.py
git commit -m "feat: add MonitorServer and MonitorCallback for real-time training metrics"
```

---

### Task 3: CLI Integration

Wire `--monitor`, `--monitor-tui`, and `--monitor-port` into the CLI and trainer pipeline.

**Files:**
- Modify: `mud_puppy/cli.py` (add 3 new arguments)
- Modify: `mud_puppy/config.py` (add 3 new fields)
- Modify: `mud_puppy/trainer.py` (inject MonitorCallback into run_training)
- Modify: `tests/test_monitor.py` (add CLI arg test)

**Step 1: Write the failing test**

Append to `tests/test_monitor.py`:

```python
def test_cli_parser_has_monitor_flags():
    """CLI parser accepts --monitor, --monitor-tui, --monitor-port."""
    from mud_puppy.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["model.bin", "data.jsonl", "--monitor", "--monitor-port", "5981"])
    assert args.monitor is True
    assert args.monitor_port == 5981

    args2 = parser.parse_args(["model.bin", "data.jsonl", "--monitor-tui"])
    assert args2.monitor_tui is True
```

**Step 2: Run test to verify it fails**

Run: `cd /home/aegis/Projects/mud-puppy && venv/bin/python -m pytest tests/test_monitor.py::test_cli_parser_has_monitor_flags -v`
Expected: FAIL (unrecognized arguments)

**Step 3: Write implementation**

Add to `mud_puppy/cli.py` parser (after the `--merge-precision` argument block, before `--distributed`):

```python
    parser.add_argument(
        "--monitor",
        dest="monitor",
        action="store_true",
        help="enable real-time web training dashboard (port 5980)",
    )
    parser.add_argument(
        "--monitor-tui",
        dest="monitor_tui",
        action="store_true",
        help="enable terminal (Rich) training monitor",
    )
    parser.add_argument(
        "--monitor-port",
        dest="monitor_port",
        type=int,
        default=5980,
        help="port for web training monitor (default: 5980)",
    )
```

Add to `mud_puppy/config.py` TrainingConfig (after `merge_precision`):

```python
    # Monitor
    monitor: bool = False
    monitor_tui: bool = False
    monitor_port: int = 5980
```

Add to `mud_puppy/cli.py` `main()` config_kwargs dict:

```python
        monitor=args.monitor,
        monitor_tui=args.monitor_tui,
        monitor_port=args.monitor_port,
```

Modify `mud_puppy/trainer.py` `run_training()` -- after the callbacks list is built (after `if config.zero_offload:` block), add:

```python
    # Monitor callback (web dashboard and/or TUI)
    monitor_server = None
    monitor_tui_inst = None
    if config.monitor or config.monitor_tui:
        from .monitor import MonitorCallback, MonitorServer, start_gpu_telemetry

        config_data = {
            "model": config.model_name_or_path,
            "method": config.finetuning_method,
            "precision": config.precision,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "dataset_size": len(dataset),
            "lora_r": config.lora_r if config.finetuning_method in ("lora", "qlora") else None,
            "lora_alpha": config.lora_alpha if config.finetuning_method in ("lora", "qlora") else None,
            "quant_backend": config.quant_backend if config.finetuning_method == "qlora" else None,
        }

        if config.monitor:
            monitor_server = MonitorServer(port=config.monitor_port)
            monitor_server.start()
            start_gpu_telemetry(monitor_server)
            print(f"[mud-puppy] Training monitor: http://localhost:{config.monitor_port}")
            import webbrowser
            webbrowser.open(f"http://localhost:{config.monitor_port}")

        if config.monitor_tui:
            from .tui import TUIMonitor
            monitor_tui_inst = TUIMonitor()
            monitor_tui_inst.start()

        callbacks.append(MonitorCallback(
            model=model,
            config_data=config_data,
            server=monitor_server,
            tui=monitor_tui_inst,
            lora_norm_interval=50 if config.finetuning_method in ("lora", "qlora") else 0,
        ))
```

And after `trainer.train()` completes (before "Saving model"), add cleanup:

```python
    # Stop monitor
    if monitor_server:
        monitor_server.stop()
    if monitor_tui_inst:
        monitor_tui_inst.stop()
```

**Step 4: Run tests**

Run: `cd /home/aegis/Projects/mud-puppy && venv/bin/python -m pytest tests/test_monitor.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
cd /home/aegis/Projects/mud-puppy
git add mud_puppy/cli.py mud_puppy/config.py mud_puppy/trainer.py tests/test_monitor.py
git commit -m "feat: wire --monitor and --monitor-tui flags into CLI and trainer"
```

---

### Task 4: TUI Monitor (Rich)

Lightweight terminal monitor using Rich Live display.

**Files:**
- Create: `mud_puppy/tui.py`
- Modify: `tests/test_monitor.py` (add TUI tests)

**Step 1: Write the failing test**

Append to `tests/test_monitor.py`:

```python
def test_tui_monitor_handles_metrics():
    """TUI monitor processes metrics messages without crashing."""
    from mud_puppy.tui import TUIMonitor

    tui = TUIMonitor(live=False)  # Don't start Live display in tests
    tui.update({
        "type": "config",
        "data": {"model": "test-model", "method": "lora", "dataset_size": 100},
    })
    tui.update({
        "type": "metrics",
        "data": {"step": 10, "max_steps": 100, "loss": 1.5, "learning_rate": 2e-5,
                 "epoch": 0.1, "eta_seconds": 90, "grad_norm": 0.5, "steps_per_sec": 1.1},
    })
    tui.update({
        "type": "gpu",
        "data": {"vram_used": 14.2, "vram_total": 24.0, "gpu_util": 97, "temperature": 72, "power_draw": 284},
    })
    # Should have stored metrics
    assert tui.latest_metrics is not None
    assert tui.latest_metrics["loss"] == 1.5


def test_tui_sparkline():
    """TUI sparkline renders loss history as block characters."""
    from mud_puppy.tui import sparkline

    values = [2.0, 1.8, 1.5, 1.2, 1.0, 0.8]
    result = sparkline(values, width=6)
    assert len(result) == 6
    # All characters should be block elements
    for ch in result:
        assert ch in " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/aegis/Projects/mud-puppy && venv/bin/python -m pytest tests/test_monitor.py::test_tui_monitor_handles_metrics tests/test_monitor.py::test_tui_sparkline -v`
Expected: FAIL (tui module not found)

**Step 3: Write implementation**

```python
# mud_puppy/tui.py
"""Terminal (Rich) training monitor for mud-puppy.

Provides a lightweight TUI alternative to the web dashboard, using
Rich's Live display for in-place updates. Designed for SSH/headless sessions.
"""

from typing import Optional
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout


SPARK_CHARS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def sparkline(values: list, width: int = 40) -> str:
    """Render a list of floats as a sparkline using Unicode block characters."""
    if not values:
        return " " * width

    # Take last `width` values
    vals = values[-width:]

    mn = min(vals)
    mx = max(vals)
    rng = mx - mn if mx != mn else 1.0

    chars = []
    for v in vals:
        normalized = (v - mn) / rng
        idx = int(normalized * (len(SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])

    # Pad if shorter than width
    while len(chars) < width:
        chars.insert(0, " ")

    return "".join(chars)


def _format_eta(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds <= 0:
        return "--:--"
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def _format_lr(lr) -> str:
    if lr is None:
        return "N/A"
    return f"{lr:.1E}"


class TUIMonitor:
    """Rich-based terminal training monitor.

    Args:
        live: Whether to start Rich Live display (False for testing).
    """

    def __init__(self, live: bool = True):
        self.console = Console()
        self._live_mode = live
        self._live: Optional[Live] = None
        self.config_data: dict = {}
        self.latest_metrics: Optional[dict] = None
        self.latest_gpu: Optional[dict] = None
        self.loss_history: list = []

    def start(self):
        if self._live_mode:
            self._live = Live(console=self.console, refresh_per_second=2)
            self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()

    def update(self, message: dict):
        """Process an incoming monitor message."""
        msg_type = message.get("type")
        data = message.get("data", {})

        if msg_type == "config":
            self.config_data = data
        elif msg_type == "metrics":
            self.latest_metrics = data
            loss = data.get("loss")
            if loss is not None:
                self.loss_history.append(loss)
        elif msg_type == "gpu":
            self.latest_gpu = data
        elif msg_type == "eval":
            pass  # Could extend to show eval in TUI
        elif msg_type == "complete":
            self._render_complete(data)
            return

        self._render()

    def _render(self):
        """Build and display the TUI layout."""
        if not self._live:
            return

        m = self.latest_metrics or {}
        g = self.latest_gpu or {}

        # Header
        model = self.config_data.get("model", "?")
        method = self.config_data.get("method", "?")
        step = m.get("step", 0)
        max_steps = m.get("max_steps", 0)
        header = f"MUD-PUPPY MONITOR [{method}] [{model}] [step {step}/{max_steps}]"

        # Metrics table
        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
        table.add_column("STEP", style="bold white")
        table.add_column("EPOCH", style="bold white")
        table.add_column("LOSS", style="bold green")
        table.add_column("LR", style="bold cyan")
        table.add_column("GNORM", style="bold yellow")
        table.add_column("ETA", style="bold white")

        epoch_str = f"{m.get('epoch', 0):.1f}/{self.config_data.get('num_epochs', '?')}"
        loss_str = f"{m.get('loss', 0):.4f}" if m.get("loss") is not None else "N/A"
        gnorm_str = f"{m.get('grad_norm', 0):.3f}" if m.get("grad_norm") is not None else "N/A"

        table.add_row(
            str(step),
            epoch_str,
            loss_str,
            _format_lr(m.get("learning_rate")),
            gnorm_str,
            _format_eta(m.get("eta_seconds", 0)),
        )

        # Sparkline
        spark = sparkline(self.loss_history, width=50)
        loss_line = Text(f"Loss: {spark}", style="green")

        # GPU line
        gpu_parts = []
        if g:
            gpu_parts.append(f"GPU: {g.get('vram_used', 0):.1f}/{g.get('vram_total', 0):.1f} GB")
            gpu_parts.append(f"{g.get('gpu_util', 0):.0f}%")
            gpu_parts.append(f"{g.get('temperature', 0):.0f}C")
            gpu_parts.append(f"{g.get('power_draw', 0):.0f}W")
        gpu_line = Text("  ".join(gpu_parts), style="cyan") if gpu_parts else Text("")

        # Combine
        panel = Panel(
            f"{table}\n{loss_line}\n{gpu_line}",
            title=header,
            border_style="cyan",
        )
        self._live.update(panel)

    def _render_complete(self, data: dict):
        """Show training complete summary."""
        if self._live:
            total = data.get("total_time", 0)
            best = data.get("best_loss")
            msg = f"TRAINING COMPLETE  |  Time: {_format_eta(total)}  |  Best loss: {best}"
            self._live.update(Panel(msg, title="MUD-PUPPY", border_style="green"))
```

**Step 4: Run tests**

Run: `cd /home/aegis/Projects/mud-puppy && venv/bin/python -m pytest tests/test_monitor.py -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
cd /home/aegis/Projects/mud-puppy
git add mud_puppy/tui.py tests/test_monitor.py
git commit -m "feat: add Rich TUI training monitor"
```

---

### Task 5: Dashboard HTML (Tempest Vector-Neon)

The main event. Single self-contained HTML file with inline CSS and JS. Canvas2D charts with glow effects. WebSocket connection with auto-reconnect.

**Files:**
- Create: `mud_puppy/static/dashboard.html`

**Step 1: Create the static directory**

```bash
mkdir -p /home/aegis/Projects/mud-puppy/mud_puppy/static
```

**Step 2: Write the dashboard**

This is a single large HTML file. Key sections:

1. **CSS**: Tempest color palette, monospace fonts (Share Tech Mono from Google Fonts CDN), CRT scanline overlay, CSS Grid layout, glow effects
2. **HTML**: Panel grid matching the design doc layout
3. **JS - WebSocket**: Connect to `ws://host:port/ws`, auto-reconnect with exponential backoff
4. **JS - Charts**: Canvas2D line charts with rolling window (500pt), glow via shadowBlur/shadowColor, grid lines, axis labels
5. **JS - Gauges**: Arc gauge renderer for VRAM/util with cyan-to-red gradient
6. **JS - LoRA norms**: Multi-line chart or mini heatmap grid
7. **JS - Stats bar**: Numeric readouts with large monospace values

The complete HTML file should be written to `mud_puppy/static/dashboard.html`.

CSS color variables:

```css
:root {
    --bg: #0a0e1a;
    --panel-bg: #0d1420;
    --grid: #1a2a3a;
    --border: #1a3a4a;
    --green: #39ff14;
    --cyan: #00ffff;
    --amber: #ffd700;
    --magenta: #ff00ff;
    --red: #ff3333;
    --text: #e0e0e0;
    --dim: #607080;
    --font: 'Share Tech Mono', 'JetBrains Mono', 'Fira Code', monospace;
}
```

Key JS architecture:
- `class Chart` - Reusable Canvas chart with rolling buffer, grid, glow, auto-scaling Y axis
- `class ArcGauge` - Canvas arc gauge with threshold coloring
- `class Dashboard` - Orchestrator that creates charts/gauges and routes WS messages
- WebSocket reconnect: `setTimeout(connect, Math.min(1000 * 2**attempts, 30000))`

Panels rendered via CSS Grid:
```css
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 8px;
    padding: 8px;
}
```

The dashboard must handle all message types from the WebSocket protocol:
- `config` -> populate header bar + model info footer
- `metrics` -> update stat readouts + push to loss/lr/gnorm charts
- `gpu` -> update gauge values
- `eval` -> push to eval loss chart
- `lora_norms` -> update LoRA norms chart
- `complete` -> flash completion banner

**Step 3: Verify by manual test**

After writing the file:
1. Start a simple test: `cd /home/aegis/Projects/mud-puppy && venv/bin/python -c "from mud_puppy.monitor import MonitorServer; s = MonitorServer(); s.start(); import time; time.sleep(999)" &`
2. Open `http://localhost:5980` in browser
3. Verify the dashboard loads and shows "DISCONNECTED" (no training data yet)
4. Kill the test server

**Step 4: Commit**

```bash
cd /home/aegis/Projects/mud-puppy
git add mud_puppy/static/dashboard.html
git commit -m "feat: add Tempest Vector-Neon training dashboard"
```

---

### Task 6: Integration Test with Simulated Training

End-to-end test that simulates a training run and verifies the full pipeline works.

**Files:**
- Modify: `tests/test_monitor.py` (add integration test)

**Step 1: Write the integration test**

```python
import asyncio
import json
import time
import threading


def test_end_to_end_monitor_pipeline():
    """Full pipeline: MonitorServer + MonitorCallback + simulated training steps."""
    from mud_puppy.monitor import MonitorServer, MonitorCallback
    from unittest.mock import MagicMock

    server = MonitorServer(port=15981)
    server.start()
    assert server.is_running()

    # Collect broadcast messages
    received = []
    original_broadcast = server.broadcast

    def capture_broadcast(msg):
        received.append(msg)
        original_broadcast(msg)

    server.broadcast = capture_broadcast

    try:
        cb = MonitorCallback(
            model=None,
            config_data={"model": "test", "method": "full"},
            server=server,
            lora_norm_interval=0,
        )

        # Simulate on_train_begin
        mock_state = MagicMock()
        mock_state.global_step = 0
        mock_state.max_steps = 100
        mock_state.epoch = 0
        cb.on_train_begin(None, mock_state, None)

        # Simulate 3 logging steps
        for step in [10, 20, 30]:
            mock_state.global_step = step
            mock_state.epoch = step / 100
            cb.on_log(None, mock_state, None, logs={
                "loss": 2.0 - step * 0.03,
                "learning_rate": 2e-5 * (1 - step / 100),
                "grad_norm": 0.5,
            })

        # Simulate on_train_end
        cb.on_train_end(None, mock_state, None)

        # Verify messages
        types = [m["type"] for m in received]
        assert "config" in types
        assert types.count("metrics") == 3
        assert "complete" in types

        # Verify loss tracking
        assert len(cb.metrics_history) == 3
        assert cb.metrics_history[0]["data"]["loss"] == 2.0 - 10 * 0.03
    finally:
        server.stop()
```

**Step 2: Run test**

Run: `cd /home/aegis/Projects/mud-puppy && venv/bin/python -m pytest tests/test_monitor.py::test_end_to_end_monitor_pipeline -v`
Expected: PASS

**Step 3: Commit**

```bash
cd /home/aegis/Projects/mud-puppy
git add tests/test_monitor.py
git commit -m "test: add end-to-end integration test for training monitor"
```

---

### Task 7: Final Polish and Docs

Update `__init__.py` exports, add monitor to the `__all__` list, update pyproject.toml if needed for aiohttp dependency.

**Files:**
- Modify: `mud_puppy/__init__.py` (add monitor exports)
- Modify: `pyproject.toml` (add aiohttp + rich as optional deps)

**Step 1: Add exports to `__init__.py`**

Add to `__all__`:
```python
    "MonitorServer",
    "MonitorCallback",
    "TUIMonitor",
```

Add lazy imports:
```python
def MonitorServer(*args, **kwargs):
    from .monitor import MonitorServer as _MonitorServer
    return _MonitorServer(*args, **kwargs)

def MonitorCallback(*args, **kwargs):
    from .monitor import MonitorCallback as _MonitorCallback
    return _MonitorCallback(*args, **kwargs)

def TUIMonitor(*args, **kwargs):
    from .tui import TUIMonitor as _TUIMonitor
    return _TUIMonitor(*args, **kwargs)
```

**Step 2: Update pyproject.toml optional deps**

Add `monitor` extras group:
```toml
[project.optional-dependencies]
monitor = ["aiohttp>=3.9", "rich>=13.0"]
```

**Step 3: Run full test suite**

Run: `cd /home/aegis/Projects/mud-puppy && venv/bin/python -m pytest tests/test_monitor.py -v`
Expected: ALL PASS

**Step 4: Final commit**

```bash
cd /home/aegis/Projects/mud-puppy
git add mud_puppy/__init__.py pyproject.toml
git commit -m "feat: export monitor components and add optional deps"
```
