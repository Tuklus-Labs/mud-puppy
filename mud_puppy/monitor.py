"""GPU telemetry and training monitor for mud-puppy.

Provides:
  - GPU telemetry (VRAM, utilization, temperature, power) via ROCm/CUDA
  - ``MonitorServer`` -- aiohttp WebSocket server for live dashboard streaming
  - ``MonitorCallback`` -- HuggingFace TrainerCallback that emits training metrics
  - ``start_gpu_telemetry()`` -- background thread that broadcasts GPU stats

Priority order for GPU stats:
  1. ``torch.cuda`` for VRAM (works on both CUDA and ROCm via HIP)
  2. ``rocm-smi`` subprocess for util / temp / power (AMD)
  3. ``nvidia-smi`` subprocess for everything (NVIDIA)
  4. Zeros for any metric that cannot be read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# ---------------------------------------------------------------------------
# MonitorServer -- aiohttp WebSocket server in a daemon thread
# ---------------------------------------------------------------------------

class MonitorServer:
    """WebSocket server that streams training metrics to browser dashboards.

    Runs an aiohttp web server in its own daemon thread with a dedicated
    asyncio event loop.  Clients connect via WebSocket at ``/ws`` and
    receive JSON messages broadcast by the training callback.  A static
    dashboard is served at ``GET /``.

    Usage::

        server = MonitorServer(port=5980)
        server.start()
        server.broadcast({"type": "metrics", "loss": 0.42})
        server.stop()
    """

    def __init__(self, port: int = 5980) -> None:
        self.port = port
        self._clients: List[Any] = []  # list of aiohttp.web.WebSocketResponse
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._runner: Any = None  # aiohttp.web.AppRunner
        self._running = threading.Event()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the server in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self._run, daemon=True, name="monitor-ws")
        self._thread.start()
        # Wait for the server to be ready (up to 5s)
        self._running.wait(timeout=5.0)

    def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._loop is None:
            return
        loop = self._loop
        asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._running.clear()
        self._loop = None
        self._thread = None

    def is_running(self) -> bool:
        """Return True if the server is accepting connections."""
        return self._running.is_set()

    def broadcast(self, message: dict) -> None:
        """Send a JSON message to every connected WebSocket client."""
        if not self._loop or not self._running.is_set():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(message), self._loop)

    # -- internals ---------------------------------------------------------

    def _run(self) -> None:
        """Entry point for the daemon thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        """Set up and run the aiohttp application."""
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/ws", self._handle_ws)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        log.info("MonitorServer listening on port %d", self.port)
        self._running.set()

        # Keep the loop alive until shutdown is requested
        try:
            while self._running.is_set():
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            await self._runner.cleanup()

    async def _shutdown(self) -> None:
        """Signal the serve loop to exit."""
        self._running.clear()
        # Close all client connections
        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:
                pass
        self._clients.clear()

    async def _handle_index(self, request: Any) -> Any:
        """Serve the static dashboard HTML."""
        from aiohttp import web

        dashboard = Path(__file__).parent / "static" / "dashboard.html"
        if dashboard.exists():
            return web.FileResponse(dashboard)
        return web.Response(text="dashboard not built yet", content_type="text/plain")

    async def _handle_ws(self, request: Any) -> Any:
        """Handle an incoming WebSocket connection."""
        from aiohttp import web

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.append(ws)
        log.debug("WebSocket client connected (%d total)", len(self._clients))

        try:
            async for _msg in ws:
                pass  # We only send, never read
        finally:
            self._clients.remove(ws)
            log.debug("WebSocket client disconnected (%d remaining)", len(self._clients))

        return ws

    async def _broadcast_async(self, message: dict) -> None:
        """Send a JSON payload to all live clients, pruning dead ones."""
        payload = json.dumps(message)
        dead = []
        for ws in self._clients:
            try:
                await ws.send_str(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                self._clients.remove(ws)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# MonitorCallback -- HuggingFace TrainerCallback
# ---------------------------------------------------------------------------

class MonitorCallback:
    """HuggingFace ``TrainerCallback`` that streams training metrics.

    Emits JSON messages to a ``MonitorServer`` (WebSocket) and/or a TUI
    object.  Maintains ``metrics_history`` for post-hoc analysis.

    This class intentionally does NOT inherit from
    ``transformers.TrainerCallback`` at import time so that the module can
    be loaded without transformers installed.  The duck-typing protocol is
    identical -- HuggingFace's Trainer dispatches by method name, not by
    isinstance checks.
    """

    def __init__(
        self,
        model: Any,
        config_data: dict,
        server: Optional[MonitorServer] = None,
        tui: Optional[Any] = None,
        lora_norm_interval: int = 50,
    ) -> None:
        self.model = model
        self.config_data = config_data
        self.server = server
        self.tui = tui
        self.lora_norm_interval = lora_norm_interval

        self.metrics_history: List[dict] = []
        self._start_time: Optional[float] = None
        self._best_loss: float = float("inf")

    # -- TrainerCallback hooks ---------------------------------------------

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Record start time and emit config."""
        self._start_time = time.time()
        self._emit({"type": "config", **self.config_data})

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[dict] = None, **kwargs: Any) -> None:
        """Emit training metrics on every logging step."""
        logs = logs or {}
        step = state.global_step
        max_steps = state.max_steps
        loss = logs.get("loss", 0.0)
        lr = logs.get("learning_rate", 0.0)
        grad_norm = logs.get("grad_norm", 0.0)

        if loss and loss < self._best_loss:
            self._best_loss = loss

        eta = self._compute_eta(state)
        elapsed = time.time() - (self._start_time or time.time())
        steps_per_sec = step / elapsed if elapsed > 0 and step > 0 else 0.0

        msg = {
            "type": "metrics",
            "step": step,
            "max_steps": max_steps,
            "epoch": getattr(state, "epoch", 0.0),
            "loss": loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "eta_seconds": eta,
            "steps_per_sec": steps_per_sec,
        }
        self.metrics_history.append(msg)
        self._emit(msg)

        # LoRA norms at interval
        if self.lora_norm_interval > 0 and step % self.lora_norm_interval == 0 and step > 0:
            self._emit_lora_norms(step)

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Optional[dict] = None, **kwargs: Any) -> None:
        """Emit evaluation results."""
        metrics = metrics or {}
        self._emit({"type": "eval", "step": state.global_step, **metrics})

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Emit completion summary."""
        elapsed = time.time() - (self._start_time or time.time())
        self._emit({
            "type": "complete",
            "total_time": elapsed,
            "total_steps": state.global_step,
            "best_loss": self._best_loss if self._best_loss != float("inf") else None,
        })

    # -- helpers -----------------------------------------------------------

    def _compute_eta(self, state: Any) -> float:
        """Estimate seconds remaining based on elapsed time and progress."""
        if self._start_time is None or state.global_step <= 0:
            return 0.0
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        steps_done = state.global_step
        steps_total = state.max_steps
        if steps_total <= 0:
            return 0.0
        steps_remaining = steps_total - steps_done
        sec_per_step = elapsed / steps_done
        return sec_per_step * steps_remaining

    def _emit(self, message: dict) -> None:
        """Send a message to both the WebSocket server and the TUI."""
        if self.server is not None:
            try:
                self.server.broadcast(message)
            except Exception as exc:
                log.debug("broadcast failed: %s", exc)
        if self.tui is not None:
            try:
                self.tui.update(message)
            except Exception as exc:
                log.debug("tui update failed: %s", exc)

    def _emit_lora_norms(self, step: int) -> None:
        """Collect per-layer L2 norms for LoRA parameters and emit them."""
        norms: Dict[str, float] = {}
        try:
            for name, param in self.model.named_parameters():
                if "lora_" in name and param.requires_grad:
                    norms[name] = float(param.data.norm(2).item())
        except Exception as exc:
            log.debug("LoRA norm collection failed: %s", exc)
            return

        if norms:
            self._emit({"type": "lora_norms", "step": step, "norms": norms})


# ---------------------------------------------------------------------------
# GPU telemetry broadcaster
# ---------------------------------------------------------------------------

def start_gpu_telemetry(server: MonitorServer, interval: float = 1.0) -> threading.Thread:
    """Start a daemon thread that broadcasts GPU telemetry at *interval* seconds.

    Returns the thread object (already started).
    """
    def _loop() -> None:
        while server.is_running():
            try:
                data = get_gpu_telemetry()
                server.broadcast({"type": "gpu", **data})
            except Exception as exc:
                log.debug("gpu telemetry broadcast failed: %s", exc)
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True, name="monitor-gpu-telem")
    t.start()
    return t
