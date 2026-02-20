"""Rich-based terminal training monitor for mud-puppy.

Provides a TUI fallback for SSH/headless sessions.  Receives the same
message dict protocol as the WebSocket dashboard (config, metrics, gpu,
complete) and renders them live in the terminal via Rich.

Usage::

    from mud_puppy.tui import TUIMonitor

    tui = TUIMonitor(live=True)
    tui.start()
    tui.update({"type": "config", "method": "lora", "model": "llama-3-8b"})
    tui.update({"type": "metrics", "step": 10, "max_steps": 100,
                "loss": 0.42, "lr": 5e-5, "grad_norm": 1.2,
                "epoch": 0.5, "eta_seconds": 90})
    tui.stop()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------

_SPARK_CHARS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def sparkline(values: List[float], width: int = 40) -> str:
    """Render a list of floats as a Unicode sparkline string.

    Uses block-element characters (space + 8 blocks) to visualize the
    values.  The output is always *width* characters wide:

    - If fewer values than *width*, the result is left-padded with spaces.
    - If more values than *width*, only the last *width* values are used.

    Values are normalized to the [min, max] range of the slice, then
    mapped to one of the 9 block characters.
    """
    if not values:
        return " " * width

    # Take last `width` values
    tail = values[-width:]

    lo = min(tail)
    hi = max(tail)
    span = hi - lo

    chars: List[str] = []
    for v in tail:
        if span == 0:
            idx = 4  # midpoint when all values are identical
        else:
            normalized = (v - lo) / span
            idx = int(normalized * (len(_SPARK_CHARS) - 1))
            # Clamp just in case of floating-point edge
            idx = max(0, min(idx, len(_SPARK_CHARS) - 1))
        chars.append(_SPARK_CHARS[idx])

    # Left-pad if fewer values than width
    pad = width - len(chars)
    if pad > 0:
        chars = [" "] * pad + chars

    return "".join(chars)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_eta(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    if seconds <= 0:
        return "--:--"
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_lr(lr: float) -> str:
    """Format a learning rate in scientific notation."""
    if lr == 0:
        return "0"
    return f"{lr:.2e}"


# ---------------------------------------------------------------------------
# TUIMonitor
# ---------------------------------------------------------------------------

class TUIMonitor:
    """Rich-based terminal training monitor.

    Parameters
    ----------
    live : bool
        When True (default), creates and manages a ``rich.live.Live``
        display that auto-refreshes.  Set to False for testing or when
        Rich Live is not desired (e.g. piped output).
    """

    def __init__(self, live: bool = True) -> None:
        self._use_live = live
        self._live: Any = None  # rich.live.Live instance

        # State populated by update()
        self.config_data: Optional[Dict[str, Any]] = None
        self.latest_metrics: Optional[Dict[str, Any]] = None
        self.latest_gpu: Optional[Dict[str, Any]] = None
        self.loss_history: List[float] = []

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the Rich Live display (no-op if live=False)."""
        if not self._use_live:
            return
        from rich.live import Live
        from rich.console import Console

        console = Console()
        self._live = Live(console=console, refresh_per_second=4)
        self._live.start()

    def stop(self) -> None:
        """Stop the Rich Live display (no-op if live=False)."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    # -- message dispatch --------------------------------------------------

    def update(self, message: dict) -> None:
        """Process an incoming monitor message by type.

        Supported message types:

        - ``"config"``   -- training configuration; stored in config_data
        - ``"metrics"``  -- step metrics; stored in latest_metrics, loss appended
        - ``"gpu"``      -- GPU telemetry; stored in latest_gpu
        - ``"complete"`` -- training finished; renders completion banner
        """
        msg_type = message.get("type")

        if msg_type == "config":
            self.config_data = message

        elif msg_type == "metrics":
            self.latest_metrics = message
            loss = message.get("loss")
            if loss is not None:
                self.loss_history.append(float(loss))

        elif msg_type == "gpu":
            self.latest_gpu = message

        elif msg_type == "complete":
            self._render_complete(message)
            return

        self._render()

    # -- rendering ---------------------------------------------------------

    def _render(self) -> None:
        """Build a Rich Panel from current state and push it to Live."""
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        parts: List[Any] = []

        # --- Header ---
        method = ""
        model = ""
        if self.config_data:
            method = self.config_data.get("method", self.config_data.get("finetuning_method", ""))
            model = self.config_data.get("model", self.config_data.get("model_name_or_path", ""))

        step_str = ""
        if self.latest_metrics:
            step = self.latest_metrics.get("step", "?")
            max_steps = self.latest_metrics.get("max_steps", "?")
            step_str = f"[step {step}/{max_steps}]"

        header = Text(f"MUD-PUPPY MONITOR {method} {model} {step_str}".strip(), style="bold cyan")
        parts.append(header)
        parts.append(Text(""))

        # --- Metrics table ---
        if self.latest_metrics:
            m = self.latest_metrics
            tbl = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
            tbl.add_column("STEP", justify="right")
            tbl.add_column("EPOCH", justify="right")
            tbl.add_column("LOSS", justify="right")
            tbl.add_column("LR", justify="right")
            tbl.add_column("GNORM", justify="right")
            tbl.add_column("ETA", justify="right")

            tbl.add_row(
                str(m.get("step", "")),
                f"{m.get('epoch', 0):.2f}",
                f"{m.get('loss', 0):.4f}",
                _format_lr(m.get("lr", 0)),
                f"{m.get('grad_norm', 0):.2f}",
                _format_eta(m.get("eta_seconds", 0)),
            )
            parts.append(tbl)
            parts.append(Text(""))

        # --- Sparkline ---
        if self.loss_history:
            spark = sparkline(self.loss_history)
            parts.append(Text(f"Loss: {spark}", style="yellow"))
            parts.append(Text(""))

        # --- GPU stats ---
        if self.latest_gpu:
            g = self.latest_gpu
            vram_used = g.get("vram_used", 0)
            vram_total = g.get("vram_total", 0)
            util = g.get("gpu_util", 0)
            temp = g.get("temperature", 0)
            power = g.get("power_draw", 0)
            gpu_line = (
                f"GPU: {util:.0f}%  "
                f"VRAM: {vram_used:.1f}/{vram_total:.1f} GB  "
                f"Temp: {temp:.0f}C  "
                f"Power: {power:.0f}W"
            )
            parts.append(Text(gpu_line, style="dim"))

        # --- Assemble panel ---
        from rich.console import Group
        panel = Panel(Group(*parts), title="[bold]mud-puppy[/bold]", border_style="blue")

        if self._live is not None:
            self._live.update(panel)

    def _render_complete(self, data: dict) -> None:
        """Render a green completion banner."""
        from rich.panel import Panel
        from rich.text import Text
        from rich.console import Group

        total_time = data.get("total_time", 0)
        total_steps = data.get("total_steps", 0)
        best_loss = data.get("best_loss")

        lines = [
            Text("Training Complete!", style="bold green"),
            Text(""),
            Text(f"  Total steps:  {total_steps}"),
            Text(f"  Total time:   {_format_eta(total_time)}"),
        ]
        if best_loss is not None:
            lines.append(Text(f"  Best loss:    {best_loss:.4f}"))

        panel = Panel(
            Group(*lines),
            title="[bold green]mud-puppy[/bold green]",
            border_style="green",
        )

        if self._live is not None:
            self._live.update(panel)
        else:
            # If not using Live, print directly
            from rich.console import Console
            Console().print(panel)
