# Mud-Puppy Training Monitor -- Design Document

**Date:** 2026-02-20
**Style:** Tempest Vector-Neon (arcade CRT command center aesthetic)

## Overview

Real-time training visualizer for mud-puppy. Web dashboard primary, Rich TUI fallback. Connects via WebSocket callback integrated into the HuggingFace Trainer pipeline.

## Architecture

Three new files in `mud_puppy/`:

| File | Purpose |
|------|---------|
| `monitor.py` | aiohttp server + MonitorCallback + GPU telemetry |
| `tui.py` | Rich Live fallback for SSH/headless |
| `static/dashboard.html` | Self-contained HTML/JS/CSS dashboard (no build step) |

### Data Flow

```
Trainer -> MonitorCallback.on_log() -> WebSocket broadcast -> Browser Canvas charts
                                    -> TUI Live update (if --monitor-tui)
         MonitorCallback.on_train_begin() -> Config/model metadata
         Background coroutine (1Hz) -> GPU telemetry -> WebSocket broadcast
```

### Port

5980 (default), override with `--monitor-port`.

## CLI Integration

```bash
mud-puppy model data.jsonl --monitor           # Web dashboard
mud-puppy model data.jsonl --monitor-tui       # TUI only
mud-puppy model data.jsonl --monitor --monitor-port 5981
```

## Metrics

### Training Metrics (from on_log, every 10 steps)

- loss, learning_rate, grad_norm
- epoch, global_step
- tokens_seen, samples_seen
- samples_per_second (computed)

### Eval Metrics (from on_evaluate)

- eval_loss
- eval metrics dict

### GPU Telemetry (1Hz background task)

- VRAM allocated / total (bytes)
- GPU utilization %
- Temperature (C)
- Power draw (W)
- tok/s throughput

### LoRA Weight Norms (every 50 steps if LoRA/QLoRA)

- Per-layer L2 norm of LoRA A and B matrices
- Tracks drift over training

### Token Stats (every on_log)

- Padding ratio (padded tokens / total tokens in batch)
- Mean sequence length
- Token throughput (tok/s)

## WebSocket Protocol

```json
{"type": "config", "data": {"model": "...", "method": "lora", "dataset_size": 12400, ...}}
{"type": "metrics", "timestamp": 1708401234.5, "data": {"step": 100, "loss": 1.23, ...}}
{"type": "gpu", "timestamp": 1708401234.5, "data": {"vram_used": 14.2e9, "vram_total": 24e9, ...}}
{"type": "eval", "timestamp": 1708401234.5, "data": {"eval_loss": 1.01, ...}}
{"type": "lora_norms", "timestamp": 1708401234.5, "data": {"layers": {"0": {"A": 0.12, "B": 0.08}, ...}}}
{"type": "tokens", "timestamp": 1708401234.5, "data": {"padding_ratio": 0.15, "mean_seq_len": 412, "tok_s": 1247}}
{"type": "complete", "data": {"total_time": 3600, "best_loss": 0.45, ...}}
```

## Dashboard Layout (Tempest Vector-Neon)

### Color Palette

| Element | Color |
|---------|-------|
| Background | #0a0e1a |
| Panel bg | #0d1420 |
| Grid lines | #1a2a3a |
| Panel border | #1a3a4a (dim cyan) |
| Loss trace | #39ff14 (phosphor green) |
| LR trace | #00ffff (neon cyan) |
| Grad norm trace | #ffd700 (amber) |
| Eval loss trace | #ff00ff (magenta) |
| GPU gauge fill | #00ffff (cyan), #ff3333 (red zone >90%) |
| Readout text | #e0e0e0 (bright white) |
| Labels | #607080 (dim) to #00ffff (accent) |

### Panel Grid

```
+------------------------------------------------------------------+
|  MUD-PUPPY MONITOR               [method] [model] [connected]    |
+------------------------------------------------------------------+
| STEP     | EPOCH    | LOSS     | LR       | GRAD NORM | ETA      |
| 1,240    | 1.3/3    | 0.847    | 1.4E-5   | 0.208     | 14:22    |
+----------+----------+----------+----------+-----------+----------+
|                                |                                  |
|  LOSS CURVE (green)            |  LEARNING RATE (cyan)            |
|  [rolling 500pt canvas]        |  [rolling 500pt canvas]          |
|                                |                                  |
+--------------------------------+----------------------------------+
|                                |                                  |
|  GRADIENT NORM (amber)         |  EVAL LOSS (magenta)             |
|  [rolling 500pt canvas]        |  [scatter + line, if eval on]    |
|                                |                                  |
+--------------------------------+----------------------------------+
|  VRAM [arc gauge] | UTIL [arc] | TEMP [bar]  | POWER [bar]       |
|  14.2/24.0 GB     | 97%        | 72C         | 284W              |
+-------------------+------------+-------------+-------------------+
|  TOKEN THROUGHPUT              |  LORA WEIGHT NORMS               |
|  [sparkline tok/s]             |  [multi-line per-layer chart]    |
|  padding: 15% | seq: 412      |  or heatmap grid                 |
+--------------------------------+----------------------------------+
|  MODEL: Qwen2.5-3B | LoRA r=16 a=32 | INT4 bf16 | 12,400 samples|
+------------------------------------------------------------------+
```

### Visual Effects

- **Glow:** `ctx.shadowBlur = 8; ctx.shadowColor = traceColor` on all chart lines
- **Grid:** Thin 1px lines at #1a2a3a, major gridlines slightly brighter
- **Gauges:** Canvas arc() with gradient from cyan to red at threshold
- **Scan lines:** Optional subtle CSS overlay for CRT effect
- **Font:** `'Share Tech Mono', 'JetBrains Mono', monospace`
- **Borders:** 1px solid, dim cyan, no border-radius

### Responsiveness

CSS Grid with `auto-fit` columns. Minimum panel width 300px. Stacks vertically on narrow screens.

## TUI Fallback (Rich)

```
MUD-PUPPY MONITOR [lora] [Qwen2.5-3B] [step 1240/9600]
+----------+-------+----------+----------+--------+-------+
| STEP     | EPOCH | LOSS     | LR       | GNORM  | ETA   |
| 1,240    | 1.3/3 | 0.847    | 1.4E-5   | 0.208  | 14:22 |
+----------+-------+----------+----------+--------+-------+
Loss: [sparkline ~~~~~~~~]  GPU: 14.2/24.0 GB  97%  72C  284W
```

Minimal, updates in-place via `rich.live.Live`. Sparkline via `rich.text.Text` with block characters.

## Implementation Notes

- aiohttp chosen over FastAPI: lighter, async-native, no uvicorn dep
- Dashboard HTML served from `mud_puppy/static/dashboard.html` via `importlib.resources` or `__file__` relative path
- GPU telemetry: try `pynvml` first (works on ROCm via rocm-smi shim), fall back to subprocess `rocm-smi --json`
- LoRA norms: iterate `model.named_parameters()` filtering for `lora_` prefix, compute `.norm().item()`
- Token stats: hook into data collator or compute from batch in `on_step_end`
- Auto-open browser: `webbrowser.open(f"http://localhost:{port}")` on first WS connection
- Reconnect: JS auto-reconnects WebSocket with exponential backoff
