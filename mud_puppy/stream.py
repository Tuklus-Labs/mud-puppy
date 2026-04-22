"""Per-layer RAM->GPU streaming for transformer models.

Implements a prefetch-ring design: at most `prefetch_layers` transformer
blocks are resident on GPU at any time. A dedicated h2d CUDA stream
copies the next block asynchronously while the compute stream executes
the current one, hiding transfer latency.

Architecture:
- Two CUDA streams: `compute_stream` (forward/backward) and `h2d_stream`
  (host-to-device copies).
- Ring of K GPU slots (K = prefetch_layers). Each slot holds one block's
  weight tensors.
- Pre-forward hook on block N: issue async H2D for block N+1 on h2d_stream;
  record a CUDA event; compute stream waits on that event before running N+1.
- Post-forward hook on block N: mark the oldest ring slot free when
  N >= prefetch_layers - 1 so the ring doesn't overflow.
- Embedding, final layernorm, and lm_head stay permanently on GPU (small,
  always needed). LoRA adapters follow the same rule.

Thread safety: designed for single-threaded training loops. Do not call
forward() from multiple threads simultaneously.

Usage::

    from mud_puppy.stream import LayerStreamer

    streamer = LayerStreamer(model, prefetch_layers=2)
    # model.model.layers are now wrapped; forward/backward work unchanged.
    # Attach to model as _streamer for introspection:
    model._streamer = streamer
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


def _find_layers(model: nn.Module) -> List[nn.Module]:
    """Locate the main transformer block list in a model.

    Supports LLaMA-style (`model.model.layers`) and GPT-2-style
    (`model.transformer.h`). Returns an empty list if neither is found.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    return []


class LayerStreamer:
    """Stream transformer blocks between CPU pinned RAM and GPU.

    Keeps at most `prefetch_layers` transformer blocks resident on GPU at any
    time. Uses a dedicated h2d CUDA stream for async copies and synchronises
    to the compute stream via CUDA events so forward pass never stalls on
    copy when prefetch is working.

    Only transformer blocks are streamed. Embedding, final layernorm, and
    lm_head stay GPU-resident always. If LoRA adapters are attached, they
    are pinned to the streamed layers and move with them.
    """

    def __init__(self, model: nn.Module, prefetch_layers: int = 2) -> None:
        self.model = model
        self.prefetch_layers = prefetch_layers

        if not torch.cuda.is_available():
            raise RuntimeError("LayerStreamer requires a CUDA/ROCm GPU")

        self.device = torch.device("cuda")
        self._cpu = torch.device("cpu")

        # Two CUDA streams: one for compute, one for H2D copies.
        self.compute_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()

        # Discover transformer blocks.
        self._layers = _find_layers(model)
        n = len(self._layers)

        # Pinned CPU copies of each block's state_dict.
        self._cpu_weights: List[Dict[str, torch.Tensor]] = []
        for layer in self._layers:
            pinned = {}
            for name, tensor in layer.state_dict().items():
                pinned_t = torch.empty_like(tensor, device=self._cpu, pin_memory=True)
                pinned_t.copy_(tensor)
                pinned[name] = pinned_t
            self._cpu_weights.append(pinned)

        # Ring slots: each slot is a dict of GPU tensors matching one block.
        # We allocate K slots where K = prefetch_layers.
        self._ring_slots: List[Optional[Dict[str, torch.Tensor]]] = []
        for _ in range(self.prefetch_layers):
            self._ring_slots.append(None)

        # Which layer index is loaded into each ring slot (-1 = empty).
        self._slot_layer: List[int] = [-1] * self.prefetch_layers

        # CUDA events for synchronisation.
        self._events: List[Optional[torch.cuda.Event]] = [None] * n

        # Counters for statistics.
        self._h2d_bytes: int = 0
        self._h2d_calls: int = 0
        self._prefetch_hits: int = 0
        self._prefetch_misses: int = 0
        self._start_time: float = time.monotonic()

        # Move all transformer layers to CPU; GPU-resident layers
        # will be loaded into slots on demand.
        for layer in self._layers:
            layer.to(self._cpu)

        # Move permanent-resident modules to GPU.
        self._move_permanent_to_gpu()

        # Install forward hooks on each layer.
        self._hook_handles = []
        for idx, layer in enumerate(self._layers):
            pre = layer.register_forward_pre_hook(self._make_pre_hook(idx))
            post = layer.register_forward_hook(self._make_post_hook(idx))
            self._hook_handles.extend([pre, post])

    # ------------------------------------------------------------------
    # Permanent-resident modules
    # ------------------------------------------------------------------

    def _move_permanent_to_gpu(self) -> None:
        """Move embedding, lm_head, and final norm to GPU permanently."""
        model = self.model

        # Try HuggingFace API first.
        moved_embed = False
        if hasattr(model, "get_input_embeddings"):
            try:
                emb = model.get_input_embeddings()
                if emb is not None:
                    emb.to(self.device)
                    moved_embed = True
            except Exception:
                pass

        # Fallback: common embedding attribute names.
        if not moved_embed:
            for attr in ("embed", "embed_tokens", "word_embeddings", "wte"):
                candidate = getattr(model, attr, None)
                if candidate is None:
                    # check under model.model.*
                    inner = getattr(model, "model", None)
                    if inner is not None:
                        candidate = getattr(inner, attr, None)
                if isinstance(candidate, nn.Module):
                    candidate.to(self.device)
                    moved_embed = True
                    break

        if hasattr(model, "lm_head"):
            model.lm_head.to(self.device)

        # LLaMA-style final norm
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            model.model.norm.to(self.device)
        # GPT-2 style final norm
        if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            model.transformer.ln_f.to(self.device)

    # ------------------------------------------------------------------
    # Ring management
    # ------------------------------------------------------------------

    def _find_free_slot(self, layer_idx: int) -> int:
        """Return a ring slot index to use for layer_idx.

        Preference: a slot already holding layer_idx (reuse), then an
        empty slot, then the oldest evictable slot. When an occupied slot
        is selected, _free_slot is called on it first so the occupant's
        layer is moved back to CPU.
        """
        # Already loaded?
        for s, li in enumerate(self._slot_layer):
            if li == layer_idx:
                return s

        # Empty slot?
        for s, li in enumerate(self._slot_layer):
            if li == -1:
                return s

        # Evict the slot whose layer is furthest behind us.
        # Prefer slots holding layers < layer_idx - 1 (already executed).
        best = 0
        best_dist = -1
        for s, li in enumerate(self._slot_layer):
            if li < layer_idx - 1:
                dist = layer_idx - li
                if dist > best_dist:
                    best_dist = dist
                    best = s
        # Free the evicted slot so the old layer goes back to CPU.
        if self._slot_layer[best] >= 0:
            self._free_slot(best)
        return best

    def _load_layer_into_slot(self, layer_idx: int, slot: int,
                              stream: Optional[torch.cuda.Stream] = None) -> None:
        """Copy CPU pinned weights for layer_idx into GPU slot, async on `stream`."""
        if stream is None:
            stream = self.h2d_stream

        cpu_sd = self._cpu_weights[layer_idx]
        layer = self._layers[layer_idx]

        with torch.cuda.stream(stream):
            gpu_sd: Dict[str, torch.Tensor] = {}
            for name, cpu_t in cpu_sd.items():
                gpu_t = torch.empty_like(cpu_t, device=self.device)
                gpu_t.copy_(cpu_t, non_blocking=True)
                self._h2d_bytes += cpu_t.nbytes
                gpu_sd[name] = gpu_t

            self._ring_slots[slot] = gpu_sd
            self._slot_layer[slot] = layer_idx
            self._h2d_calls += 1

        # Patch the layer's parameters to point to the GPU tensors.
        # We do this after the copy completes (h2d_stream event will sync).
        # Store for patching after sync.
        self._pending_patch = (layer_idx, slot)

    def _patch_layer_weights(self, layer_idx: int, slot: int) -> None:
        """Redirect layer parameter storage to the GPU slot tensors."""
        layer = self._layers[layer_idx]
        gpu_sd = self._ring_slots[slot]
        if gpu_sd is None:
            return

        # Load state_dict into the layer (GPU tensors).
        layer.load_state_dict(gpu_sd, strict=True)
        # Ensure layer is on GPU device.
        layer.to(self.device)

    def _free_slot(self, slot: int) -> None:
        """Mark a ring slot as free and move the layer back to CPU."""
        layer_idx = self._slot_layer[slot]
        if layer_idx >= 0:
            # Move the layer's parameters back to CPU so VRAM is freed.
            self._layers[layer_idx].to(self._cpu)
        self._slot_layer[slot] = -1
        self._ring_slots[slot] = None

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _make_pre_hook(self, idx: int):
        """Return the pre-forward hook for layer `idx`."""
        def hook(module: nn.Module, args: tuple) -> None:
            n_layers = len(self._layers)

            # 1. Ensure current layer (idx) is loaded and synced.
            slot = self._find_free_slot(idx)
            if self._slot_layer[slot] != idx:
                # Load synchronously if not already prefetched.
                self._load_layer_into_slot(idx, slot, stream=self.h2d_stream)
                self._prefetch_misses += 1
                # Sync: wait for h2d to finish before patching.
                torch.cuda.current_stream().wait_stream(self.h2d_stream)
            else:
                self._prefetch_hits += 1
                # Wait for any pending prefetch to finish.
                evt = self._events[idx]
                if evt is not None:
                    torch.cuda.current_stream().wait_event(evt)

            self._patch_layer_weights(idx, slot)

            # 2. Issue prefetch for next layer asynchronously.
            next_idx = idx + 1
            if next_idx < n_layers:
                next_slot = self._find_free_slot(next_idx)
                if self._slot_layer[next_slot] != next_idx:
                    self._load_layer_into_slot(next_idx, next_slot,
                                               stream=self.h2d_stream)
                    # Record event on h2d_stream for next layer to wait on.
                    evt = torch.cuda.Event()
                    evt.record(self.h2d_stream)
                    self._events[next_idx] = evt

        return hook

    def _make_post_hook(self, idx: int):
        """Return the post-forward hook for layer `idx`."""
        def hook(module: nn.Module, args: tuple, output: Any) -> None:
            # Eviction now happens eagerly in _find_free_slot when a new
            # layer claims a ring slot. The post-hook is kept as a hook
            # attachment point for future backward-pass coordination.
            pass

        return hook

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return a snapshot of streaming statistics."""
        elapsed = time.monotonic() - self._start_time
        bw = self._h2d_bytes / elapsed / 1e9 if elapsed > 0 else 0.0
        total = self._prefetch_hits + self._prefetch_misses
        hit_rate = self._prefetch_hits / total if total > 0 else 0.0
        resident = sum(1 for s in self._slot_layer if s >= 0)

        return {
            "layers_total": len(self._layers),
            "layers_resident": resident,
            "h2d_bandwidth_gbps": bw,
            "prefetch_hit_rate": hit_rate,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Remove all installed forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    # ------------------------------------------------------------------
    # Convenience classmethod
    # ------------------------------------------------------------------

    @classmethod
    def wrap(cls, model: nn.Module, prefetch_layers: int = 2) -> nn.Module:
        """Attach a LayerStreamer to model and store it as model._streamer.

        Returns the original model (mutated in-place with hooks).
        """
        streamer = cls(model, prefetch_layers=prefetch_layers)
        model._streamer = streamer
        return model
