"""Per-layer RAM->GPU streaming for transformer models.

Implements a prefetch-ring design: at most `prefetch_layers` transformer
blocks are resident on GPU at any time. A dedicated h2d CUDA stream
copies the next block asynchronously while the compute stream executes
the current one, hiding transfer latency.

Architecture:
- Dedicated `h2d_stream` handles host-to-device copies; compute runs on
  whatever stream the trainer has installed as `torch.cuda.current_stream()`
  (PyTorch's default stream by default).
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


def _is_lora_param_name(name: str) -> bool:
    """Return True if a parameter name looks like a peft LoRA adapter.

    Matches names produced by peft.tuners.lora (e.g. "...lora_A.default.weight",
    "...lora_B.default.weight", "...lora_embedding_A.default").
    """
    return ("lora_A" in name) or ("lora_B" in name) or ("lora_embedding" in name)


def _pin_lora_to_gpu(layer: nn.Module, device: torch.device) -> None:
    """Force every LoRA parameter inside *layer* onto *device* in-place."""
    for pname, param in layer.named_parameters():
        if _is_lora_param_name(pname) and param.device != device:
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device)


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

    def __init__(self, model: nn.Module, prefetch_layers: int = 2,
                 training_method: str = "inference") -> None:
        """Initialise the LayerStreamer.

        Parameters
        ----------
        model:
            The model to stream. Transformer blocks are moved to CPU; all
            other modules (embeddings, lm_head, LoRA adapters) stay on GPU.
        prefetch_layers:
            Number of GPU ring slots (>= 1).
        training_method:
            The mud-puppy training method in use. Streaming is only safe when
            backward does NOT need the base layer weights (they get evicted from
            the ring after forward). This is true for ``"qlora"`` (frozen base,
            only adapters train), ``"inference"`` (no backward at all), and
            ``"embedding"`` (only the embedding head trains).

            For full fine-tuning, LoRA, or any method where the base weights
            participate in backward, you MUST keep the model fully on GPU --
            streaming will silently produce wrong gradients or crash because
            backward tries to access a layer that has been evicted from the
            ring.

            Raises ``NotImplementedError`` for unsafe method combinations.

        .. note::
            Thread safety: ``LayerStreamer`` is designed for single-threaded
            training loops. Do not call ``forward()`` from multiple threads
            simultaneously (``_current_layer_idx`` is not atomic).
        """
        _SAFE_STREAM_METHODS = {"qlora", "inference", "embedding"}
        if training_method not in _SAFE_STREAM_METHODS:
            raise NotImplementedError(
                f"LayerStreamer is not safe for training_method={training_method!r}. "
                "Streaming evicts base layer weights from GPU after forward; backward "
                "for full/lora/preference/rl/etc. needs those weights and will either "
                "crash or silently train on wrong gradients. "
                f"Safe methods: {sorted(_SAFE_STREAM_METHODS)}. "
                "Disable --stream or switch to --method qlora."
            )

        if prefetch_layers < 1:
            raise ValueError(
                f"prefetch_layers must be >= 1, got {prefetch_layers}. "
                "A zero-slot ring has nowhere to load the current layer."
            )

        self.model = model
        self.prefetch_layers = prefetch_layers

        if not torch.cuda.is_available():
            raise RuntimeError("LayerStreamer requires a CUDA/ROCm GPU")

        self.device = torch.device("cuda")
        self._cpu = torch.device("cpu")

        # Dedicated H2D stream so async copies can overlap with compute
        # running on torch.cuda.current_stream() (the trainer's main
        # stream -- PyTorch's default or a custom one it installed).
        self.h2d_stream = torch.cuda.Stream()

        # Discover transformer blocks.
        self._layers = _find_layers(model)
        n = len(self._layers)

        # Identify LoRA parameter names per layer. These stay GPU-resident
        # and are skipped when snapshotting / reloading CPU state_dicts.
        # Keys look like "self_attn.q_proj.lora_A.default.weight".
        self._lora_param_names: List[set] = []
        for layer in self._layers:
            names: set = set()
            for pname, _ in layer.named_parameters():
                if _is_lora_param_name(pname):
                    names.add(pname)
            self._lora_param_names.append(names)

        # Pinned CPU copies of each block's state_dict. Skip LoRA entries.
        self._cpu_weights: List[Dict[str, torch.Tensor]] = []
        for layer, lora_names in zip(self._layers, self._lora_param_names):
            pinned = {}
            for name, tensor in layer.state_dict().items():
                if name in lora_names:
                    continue  # LoRA stays on GPU; not part of streamed weights
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

        # Currently-executing layer (set by pre-forward hook). Used by
        # _find_free_slot to avoid evicting the in-flight layer.
        self._current_layer_idx: int = -1

        # Move all transformer layers to CPU; GPU-resident layers
        # will be loaded into slots on demand. LoRA adapters are
        # re-pinned to GPU immediately -- they are permanent residents.
        for layer in self._layers:
            layer.to(self._cpu)
            _pin_lora_to_gpu(layer, self.device)

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

        Never evicts the currently-executing layer (_current_layer_idx);
        that would tear out the weights from under an in-flight forward.
        """
        # Already loaded?
        for s, li in enumerate(self._slot_layer):
            if li == layer_idx:
                return s

        # Empty slot?
        for s, li in enumerate(self._slot_layer):
            if li == -1:
                return s

        # Pick an eviction victim. Constraints:
        #   1. Never evict the currently-executing layer.
        #   2. Never evict layer_idx itself (handled above by "already
        #      loaded" check, but defend anyway).
        # Preference: layers furthest behind the request point (executed
        # longest ago).
        current = getattr(self, "_current_layer_idx", -1)
        best = -1
        best_score = None
        for s, li in enumerate(self._slot_layer):
            if li == current or li == layer_idx:
                continue
            # Score: how far behind layer_idx is this slot's occupant?
            # Positive = already-executed (prefer these). Negative = future
            # prefetch (last resort).
            score = layer_idx - li
            if best_score is None or score > best_score:
                best_score = score
                best = s

        if best == -1:
            # Every slot holds either the current or requested layer; fall
            # back to slot 0 but do not evict. Caller will overwrite in
            # place via copy_.
            return 0

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

        # Load state_dict into the layer (GPU tensors). strict=False because
        # LoRA adapter keys are NOT in the CPU snapshot; they stay GPU-resident
        # and would be missing from gpu_sd.
        layer.load_state_dict(gpu_sd, strict=False)
        # Ensure layer is on GPU device.
        layer.to(self.device)
        # LoRA adapters should already be on GPU, but force-pin in case
        # load_state_dict or .to() shuffled anything.
        _pin_lora_to_gpu(layer, self.device)

    def _free_slot(self, slot: int) -> None:
        """Mark a ring slot as free and move the layer back to CPU."""
        layer_idx = self._slot_layer[slot]
        if layer_idx >= 0:
            # Move the layer's parameters back to CPU so VRAM is freed.
            self._layers[layer_idx].to(self._cpu)
            # Re-pin LoRA adapters to GPU -- they must NEVER leave the GPU
            # per v0.4 spec (Section A: LoRA stays permanently GPU-resident).
            _pin_lora_to_gpu(self._layers[layer_idx], self.device)
        self._slot_layer[slot] = -1
        self._ring_slots[slot] = None

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _make_pre_hook(self, idx: int):
        """Return the pre-forward hook for layer `idx`."""
        def hook(module: nn.Module, args: tuple) -> None:
            n_layers = len(self._layers)

            # Mark this layer as "in flight" so the eviction policy does
            # not rip its weights out while prefetching the next layer.
            self._current_layer_idx = idx

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
    def wrap(
        cls,
        model: nn.Module,
        prefetch_layers: int = 2,
        training_method: str = "inference",
    ) -> nn.Module:
        """Attach a LayerStreamer to model and store it as model._streamer.

        ``training_method`` must be one of the safe-for-streaming methods;
        ``__init__`` raises ``NotImplementedError`` for unsafe ones (e.g.
        ``"lora"``, ``"full"``, ``"preference"``). Callers in
        ``run_training`` must forward ``config.finetuning_method`` so the
        guard actually fires; defaulting to ``"inference"`` here used to
        silently bypass the check (pass-3 audit finding, 2026-04-22).

        Returns the original model (mutated in-place with hooks).
        """
        streamer = cls(
            model,
            prefetch_layers=prefetch_layers,
            training_method=training_method,
        )
        model._streamer = streamer
        return model
