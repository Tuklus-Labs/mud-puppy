"""ZeRO-Offload implementation for ROCm.

Offloads optimizer states and gradients to CPU RAM, enabling training of
models that wouldn't otherwise fit in VRAM.

Key features:
- Optimizer states (momentum, variance) stored on CPU
- Gradients accumulated on GPU, copied to CPU before optimizer step
- Parameters updated on CPU, copied back to GPU
- Pin memory for faster CPU<->GPU transfers
- Compatible with any PyTorch optimizer

Memory savings for AdamW:
- Without offload: model (2x) + optimizer (8x) + gradients (2x) = 12x params in VRAM
- With offload: model (2x) + gradients (2x) = 4x params in VRAM
  (optimizer states on CPU: 8x params in RAM)

For a 20B model in bf16:
- Without: ~240GB VRAM needed (impossible)
- With offload: ~80GB VRAM for forward/backward (still too much)
- With offload + gradient checkpointing: ~40GB VRAM (borderline on 24GB)
- With offload + grad checkpoint + 4-bit: ~15GB VRAM (fits!)

Usage:
    from mud_puppy.zero_offload import CPUOffloadOptimizer, OffloadConfig

    config = OffloadConfig(
        offload_optimizer=True,
        offload_gradients=True,
        pin_memory=True,
    )

    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = CPUOffloadOptimizer(base_optimizer, config)

    # Training loop
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()  # Handles CPU<->GPU transfers automatically
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import logging

log = logging.getLogger(__name__)


@dataclass
class OffloadConfig:
    """Configuration for CPU offloading."""

    # What to offload
    offload_optimizer: bool = True  # Offload optimizer states to CPU
    offload_gradients: bool = True  # Accumulate gradients on CPU
    offload_params: bool = False    # Keep params on CPU (extreme mode)

    # Memory optimization
    pin_memory: bool = True         # Use pinned memory for faster transfers
    async_transfer: bool = True     # Use async GPU<->CPU copies where possible

    # Computation
    cpu_optimizer_step: bool = True # Run optimizer.step() on CPU
    overlap_comm: bool = True       # Overlap gradient transfer with backward

    # Precision
    grad_dtype: torch.dtype = torch.float32  # Gradient accumulation dtype on CPU
    state_dtype: torch.dtype = torch.float32 # Optimizer state dtype on CPU


class CPUOffloadOptimizer:
    """Wrapper that offloads optimizer states and gradients to CPU.

    This enables training models that wouldn't fit in VRAM by keeping
    optimizer states (which are 2x model size for Adam) on CPU RAM.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: OffloadConfig = None,
    ):
        self.optimizer = optimizer
        self.config = config or OffloadConfig()
        self._cpu = torch.device("cpu")
        self._gpu = torch.device("cuda") if torch.cuda.is_available() else self._cpu

        # CPU storage for optimizer states
        self._cpu_states: Dict[int, Dict[str, torch.Tensor]] = {}

        # CPU storage for gradients
        self._cpu_grads: Dict[int, torch.Tensor] = {}

        # Pinned memory buffers for faster transfers
        self._pinned_buffers: Dict[int, torch.Tensor] = {}

        # Track which params we're managing
        self._param_ids: List[int] = []
        self._param_map: Dict[int, nn.Parameter] = {}

        # Async grad transfer stream (D2H runs here while forward proceeds).
        self._grad_stream: Optional[Any] = None
        if torch.cuda.is_available():
            self._grad_stream = torch.cuda.Stream()
        self._grad_sync_event: Optional[Any] = None

        # Initialize
        self._setup_offload()

    def _setup_offload(self):
        """Initialize CPU storage for optimizer states."""
        log.info("Setting up ZeRO-Offload for %d parameter groups",
                 len(self.optimizer.param_groups))

        total_params = 0
        total_state_bytes = 0

        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue

                pid = id(param)
                self._param_ids.append(pid)
                self._param_map[pid] = param
                total_params += param.numel()

                # Allocate CPU gradient storage
                if self.config.offload_gradients:
                    cpu_grad = torch.zeros(
                        param.shape,
                        dtype=self.config.grad_dtype,
                        device=self._cpu,
                        pin_memory=self.config.pin_memory,
                    )
                    self._cpu_grads[pid] = cpu_grad
                    total_state_bytes += cpu_grad.numel() * cpu_grad.element_size()

                # Allocate pinned buffer for gradient transfer
                if self.config.pin_memory:
                    pinned = torch.zeros(
                        param.shape,
                        dtype=self.config.grad_dtype,
                        device=self._cpu,
                        pin_memory=True,
                    )
                    self._pinned_buffers[pid] = pinned

        log.info("ZeRO-Offload initialized: %d params, %.2f GB CPU allocation",
                 total_params, total_state_bytes / 1e9)

    def _move_state_to_cpu(self, pid: int, state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Move optimizer state tensors to CPU."""
        cpu_state = {}
        for key, val in state.items():
            if isinstance(val, torch.Tensor):
                if self.config.pin_memory:
                    cpu_tensor = torch.empty(
                        val.shape,
                        dtype=self.config.state_dtype,
                        device=self._cpu,
                        pin_memory=True,
                    )
                    cpu_tensor.copy_(val)
                else:
                    cpu_tensor = val.to(self._cpu, dtype=self.config.state_dtype)
                cpu_state[key] = cpu_tensor
            else:
                cpu_state[key] = val
        return cpu_state

    def _move_state_to_gpu(self, cpu_state: Dict[str, Any], dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Move optimizer state tensors back to GPU."""
        gpu_state = {}
        for key, val in cpu_state.items():
            if isinstance(val, torch.Tensor):
                gpu_state[key] = val.to(self._gpu, dtype=dtype, non_blocking=True)
            else:
                gpu_state[key] = val
        return gpu_state

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients, optionally setting to None for memory savings."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

        # Also zero CPU gradient accumulators
        if self.config.offload_gradients:
            for cpu_grad in self._cpu_grads.values():
                cpu_grad.zero_()

    def _transfer_grads_to_cpu(self):
        """Copy gradients from GPU to CPU storage, async on _grad_stream."""
        stream = self._grad_stream
        if stream is not None and self.config.async_transfer:
            # Phase 1: launch all async D2H copies on the dedicated stream.
            copies: List = []
            with torch.cuda.stream(stream):
                for pid, param in self._param_map.items():
                    if param.grad is None:
                        continue
                    if self.config.pin_memory and pid in self._pinned_buffers:
                        pinned = self._pinned_buffers[pid]
                        pinned.copy_(param.grad, non_blocking=True)
                        copies.append((pid, pinned))
                    else:
                        cpu_grad = param.grad.to(self._cpu, non_blocking=True)
                        copies.append((pid, cpu_grad))
                    param.grad = None

            # Phase 2: block the CPU until all D2H copies on the stream have
            # completed before reading any pinned/cpu buffer. Without this
            # synchronize, the .add_() below can read from a pinned buffer
            # while the GPU copy is still in flight, silently accumulating
            # stale or zero data.
            stream.synchronize()

            for pid, buf in copies:
                self._cpu_grads[pid].add_(buf)

            # Record an event so step()'s existing wait path remains valid
            # (now purely informational since we already synchronized above).
            evt = torch.cuda.Event()
            evt.record(stream)
            self._grad_sync_event = evt
        else:
            for pid, param in self._param_map.items():
                if param.grad is None:
                    continue
                if self.config.pin_memory and pid in self._pinned_buffers:
                    pinned = self._pinned_buffers[pid]
                    pinned.copy_(param.grad)
                    self._cpu_grads[pid].add_(pinned)
                else:
                    self._cpu_grads[pid].add_(param.grad.to(self._cpu))
                param.grad = None
            self._grad_sync_event = None

    def _run_optimizer_on_cpu(self):
        """Run optimizer step on CPU with offloaded states.

        When the wrapped optimizer is AdamW and the PyTorch build supports
        foreach, rebuilds it as AdamW(foreach=True) on first invocation for
        ~2x CPU step speedup via vectorised foreach fusion.
        """
        # One-time switch to foreach AdamW when possible.
        if not getattr(self, "_foreach_checked", False):
            self._foreach_checked = True
            opt = self.optimizer
            if isinstance(opt, torch.optim.AdamW) and getattr(opt, "foreach", None) is not False:
                try:
                    # Rebuild preserving ALL per-group hyperparameters
                    # (lr, weight_decay, eps, betas, amsgrad, etc.). Passing
                    # opt.param_groups to the constructor picks up only the
                    # params list; subsequent add_param_group merges each
                    # group's full dict so LR scheduler state, per-group
                    # weight_decay overrides (common with LoRA), and other
                    # tunables are preserved.
                    first_group = opt.param_groups[0]
                    new_opt = torch.optim.AdamW(
                        first_group["params"],
                        lr=first_group.get("lr", 1e-3),
                        betas=first_group.get("betas", (0.9, 0.999)),
                        eps=first_group.get("eps", 1e-8),
                        weight_decay=first_group.get("weight_decay", 0.0),
                        amsgrad=first_group.get("amsgrad", False),
                        foreach=True,
                    )
                    # Merge remaining attributes of the first group that the
                    # AdamW constructor might have normalised. Skip keys
                    # that the new (foreach) optimizer already set itself;
                    # copying the old group's foreach=None would wipe out
                    # the foreach=True switch we just asked for.
                    _FOREACH_OWNED = {"foreach", "fused", "capturable"}
                    for key, val in first_group.items():
                        if key == "params" or key in _FOREACH_OWNED:
                            continue
                        new_opt.param_groups[0][key] = val

                    # Append subsequent groups verbatim so per-group
                    # hyperparameter overrides survive. Strip foreach-owned
                    # keys so add_param_group uses the new optimizer's
                    # defaults (foreach=True) rather than the old None.
                    for group in opt.param_groups[1:]:
                        clean = {
                            k: v for k, v in group.items()
                            if k not in _FOREACH_OWNED
                        }
                        new_opt.add_param_group(clean)

                    # Copy existing state (may be empty on first step).
                    new_opt.state.update(opt.state)
                    self.optimizer = new_opt
                    log.info("CPUOffloadOptimizer: switched to foreach=True AdamW")
                except Exception as exc:
                    log.debug("foreach AdamW switch failed: %s", exc)

        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue

                pid = id(param)
                state = self.optimizer.state[param]

                # First step: optimizer state doesn't exist yet
                if len(state) == 0:
                    continue

                # Move state to CPU if not already
                if pid not in self._cpu_states:
                    self._cpu_states[pid] = self._move_state_to_cpu(pid, state)

                # Point optimizer state to CPU tensors
                for key, val in self._cpu_states[pid].items():
                    if isinstance(val, torch.Tensor):
                        state[key] = val

        # Now run the actual optimizer step on CPU
        # We need to temporarily move params to CPU
        gpu_params = {}
        saved_grads = {}
        for pid, param in self._param_map.items():
            gpu_params[pid] = param.data
            saved_grads[pid] = param.grad  # may be None
            # Move param to CPU for optimizer
            param.data = param.data.to(self._cpu, dtype=self.config.state_dtype)
            # Source the gradient.
            if self.config.offload_gradients and pid in self._cpu_grads:
                # CPU accumulator, already populated by _transfer_grads_to_cpu.
                param.grad = self._cpu_grads[pid]
            elif saved_grads[pid] is not None:
                # Not offloading gradients: pull the live GPU grad directly
                # down to CPU for this step. Without this the CPU optimizer
                # either blew up on device mismatch or silently routed to
                # the zero-initialised _cpu_grads and trained on garbage.
                param.grad = saved_grads[pid].detach().to(
                    self._cpu, dtype=self.config.state_dtype
                )
            else:
                param.grad = None

        # Run optimizer
        self.optimizer.step()

        # Move params back to GPU
        for pid, param in self._param_map.items():
            original_dtype = gpu_params[pid].dtype
            param.data = param.data.to(self._gpu, dtype=original_dtype, non_blocking=True)
            param.grad = None

        # Update CPU state storage
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                pid = id(param)
                state = self.optimizer.state[param]
                if len(state) > 0:
                    self._cpu_states[pid] = self._move_state_to_cpu(pid, state)

    def step(self, closure: Optional[Callable] = None):
        """Perform optimizer step with CPU offloading."""
        if closure is not None:
            raise NotImplementedError("Closure not supported with CPU offload")

        # Transfer gradients to CPU (async when configured).
        if self.config.offload_gradients:
            self._transfer_grads_to_cpu()

        # CRITICAL: block the CPU thread until the D2H grad transfer on the
        # dedicated stream has actually completed. wait_event on the current
        # stream only serialises GPU work, not the Python reader. Without
        # this, _run_optimizer_on_cpu reads zero/stale pinned buffers and
        # silently trains on wrong gradients.
        if (
            self.config.offload_gradients
            and self.config.async_transfer
            and self._grad_stream is not None
        ):
            self._grad_stream.synchronize()
        self._grad_sync_event = None  # no-op bookkeeping reset

        # Run optimizer on CPU
        if self.config.cpu_optimizer_step:
            self._run_optimizer_on_cpu()
        else:
            # Run on GPU (states will be moved on first step)
            self.optimizer.step()

        # Synchronize
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def state_dict(self) -> Dict:
        """Return optimizer state dict with CPU states."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "cpu_states": self._cpu_states,
            "config": self.config,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load optimizer state including CPU states."""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._cpu_states = state_dict.get("cpu_states", {})

    @property
    def param_groups(self):
        """Access underlying optimizer param groups."""
        return self.optimizer.param_groups


def wrap_optimizer_for_offload(
    optimizer: Optimizer,
    offload_optimizer: bool = True,
    offload_gradients: bool = True,
    pin_memory: bool = True,
) -> CPUOffloadOptimizer:
    """Convenience wrapper to add CPU offloading to any optimizer.

    Args:
        optimizer: Base PyTorch optimizer
        offload_optimizer: Offload optimizer states to CPU
        offload_gradients: Accumulate gradients on CPU
        pin_memory: Use pinned memory for transfers

    Returns:
        Wrapped optimizer with CPU offloading
    """
    config = OffloadConfig(
        offload_optimizer=offload_optimizer,
        offload_gradients=offload_gradients,
        pin_memory=pin_memory,
    )
    return CPUOffloadOptimizer(optimizer, config)


# Convenience function for trainer.py integration
def create_offload_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    offload: bool = True,
) -> CPUOffloadOptimizer:
    """Create an AdamW optimizer with CPU offloading.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        offload: Whether to enable CPU offloading

    Returns:
        Optimizer (with offloading if enabled)
    """
    base_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    if offload:
        return wrap_optimizer_for_offload(base_optimizer)
    return base_optimizer
