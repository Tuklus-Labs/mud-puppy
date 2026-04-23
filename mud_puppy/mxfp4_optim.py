"""Stochastic-rounding optimizer wrapper for MXFP4 QAT.

Wraps any base optimizer (Adam, AdamW, SGD, etc.) and snaps
``MXFP4QATLinear`` weights to the MXFP4 grid via stochastic rounding
after each ``step()``. This keeps the master weights near the grid
throughout training, so at the end of training the round-to-nearest
deploy snap is approximately a no-op.

Why stochastic rounding specifically:
    * Deterministic round-to-nearest biases the master weight toward the
      nearest grid point every step. Over many steps, small gradients
      that don't cross the rounding threshold never move the weight --
      training stalls at sub-grid resolution.
    * Stochastic rounding is unbiased: ``E[snap(w)] = w``. Small
      gradients produce the right long-run update in expectation, even
      though each individual step lands on a grid point.
    * Reference: "Training Deep Neural Networks with 8-bit Floating
      Point Numbers" (Wang et al., 2018); "Adam Can Converge Without
      Any Modification On Update Rules" (Zhang et al., 2022) for why
      this composes with adaptive optimizers.

Usage::

    model = apply_mxfp4_qat(my_model)
    base_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt = MXFP4StochasticOptimizer(base_opt, model, snap_interval=1)

    for batch in data:
        loss = model(batch).loss
        loss.backward()
        opt.step()
        opt.zero_grad()

    opt.final_snap()     # deterministic round-to-nearest for deploy
    save_checkpoint(model)

The final deterministic snap makes the saved weights match exactly
what the forward pass produces on that state, so inference and training
eval agree bit-for-bit.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .mxfp4 import quantize_mxfp4
from .mxfp4_train import MXFP4QATLinear


class MXFP4StochasticOptimizer:
    """Optimizer wrapper that snaps MXFP4-QAT weights to the grid after each step.

    This is deliberately NOT a subclass of ``torch.optim.Optimizer`` --
    we wrap an existing optimizer to keep its state_dict, scheduler
    compatibility, and param_group semantics. Most trainer loops only
    call ``step()`` and ``zero_grad()``, which we forward.

    Attributes:
        optimizer: the wrapped base optimizer. ``state_dict``, parameter
            groups, etc. are accessed through it.
        model: model to scan for :class:`MXFP4QATLinear` modules.
        block_size: MXFP4 block size; must match what the modules use.
        snap_interval: snap every N steps. 1 (default) snaps every step;
            larger values amortize the cost at some accuracy risk.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        block_size: int = 32,
        snap_interval: int = 1,
    ) -> None:
        if snap_interval < 1:
            raise ValueError(f"snap_interval must be >= 1, got {snap_interval}")
        self.optimizer = optimizer
        self.model = model
        self.block_size = int(block_size)
        self.snap_interval = int(snap_interval)
        self._step_count = 0

    # ------------------------------------------------------------------
    # Torch optimizer-style API
    # ------------------------------------------------------------------

    def step(self, closure: Optional[Any] = None):
        """Run the wrapped optimizer's step, then stochastically snap."""
        loss = self.optimizer.step(closure)
        self._step_count += 1
        if self._step_count % self.snap_interval == 0:
            self._snap_weights(stochastic=True)
        return loss

    def zero_grad(self, *args, **kwargs):
        self.optimizer.zero_grad(*args, **kwargs)

    # ------------------------------------------------------------------
    # Passthroughs so schedulers / state_dict / param_groups work
    # ------------------------------------------------------------------

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
            "block_size": self.block_size,
            "snap_interval": self.snap_interval,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._step_count = int(state_dict.get("step_count", 0))
        self.block_size = int(state_dict.get("block_size", self.block_size))
        self.snap_interval = int(state_dict.get("snap_interval", self.snap_interval))

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    # ------------------------------------------------------------------
    # MXFP4-specific ops
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _snap_weights(self, stochastic: bool) -> int:
        """Snap every MXFP4QATLinear master weight to the grid.

        ``stochastic=True`` during training preserves unbiased dynamics;
        ``stochastic=False`` (for :meth:`final_snap`) gives deterministic
        deploy weights.

        Returns the number of layers snapped (for logging).
        """
        n = 0
        for module in self.model.modules():
            if isinstance(module, MXFP4QATLinear):
                snapped = quantize_mxfp4(
                    module.weight.data,
                    block_size=self.block_size,
                    stochastic=stochastic,
                )
                module.weight.data.copy_(snapped)
                n += 1
        return n

    def final_snap(self) -> int:
        """Deterministic round-to-nearest snap, for end-of-training deploy.

        Call this once after the last training step and before saving the
        checkpoint. Ensures the saved master weights are exactly what the
        forward pass produces, so deploy and training-eval agree.
        """
        return self._snap_weights(stochastic=False)

    @property
    def step_count(self) -> int:
        return self._step_count
