"""Correctness tests for the stochastic-rounding optimizer wrapper.

These gate three properties that matter for real training:
    1. After each step, MXFP4QATLinear master weights sit on the grid.
    2. Stochastic snap is unbiased in expectation (same statistical
       guarantee as the underlying quantize_mxfp4 stochastic path).
    3. Training loop still converges when the optimizer is wrapped.

If any of these fail, long training runs drift off the grid, distort
gradient statistics, or stop learning entirely -- all subtle and
expensive failure modes to diagnose at scale.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mud_puppy.mxfp4 import E2M1_POSITIVE_VALUES, quantize_mxfp4_with_scale, _e8m0_decode
from mud_puppy.mxfp4_optim import MXFP4StochasticOptimizer
from mud_puppy.mxfp4_train import MXFP4QATLinear, apply_mxfp4_qat


def _all_weights_on_grid(module: MXFP4QATLinear) -> bool:
    """True iff every master weight block has SOME shared power-of-2 scale
    under which all element magnitudes are E2M1 values.

    We can't reconstruct the exact scale used at snap time (it was based
    on the pre-snap block max, which we no longer know). Instead we try
    every plausible scale candidate -- each block element magnitude
    divided by each E2M1 non-zero level -- and accept the block if ANY
    candidate is a power of 2 that explains every element.
    """
    import math

    w = module.weight.detach().to(torch.float32)
    blocks = w.reshape(-1, module.block_size)
    pos_levels = [lvl for lvl in E2M1_POSITIVE_VALUES if lvl > 0]

    for block in blocks:
        abs_block = block.abs()
        nonzero = abs_block[abs_block > 0]
        if nonzero.numel() == 0:
            continue  # all-zero block is trivially on any grid
        # Candidate scales: any nonzero element / any positive E2M1 level.
        # The true scale is among these (since each element = level * scale).
        found = False
        for v in nonzero.tolist():
            for lvl in pos_levels:
                scale = v / lvl
                log2s = math.log2(scale)
                if abs(log2s - round(log2s)) > 1e-4:
                    continue  # scale not a power of 2
                # Verify every block element is a valid grid point at this scale.
                all_ok = True
                for u in block.tolist():
                    normalized = abs(u) / scale
                    if not any(abs(normalized - l) < 1e-3 for l in E2M1_POSITIVE_VALUES):
                        all_ok = False
                        break
                if all_ok:
                    found = True
                    break
            if found:
                break
        if not found:
            return False
    return True


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_rejects_invalid_snap_interval() -> None:
    model = nn.Sequential(MXFP4QATLinear(8, 4, bias=False))
    base = torch.optim.SGD(model.parameters(), lr=1e-2)
    with pytest.raises(ValueError):
        MXFP4StochasticOptimizer(base, model, snap_interval=0)
    with pytest.raises(ValueError):
        MXFP4StochasticOptimizer(base, model, snap_interval=-3)


# ---------------------------------------------------------------------------
# Post-step grid membership
# ---------------------------------------------------------------------------


def test_weights_on_grid_after_each_step() -> None:
    """Every step must leave master weights on the MXFP4 grid."""
    torch.manual_seed(0)
    layer = MXFP4QATLinear(32, 16, bias=False, block_size=32)
    base = torch.optim.SGD(layer.parameters(), lr=1e-2)
    opt = MXFP4StochasticOptimizer(base, layer, snap_interval=1)

    x = torch.randn(4, 32)
    for _ in range(5):
        y = layer(x)
        y.sum().backward()
        opt.step()
        opt.zero_grad()
        assert _all_weights_on_grid(layer), "weights drifted off MXFP4 grid after step"


def test_snap_interval_larger_than_one() -> None:
    """With interval=3, weights snap every 3rd step."""
    torch.manual_seed(1)
    layer = MXFP4QATLinear(32, 16, bias=False, block_size=32)
    base = torch.optim.SGD(layer.parameters(), lr=1e-2)
    opt = MXFP4StochasticOptimizer(base, layer, snap_interval=3)

    x = torch.randn(4, 32)
    # Run 2 steps -- no snap yet
    for _ in range(2):
        layer(x).sum().backward()
        opt.step()
        opt.zero_grad()
    # After 3rd step, snap fires
    layer(x).sum().backward()
    opt.step()
    opt.zero_grad()
    assert _all_weights_on_grid(layer)


# ---------------------------------------------------------------------------
# Unbiased snap
# ---------------------------------------------------------------------------


def test_stochastic_snap_unbiased_over_many_runs() -> None:
    """Averaging stochastic snaps of the same starting weight converges to it.

    We create a weight that sits off-grid, then snap it stochastically
    many times, averaging. Mean must match original within MC noise.
    """
    torch.manual_seed(2)
    layer = MXFP4QATLinear(32, 16, bias=False, block_size=32)
    # Set a known off-grid weight
    original = torch.randn_like(layer.weight) * 0.7
    layer.weight.data.copy_(original)

    base = torch.optim.SGD(layer.parameters(), lr=0.0)  # lr=0 so step() is no-op on weights
    opt = MXFP4StochasticOptimizer(base, layer, snap_interval=1)

    n_trials = 200
    accum = torch.zeros_like(original)
    for _ in range(n_trials):
        # Reset weight to original
        layer.weight.data.copy_(original)
        # Do a dummy step (lr=0 so weight unchanged except for the snap)
        layer.weight.grad = torch.zeros_like(layer.weight)
        opt.step()
        accum += layer.weight.detach()

    mean = accum / n_trials
    # Average of stochastic snaps should approximate the original weight.
    max_err = (mean - original).abs().max().item()
    # Standard error ~ typical block scale / sqrt(n). Typical scale here
    # is ~2^-1, so SE ~ 0.5/sqrt(200) ~ 0.035. Use 0.1 for margin.
    assert max_err < 0.1, f"stochastic snap biased: max|E[snap(w)] - w| = {max_err}"


# ---------------------------------------------------------------------------
# Final snap
# ---------------------------------------------------------------------------


def test_final_snap_is_deterministic() -> None:
    """final_snap() must give the same result every call (no stochastic)."""
    torch.manual_seed(3)
    layer = MXFP4QATLinear(32, 16, bias=False, block_size=32)
    base = torch.optim.SGD(layer.parameters(), lr=0.0)
    opt = MXFP4StochasticOptimizer(base, layer, snap_interval=1)

    # Push the weight off-grid
    layer.weight.data += torch.randn_like(layer.weight) * 0.01

    opt.final_snap()
    snapshot1 = layer.weight.detach().clone()
    opt.final_snap()
    snapshot2 = layer.weight.detach().clone()
    assert torch.equal(snapshot1, snapshot2), "final_snap must be deterministic"


def test_final_snap_reports_layer_count() -> None:
    model = nn.Sequential(
        MXFP4QATLinear(16, 8, bias=False),
        nn.ReLU(),
        MXFP4QATLinear(8, 4, bias=False),
    )
    base = torch.optim.SGD(model.parameters(), lr=1e-3)
    opt = MXFP4StochasticOptimizer(base, model, snap_interval=1)
    n = opt.final_snap()
    assert n == 2


def test_final_snap_output_on_grid() -> None:
    torch.manual_seed(4)
    layer = MXFP4QATLinear(32, 16, bias=False, block_size=32)
    # Inject random perturbation
    layer.weight.data += torch.randn_like(layer.weight) * 0.05
    base = torch.optim.SGD(layer.parameters(), lr=0.0)
    opt = MXFP4StochasticOptimizer(base, layer)
    opt.final_snap()
    assert _all_weights_on_grid(layer)


# ---------------------------------------------------------------------------
# state_dict / load_state_dict
# ---------------------------------------------------------------------------


def test_state_dict_round_trip() -> None:
    torch.manual_seed(5)
    layer = MXFP4QATLinear(8, 4, bias=False)
    base = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    opt = MXFP4StochasticOptimizer(base, layer, snap_interval=2)

    # Take a step so optimizer state exists
    layer(torch.randn(2, 8)).sum().backward()
    opt.step()

    sd = opt.state_dict()
    assert "optimizer" in sd
    assert sd["step_count"] == 1
    assert sd["snap_interval"] == 2

    # Build a fresh wrapper and load
    layer2 = MXFP4QATLinear(8, 4, bias=False)
    base2 = torch.optim.AdamW(layer2.parameters(), lr=1e-3)
    opt2 = MXFP4StochasticOptimizer(base2, layer2, snap_interval=1)
    opt2.load_state_dict(sd)
    assert opt2.step_count == 1
    assert opt2.snap_interval == 2


# ---------------------------------------------------------------------------
# Integration: training still converges
# ---------------------------------------------------------------------------


def test_training_with_wrapper_reduces_loss() -> None:
    """A full training loop with the wrapper must still drive loss down.

    This is the canary: if stochastic snap disrupts gradient dynamics
    more than theory predicts, losses will plateau. Comparable
    threshold to the phase-2 test (40% drop over 80 Adam steps).
    """
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 8),
    )
    apply_mxfp4_qat(model)
    base = torch.optim.Adam(model.parameters(), lr=5e-3)
    opt = MXFP4StochasticOptimizer(base, model, snap_interval=1)

    x = torch.randn(64, 16)
    target = torch.randn(64, 8)

    losses = []
    for _ in range(80):
        opt.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.6, (
        f"training with stochastic snap failed: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    # Final weights should be on grid without calling final_snap (every step snapped).
    for m in model.modules():
        if isinstance(m, MXFP4QATLinear):
            assert _all_weights_on_grid(m)


def test_training_converges_with_sparser_snap() -> None:
    """snap_interval=5 (amortized snapping) still converges."""
    torch.manual_seed(8)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    apply_mxfp4_qat(model)
    base = torch.optim.Adam(model.parameters(), lr=5e-3)
    opt = MXFP4StochasticOptimizer(base, model, snap_interval=5)

    x = torch.randn(64, 16)
    target = torch.randn(64, 8)

    losses = []
    for _ in range(80):
        opt.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.6


# ---------------------------------------------------------------------------
# GPU smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA/HIP")
def test_wrapper_on_gpu() -> None:
    layer = MXFP4QATLinear(32, 16, bias=False).cuda().to(torch.bfloat16)
    base = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    opt = MXFP4StochasticOptimizer(base, layer, snap_interval=1)
    x = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
    layer(x).sum().backward()
    opt.step()
    opt.zero_grad()
    assert _all_weights_on_grid(layer)
