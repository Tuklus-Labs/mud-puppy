"""Tests for mud_puppy.zero_offload.CPUOffloadOptimizer."""
import pytest
import torch
import torch.nn as nn

from mud_puppy.zero_offload import CPUOffloadOptimizer, OffloadConfig


def _make_two_group_optimizer():
    """Build a toy model and an AdamW with two distinct param groups."""
    m = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 4))
    # Group 0: first linear at lr=1e-3, wd=0.1
    # Group 1: second linear at lr=5e-4, wd=0.0
    g0 = {"params": list(m[0].parameters()), "lr": 1e-3, "weight_decay": 0.1}
    g1 = {"params": list(m[1].parameters()), "lr": 5e-4, "weight_decay": 0.0}
    opt = torch.optim.AdamW([g0, g1], eps=1e-8)
    return m, opt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_foreach_rebuild_preserves_per_group_hyperparameters():
    """The one-shot foreach=True rebuild must preserve per-group lr/wd."""
    model, opt = _make_two_group_optimizer()
    model.cuda()

    cfg = OffloadConfig(
        offload_optimizer=True,
        offload_gradients=True,
        pin_memory=True,
        async_transfer=False,  # deterministic path for the test
        cpu_optimizer_step=True,
    )
    wrapped = CPUOffloadOptimizer(opt, cfg)

    # Force the foreach-rebuild path by running one step.
    x = torch.randn(4, 16, device="cuda")
    out = model(x).sum()
    out.backward()
    wrapped.step()

    # After the rebuild, there must still be two groups with their
    # original hyperparameters intact.
    groups = wrapped.optimizer.param_groups
    assert len(groups) == 2, f"expected 2 groups, got {len(groups)}"
    assert groups[0]["lr"] == pytest.approx(1e-3)
    assert groups[0]["weight_decay"] == pytest.approx(0.1)
    assert groups[1]["lr"] == pytest.approx(5e-4)
    assert groups[1]["weight_decay"] == pytest.approx(0.0)

    # foreach mode should have been selected for every group.
    for g in groups:
        assert g.get("foreach") is True, (
            f"foreach not enabled on group with lr={g.get('lr')}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_offload_disabled_grads_reads_direct():
    """With offload_gradients=False, CPU optimizer must read live GPU grads.

    Regression: previously the CPU step still routed through
    self._cpu_grads (a zero-initialised accumulator that is never filled
    when offload_gradients is off), so the optimizer trained on zeros or
    crashed on device mismatch.
    """
    model = nn.Linear(8, 8).cuda()
    w0 = model.weight.detach().clone()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    cfg = OffloadConfig(
        offload_optimizer=True,
        offload_gradients=False,  # critical switch
        pin_memory=True,
        async_transfer=False,
        cpu_optimizer_step=True,
    )
    wrapped = CPUOffloadOptimizer(opt, cfg)

    x = torch.randn(4, 8, device="cuda")
    out = (model(x) ** 2).sum()
    out.backward()
    assert model.weight.grad is not None
    assert model.weight.grad.abs().sum().item() > 0

    wrapped.step()

    # Parameter must have moved. If the bug returns, the optimizer reads
    # zero CPU grads and the weight stays exactly equal to w0.
    delta = (model.weight.detach() - w0.to(model.weight.device)).abs().sum().item()
    assert delta > 0, "parameters did not update; optimizer ran on zero grads"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_state_dict_roundtrip():
    """state_dict() keyed by positional index must survive a load_state_dict().

    Verifies that:
    1. state_dict() produces a cpu_states dict keyed by int (not by id(param)).
    2. load_state_dict() correctly translates back so training continues
       from the restored state without crashing or resetting momentum.
    """
    model = nn.Linear(8, 8).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    cfg = OffloadConfig(
        offload_optimizer=True,
        offload_gradients=True,
        pin_memory=True,
        async_transfer=False,
        cpu_optimizer_step=True,
    )
    wrapped = CPUOffloadOptimizer(opt, cfg)

    # Run two steps so optimizer state (exp_avg, exp_avg_sq) is populated.
    for _ in range(2):
        model.zero_grad()
        out = (model(torch.randn(4, 8, device="cuda")) ** 2).sum()
        out.backward()
        wrapped.step()

    sd = wrapped.state_dict()

    # cpu_states must exist and be keyed by positional int index (not id).
    assert "cpu_states" in sd, "state_dict missing cpu_states key"
    cpu_states = sd["cpu_states"]
    for k in cpu_states:
        assert isinstance(k, int), (
            f"cpu_states key {k!r} is not an int -- id(param) keying regressed"
        )

    # Reconstruct a fresh optimizer from the same model and restore state.
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-2)
    wrapped2 = CPUOffloadOptimizer(opt2, cfg)
    wrapped2.load_state_dict(sd)

    # After restoring, a third step must complete without error and must move
    # parameters (optimizer state is live, not zeroed).
    w_before = model.weight.detach().clone()
    model.zero_grad()
    out = (model(torch.randn(4, 8, device="cuda")) ** 2).sum()
    out.backward()
    wrapped2.step()
    delta = (model.weight.detach() - w_before).abs().sum().item()
    assert delta > 0, "parameters did not move after load_state_dict restore"
