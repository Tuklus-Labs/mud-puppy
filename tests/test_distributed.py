"""Tests for FSDP/distributed wiring in TrainingConfig and trainer.

These tests do NOT spawn a real distributed group. They verify that:
- TrainingConfig validates FSDP options correctly
- FSDP implies distributed
- create_training_args() produces the expected fsdp / fsdp_config payload
- setup_distributed auto-promotes when WORLD_SIZE>1 is in the env

No GPU required.
"""
from __future__ import annotations

import os
import pytest

from mud_puppy.config import TrainingConfig


def _baseline(tmp_path) -> dict:
    ds = tmp_path / "data.jsonl"
    ds.write_text('{"text": "hello"}\n')
    return dict(
        model_name_or_path="unused",
        dataset_path=str(ds),
        output_dir=str(tmp_path / "out"),
        finetuning_method="full",
    )


@pytest.fixture(autouse=True)
def _skip_gpu_count_check(monkeypatch):
    """Bypass the 'need >=2 GPUs' check -- we're testing config wiring."""
    monkeypatch.setenv("MUD_PUPPY_SKIP_GPU_COUNT_CHECK", "1")
    yield


def test_fsdp_implies_distributed(tmp_path):
    """Setting --fsdp must auto-enable --distributed."""
    cfg = TrainingConfig(**_baseline(tmp_path), fsdp_mode="full_shard")
    assert cfg.distributed is True
    assert cfg.fsdp_mode == "full_shard"


def test_fsdp_full_shard_and_offload(tmp_path):
    cfg = TrainingConfig(
        **_baseline(tmp_path),
        fsdp_mode="full_shard",
        fsdp_cpu_offload=True,
        fsdp_transformer_layer_cls="LlamaDecoderLayer",
    )
    assert cfg.fsdp_cpu_offload is True
    assert cfg.fsdp_transformer_layer_cls == "LlamaDecoderLayer"


def test_invalid_fsdp_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="fsdp_mode"):
        TrainingConfig(**_baseline(tmp_path), fsdp_mode="bogus")


def test_invalid_distributed_backend_rejected(tmp_path):
    with pytest.raises(ValueError, match="distributed_backend"):
        TrainingConfig(**_baseline(tmp_path), distributed_backend="magic")


def test_all_four_fsdp_modes_accepted(tmp_path):
    for mode in ("full_shard", "shard_grad_op", "no_shard", "hybrid_shard"):
        cfg = TrainingConfig(**_baseline(tmp_path), fsdp_mode=mode)
        assert cfg.fsdp_mode == mode


def test_non_fsdp_distributed_still_works(tmp_path):
    """Plain DDP (distributed=True, fsdp_mode='') remains valid."""
    cfg = TrainingConfig(**_baseline(tmp_path), distributed=True)
    assert cfg.distributed is True
    assert cfg.fsdp_mode == ""


def test_default_fsdp_is_off(tmp_path):
    """Brand-new config must NOT enable any distributed behavior by default."""
    cfg = TrainingConfig(**_baseline(tmp_path))
    assert cfg.distributed is False
    assert cfg.fsdp_mode == ""
    assert cfg.distributed_backend == "nccl"


def test_create_training_args_emits_fsdp_payload(tmp_path, monkeypatch):
    """HF TrainingArguments must receive the right fsdp / fsdp_config."""
    # The trainer imports torch heavily; skip if not available in CI.
    torch = pytest.importorskip("torch")
    from mud_puppy.trainer import create_training_args

    cfg = TrainingConfig(
        **_baseline(tmp_path),
        fsdp_mode="full_shard",
        fsdp_transformer_layer_cls="LlamaDecoderLayer",
        fsdp_activation_checkpointing=True,
    )
    ta = create_training_args(cfg)
    # HF stores fsdp as a list of tokens after parse.
    fsdp_attr = getattr(ta, "fsdp", None)
    assert fsdp_attr, "fsdp not set on TrainingArguments"
    # Either list or space-separated string depending on HF version
    fsdp_str = " ".join(fsdp_attr) if isinstance(fsdp_attr, list) else str(fsdp_attr)
    assert "full_shard" in fsdp_str
    assert "auto_wrap" in fsdp_str

    fsdp_config = getattr(ta, "fsdp_config", None) or {}
    # HF may normalize the key name; check both common forms.
    wrap_cls = fsdp_config.get("transformer_layer_cls_to_wrap") or \
        fsdp_config.get("fsdp_transformer_layer_cls_to_wrap")
    assert wrap_cls == ["LlamaDecoderLayer"]
    assert fsdp_config.get("activation_checkpointing") is True or \
        fsdp_config.get("fsdp_activation_checkpointing") is True


def test_create_training_args_no_fsdp_when_unset(tmp_path):
    """Non-FSDP configs must NOT emit fsdp kwarg (avoids HF 'fsdp=' noise)."""
    pytest.importorskip("torch")
    from mud_puppy.trainer import create_training_args

    cfg = TrainingConfig(**_baseline(tmp_path))
    ta = create_training_args(cfg)
    # HF's default for fsdp is [] or "", never truthy unless we set it.
    fsdp_attr = getattr(ta, "fsdp", None)
    if isinstance(fsdp_attr, list):
        assert fsdp_attr == []
    else:
        assert not fsdp_attr


def test_setup_distributed_auto_promotes_on_torchrun(tmp_path, monkeypatch):
    """If torchrun set WORLD_SIZE>1, setup_distributed must enable distributed.

    We don't actually init the process group here; we just verify the
    auto-promotion logic flips the config flag. The real init would
    require a running rendezvous and multiple processes.
    """
    pytest.importorskip("torch")
    import torch.distributed as dist
    from mud_puppy import trainer

    cfg = TrainingConfig(**_baseline(tmp_path))
    assert cfg.distributed is False

    # Pretend torchrun launched us.
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")

    # Stub init_process_group so we don't actually connect.
    called = {}
    def fake_init(backend, **kw):
        called["backend"] = backend
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "init_process_group", fake_init)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    # setup_distributed flips config.distributed based on WORLD_SIZE and
    # exits early because is_initialized() returns True.
    ok = trainer.setup_distributed(cfg)
    assert ok is True
    assert cfg.distributed is True
