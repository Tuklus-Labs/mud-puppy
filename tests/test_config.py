"""Validation of TrainingConfig.__post_init__ numeric sanity checks."""
import importlib
import math

import pytest
from mud_puppy.config import TrainingConfig


def _baseline(tmp_path) -> dict:
    """Minimal valid TrainingConfig kwargs. dataset_path must exist on disk."""
    ds = tmp_path / "data.jsonl"
    ds.write_text('{"text": "hello"}\n')
    return dict(
        model_name_or_path="unused-model",
        dataset_path=str(ds),
        output_dir=str(tmp_path / "out"),
        finetuning_method="lora",
    )


def test_config_valid_baseline(tmp_path):
    """Baseline config with sane defaults must not raise."""
    cfg = TrainingConfig(**_baseline(tmp_path))
    assert cfg.batch_size == 1
    assert cfg.num_epochs == 1
    assert cfg.learning_rate > 0


def test_config_rejects_zero_batch_size(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(**_baseline(tmp_path), batch_size=0)


def test_config_rejects_negative_batch_size(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(**_baseline(tmp_path), batch_size=-1)


def test_config_rejects_zero_epochs(tmp_path):
    with pytest.raises(ValueError, match="num_epochs"):
        TrainingConfig(**_baseline(tmp_path), num_epochs=0)


def test_config_rejects_zero_learning_rate(tmp_path):
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(**_baseline(tmp_path), learning_rate=0.0)


@pytest.mark.parametrize("method", ["lora", "qlora"])
def test_config_rejects_zero_lora_r_for_lora(tmp_path, method):
    with pytest.raises(ValueError, match="lora_r"):
        TrainingConfig(**{**_baseline(tmp_path), "finetuning_method": method}, lora_r=0)


def test_learning_rate_rejects_nan(tmp_path):
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(**_baseline(tmp_path), learning_rate=float("nan"))


def test_learning_rate_rejects_inf(tmp_path):
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(**_baseline(tmp_path), learning_rate=float("inf"))


def test_lora_dropout_rejects_nan(tmp_path):
    with pytest.raises(ValueError, match="lora_dropout"):
        TrainingConfig(**_baseline(tmp_path), lora_dropout=float("nan"))


def test_max_grad_norm_rejects_zero(tmp_path):
    with pytest.raises(ValueError, match="max_grad_norm"):
        TrainingConfig(**_baseline(tmp_path), max_grad_norm=0.0)


def test_max_grad_norm_rejects_negative(tmp_path):
    with pytest.raises(ValueError, match="max_grad_norm"):
        TrainingConfig(**_baseline(tmp_path), max_grad_norm=-0.5)


def test_empty_local_rank_env_does_not_crash_import(monkeypatch):
    """LOCAL_RANK='' in env must not crash at dataclass class-body evaluation."""
    monkeypatch.setenv("LOCAL_RANK", "")
    import mud_puppy.config as cfg_mod
    importlib.reload(cfg_mod)
    # Baseline: local_rank should default to 0 under empty env.
    assert cfg_mod._env_int("LOCAL_RANK", 0) == 0


def test_cli_parser_handles_empty_local_rank(monkeypatch):
    """build_parser() must not crash when LOCAL_RANK is set to an empty string.

    Regression: cli.py used ``int(os.environ.get("LOCAL_RANK", 0))`` which
    raises ValueError on empty strings. ``_env_int`` is the correct helper.
    """
    monkeypatch.setenv("LOCAL_RANK", "")
    import mud_puppy.config as cfg_mod
    import mud_puppy.cli as cli_mod
    importlib.reload(cfg_mod)
    importlib.reload(cli_mod)
    parser = cli_mod.build_parser()
    # Parse a minimal positional arg set to exercise the default.
    args = parser.parse_args(["some_model", "some_dataset.jsonl"])
    assert args.local_rank == 0


def test_cli_parser_reads_valid_local_rank(monkeypatch):
    """build_parser() should pick up a non-empty integer LOCAL_RANK from env."""
    monkeypatch.setenv("LOCAL_RANK", "3")
    import mud_puppy.config as cfg_mod
    import mud_puppy.cli as cli_mod
    importlib.reload(cfg_mod)
    importlib.reload(cli_mod)
    parser = cli_mod.build_parser()
    args = parser.parse_args(["some_model", "some_dataset.jsonl"])
    assert args.local_rank == 3
