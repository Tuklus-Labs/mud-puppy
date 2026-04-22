"""Validation of TrainingConfig.__post_init__ numeric sanity checks."""
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
