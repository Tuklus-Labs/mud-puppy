"""Preference-based fine tuning algorithms."""

from .config import TrainingConfig


SUPPORTED_PREFERENCES = {"dpo", "ipo", "kto", "orpo"}


def run_preference_training(config: TrainingConfig):
    if config.preference not in SUPPORTED_PREFERENCES:
        raise ValueError(f"Unsupported preference method: {config.preference}")
    raise NotImplementedError("Preference tuning is not implemented yet")

