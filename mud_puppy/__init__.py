"""mud-puppy: A ROCm-first LLM fine-tuning framework."""

__all__ = [
    "TrainingConfig",
    "run_training",
]

from .config import TrainingConfig
from .trainer import run_training

