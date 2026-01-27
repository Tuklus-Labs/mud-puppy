"""LoRA Library - TPU-powered adapter training and management system.

This package provides tools for training, managing, and serving a library of
specialized LoRA adapters that can be hot-swapped at runtime.

Example usage:
    from lora_library import AdapterRegistry, BatchTrainer, AdapterServer

    # Create registry
    registry = AdapterRegistry("./adapters")

    # Train adapters
    trainer = BatchTrainer(registry, config)
    trainer.train_all()

    # Serve adapters
    server = AdapterServer(registry)
    server.load_adapter("coding")
    output = server.generate("def fibonacci(n):", adapter="coding")
"""

from .config import (
    LoraLibraryConfig,
    AdapterConfig,
    TrainingHyperparameters,
    DatasetConfig,
    TPUMeshConfig,
)
from .registry import (
    AdapterRegistry,
    AdapterMetadata,
    AdapterVersion,
)
from .datasets import (
    TaskDatasetLoader,
    CodeDataset,
    MathDataset,
    InstructionDataset,
    TranslationDataset,
    SummarizationDataset,
)
from .batch_trainer import (
    BatchTrainer,
    TrainingProgress,
    TrainingResult,
)
from .evaluator import (
    AdapterEvaluator,
    EvaluationResult,
    EvaluationMetrics,
)
from .merger import (
    AdapterMerger,
    MergeStrategy,
)
from .server import (
    AdapterServer,
    AdapterCache,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "LoraLibraryConfig",
    "AdapterConfig",
    "TrainingHyperparameters",
    "DatasetConfig",
    "TPUMeshConfig",
    # Registry
    "AdapterRegistry",
    "AdapterMetadata",
    "AdapterVersion",
    # Datasets
    "TaskDatasetLoader",
    "CodeDataset",
    "MathDataset",
    "InstructionDataset",
    "TranslationDataset",
    "SummarizationDataset",
    # Training
    "BatchTrainer",
    "TrainingProgress",
    "TrainingResult",
    # Evaluation
    "AdapterEvaluator",
    "EvaluationResult",
    "EvaluationMetrics",
    # Merging
    "AdapterMerger",
    "MergeStrategy",
    # Serving
    "AdapterServer",
    "AdapterCache",
]
