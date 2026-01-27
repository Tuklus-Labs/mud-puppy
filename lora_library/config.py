"""Configuration classes for the LoRA library system.

This module defines all configuration dataclasses for managing a library
of specialized LoRA adapters, including training hyperparameters, dataset
specifications, and TPU mesh configurations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class TPUMeshConfig:
    """Configuration for TPU/GPU device mesh.

    Attributes:
        mesh_shape: Shape of the device mesh (data_parallel, model_parallel).
        axis_names: Names for the mesh axes.
        use_pjit: Whether to use pjit for distributed computation.
        fsdp_enabled: Enable Fully Sharded Data Parallel.
    """
    mesh_shape: Tuple[int, ...] = (1, 1)
    axis_names: Tuple[str, ...] = ("data", "model")
    use_pjit: bool = False
    fsdp_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mesh_shape": list(self.mesh_shape),
            "axis_names": list(self.axis_names),
            "use_pjit": self.use_pjit,
            "fsdp_enabled": self.fsdp_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TPUMeshConfig":
        return cls(
            mesh_shape=tuple(data.get("mesh_shape", (1, 1))),
            axis_names=tuple(data.get("axis_names", ("data", "model"))),
            use_pjit=data.get("use_pjit", False),
            fsdp_enabled=data.get("fsdp_enabled", False),
        )


@dataclass
class TrainingHyperparameters:
    """Training hyperparameters for a LoRA adapter.

    Attributes:
        learning_rate: Initial learning rate.
        batch_size: Per-device batch size.
        gradient_accumulation: Number of gradient accumulation steps.
        num_epochs: Number of training epochs.
        max_steps: Maximum training steps (overrides epochs if set).
        warmup_ratio: Fraction of steps for learning rate warmup.
        weight_decay: AdamW weight decay.
        max_grad_norm: Maximum gradient norm for clipping.
        lr_scheduler: Learning rate scheduler type.
        precision: Training precision (bf16, fp16, fp32).
        seed: Random seed for reproducibility.
    """
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation: int = 4
    num_epochs: int = 3
    max_steps: Optional[int] = None
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    precision: str = "bf16"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler": self.lr_scheduler,
            "precision": self.precision,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingHyperparameters":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetConfig:
    """Configuration for a training dataset.

    Attributes:
        path: Path to the dataset (local file or HuggingFace dataset).
        format: Dataset format (jsonl, parquet, csv, hf).
        split: Dataset split to use.
        text_column: Column name for text data.
        max_length: Maximum sequence length.
        preprocessing_workers: Number of preprocessing workers.
        streaming: Whether to stream the dataset.
    """
    path: str
    format: str = "jsonl"
    split: str = "train"
    text_column: str = "text"
    max_length: int = 2048
    preprocessing_workers: int = 4
    streaming: bool = False

    # Optional columns for specific formats
    input_column: Optional[str] = None
    output_column: Optional[str] = None
    messages_column: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "split": self.split,
            "text_column": self.text_column,
            "max_length": self.max_length,
            "preprocessing_workers": self.preprocessing_workers,
            "streaming": self.streaming,
            "input_column": self.input_column,
            "output_column": self.output_column,
            "messages_column": self.messages_column,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AdapterConfig:
    """Configuration for a single LoRA adapter.

    Attributes:
        name: Unique identifier for the adapter.
        task: Task type (coding, math, creative, translation, etc.).
        description: Human-readable description.
        r: LoRA rank.
        alpha: LoRA alpha scaling factor.
        dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to.
        use_rslora: Use rank-stabilized LoRA scaling.
        bias: Bias training strategy ("none", "all", "lora_only").
        hyperparameters: Training hyperparameters.
        dataset: Dataset configuration.
        enabled: Whether this adapter is enabled for training.
    """
    name: str
    task: str
    description: str = ""

    # LoRA parameters
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_rslora: bool = False
    bias: str = "none"

    # Training config
    hyperparameters: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    dataset: Optional[DatasetConfig] = None

    # State
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "task": self.task,
            "description": self.description,
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "use_rslora": self.use_rslora,
            "bias": self.bias,
            "hyperparameters": self.hyperparameters.to_dict(),
            "enabled": self.enabled,
        }
        if self.dataset:
            result["dataset"] = self.dataset.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdapterConfig":
        hyperparameters = data.pop("hyperparameters", {})
        dataset_data = data.pop("dataset", None)

        config = cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        )

        if hyperparameters:
            config.hyperparameters = TrainingHyperparameters.from_dict(hyperparameters)
        if dataset_data:
            config.dataset = DatasetConfig.from_dict(dataset_data)

        return config

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AdapterConfig":
        """Load adapter config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save adapter config to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class LoraLibraryConfig:
    """Top-level configuration for the LoRA library.

    Attributes:
        base_model_path: Path to the base model (HuggingFace model or local path).
        output_dir: Root directory for adapter outputs.
        adapters: Dictionary mapping adapter names to their configurations.
        tpu_mesh: TPU/GPU mesh configuration.
        default_hyperparameters: Default training hyperparameters for all adapters.
        checkpoint_interval: Steps between checkpoints.
        eval_interval: Steps between evaluations.
        log_interval: Steps between log entries.
        save_total_limit: Maximum number of checkpoints to keep per adapter.
        resume_from_checkpoint: Whether to resume training from checkpoints.
        trust_remote_code: Allow loading models with custom code.
    """
    base_model_path: str
    output_dir: str
    adapters: Dict[str, AdapterConfig] = field(default_factory=dict)
    tpu_mesh: TPUMeshConfig = field(default_factory=TPUMeshConfig)
    default_hyperparameters: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)

    # Training control
    checkpoint_interval: int = 500
    eval_interval: int = 500
    log_interval: int = 10
    save_total_limit: int = 3
    resume_from_checkpoint: bool = True
    trust_remote_code: bool = False

    # Resource management
    memory_cleanup_between_adapters: bool = True
    max_parallel_adapters: int = 1

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.base_model_path:
            raise ValueError("base_model_path must be specified")

        # Ensure output directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "adapters"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)

    def add_adapter(self, config: AdapterConfig) -> None:
        """Add an adapter configuration to the library."""
        self.adapters[config.name] = config

    def remove_adapter(self, name: str) -> bool:
        """Remove an adapter configuration from the library."""
        if name in self.adapters:
            del self.adapters[name]
            return True
        return False

    def get_adapter(self, name: str) -> Optional[AdapterConfig]:
        """Get an adapter configuration by name."""
        return self.adapters.get(name)

    def get_enabled_adapters(self) -> List[AdapterConfig]:
        """Get all enabled adapter configurations."""
        return [a for a in self.adapters.values() if a.enabled]

    def get_adapter_output_dir(self, adapter_name: str) -> str:
        """Get the output directory for a specific adapter."""
        return os.path.join(self.output_dir, "adapters", adapter_name)

    def get_checkpoint_dir(self, adapter_name: str) -> str:
        """Get the checkpoint directory for a specific adapter."""
        return os.path.join(self.output_dir, "checkpoints", adapter_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model_path": self.base_model_path,
            "output_dir": self.output_dir,
            "adapters": {name: cfg.to_dict() for name, cfg in self.adapters.items()},
            "tpu_mesh": self.tpu_mesh.to_dict(),
            "default_hyperparameters": self.default_hyperparameters.to_dict(),
            "checkpoint_interval": self.checkpoint_interval,
            "eval_interval": self.eval_interval,
            "log_interval": self.log_interval,
            "save_total_limit": self.save_total_limit,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "trust_remote_code": self.trust_remote_code,
            "memory_cleanup_between_adapters": self.memory_cleanup_between_adapters,
            "max_parallel_adapters": self.max_parallel_adapters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoraLibraryConfig":
        adapters_data = data.pop("adapters", {})
        tpu_mesh_data = data.pop("tpu_mesh", {})
        default_hp_data = data.pop("default_hyperparameters", {})

        config = cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        )

        config.tpu_mesh = TPUMeshConfig.from_dict(tpu_mesh_data)
        config.default_hyperparameters = TrainingHyperparameters.from_dict(default_hp_data)

        for name, adapter_data in adapters_data.items():
            adapter_data["name"] = name
            config.adapters[name] = AdapterConfig.from_dict(adapter_data)

        return config

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "LoraLibraryConfig":
        """Load library config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save library config to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def create_default(
        cls,
        base_model_path: str,
        output_dir: str,
        adapter_configs_dir: Optional[str] = None,
    ) -> "LoraLibraryConfig":
        """Create a default configuration, optionally loading adapter configs from a directory.

        Args:
            base_model_path: Path to the base model.
            output_dir: Output directory for the library.
            adapter_configs_dir: Optional directory containing adapter YAML files.

        Returns:
            LoraLibraryConfig with loaded adapters.
        """
        config = cls(
            base_model_path=base_model_path,
            output_dir=output_dir,
        )

        if adapter_configs_dir and os.path.isdir(adapter_configs_dir):
            for filename in os.listdir(adapter_configs_dir):
                if filename.endswith((".yaml", ".yml")):
                    filepath = os.path.join(adapter_configs_dir, filename)
                    try:
                        adapter = AdapterConfig.from_yaml(filepath)
                        config.add_adapter(adapter)
                    except Exception as e:
                        print(f"Warning: Failed to load adapter config {filename}: {e}")

        return config
