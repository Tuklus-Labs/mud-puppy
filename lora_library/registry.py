"""Adapter registry for managing LoRA adapter metadata and versions.

This module provides the AdapterRegistry class for tracking, versioning,
and managing a library of trained LoRA adapters.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import jax.numpy as jnp
    from flax.core import freeze, unfreeze
    from flax.traverse_util import flatten_dict, unflatten_dict
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@dataclass
class AdapterVersion:
    """Version information for an adapter.

    Attributes:
        version: Semantic version string.
        created_at: Timestamp when version was created.
        checkpoint_path: Path to the checkpoint files.
        training_steps: Number of training steps completed.
        final_loss: Final training loss.
        eval_metrics: Evaluation metrics for this version.
        commit_hash: Hash of the adapter weights for integrity checking.
        notes: Optional notes about this version.
    """
    version: str
    created_at: str
    checkpoint_path: str
    training_steps: int = 0
    final_loss: float = 0.0
    eval_metrics: Dict[str, float] = field(default_factory=dict)
    commit_hash: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "checkpoint_path": self.checkpoint_path,
            "training_steps": self.training_steps,
            "final_loss": self.final_loss,
            "eval_metrics": self.eval_metrics,
            "commit_hash": self.commit_hash,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdapterVersion":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AdapterMetadata:
    """Metadata for a registered adapter.

    Attributes:
        name: Unique identifier for the adapter.
        task: Task type (coding, math, creative, etc.).
        base_model: Name/path of the base model this adapter was trained for.
        description: Human-readable description of the adapter.
        created_at: Timestamp when adapter was first created.
        updated_at: Timestamp of last update.
        versions: List of adapter versions.
        current_version: Currently active version.
        lora_config: LoRA configuration used for training.
        performance_metrics: Best performance metrics across versions.
        tags: Tags for categorization and search.
    """
    name: str
    task: str
    base_model: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    versions: List[AdapterVersion] = field(default_factory=list)
    current_version: str = ""
    lora_config: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def add_version(self, version: AdapterVersion, set_current: bool = True) -> None:
        """Add a new version to this adapter."""
        self.versions.append(version)
        self.updated_at = datetime.now().isoformat()
        if set_current:
            self.current_version = version.version

        # Update best performance metrics
        for metric, value in version.eval_metrics.items():
            if metric not in self.performance_metrics:
                self.performance_metrics[metric] = value
            elif "loss" in metric.lower() or "error" in metric.lower():
                # Lower is better
                self.performance_metrics[metric] = min(self.performance_metrics[metric], value)
            else:
                # Higher is better (accuracy, score, etc.)
                self.performance_metrics[metric] = max(self.performance_metrics[metric], value)

    def get_version(self, version_str: str) -> Optional[AdapterVersion]:
        """Get a specific version."""
        for v in self.versions:
            if v.version == version_str:
                return v
        return None

    def get_current_version(self) -> Optional[AdapterVersion]:
        """Get the current version."""
        return self.get_version(self.current_version)

    def get_latest_version(self) -> Optional[AdapterVersion]:
        """Get the most recent version."""
        if self.versions:
            return self.versions[-1]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "base_model": self.base_model,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "versions": [v.to_dict() for v in self.versions],
            "current_version": self.current_version,
            "lora_config": self.lora_config,
            "performance_metrics": self.performance_metrics,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdapterMetadata":
        versions_data = data.pop("versions", [])
        meta = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        meta.versions = [AdapterVersion.from_dict(v) for v in versions_data]
        return meta


class AdapterRegistry:
    """Registry for managing LoRA adapters.

    Provides functionality to:
    - Register, list, and retrieve adapters
    - Track versions and metadata
    - Save/load adapter weights and configs
    - Search and filter adapters by task, tags, or metrics
    """

    def __init__(self, registry_path: Union[str, Path]):
        """Initialize the adapter registry.

        Args:
            registry_path: Root directory for the registry.
        """
        self.registry_path = Path(registry_path)
        self.adapters_dir = self.registry_path / "adapters"
        self.metadata_file = self.registry_path / "registry.json"

        # Create directory structure
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.adapters_dir.mkdir(exist_ok=True)

        # Load or initialize metadata
        self._adapters: Dict[str, AdapterMetadata] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load registry metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                self._adapters = {
                    name: AdapterMetadata.from_dict(meta)
                    for name, meta in data.get("adapters", {}).items()
                }
            except Exception as e:
                print(f"Warning: Failed to load registry metadata: {e}")
                self._adapters = {}

    def _save_metadata(self) -> None:
        """Save registry metadata to disk."""
        data = {
            "adapters": {name: meta.to_dict() for name, meta in self._adapters.items()},
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        name: str,
        task: str,
        base_model: str,
        description: str = "",
        lora_config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> AdapterMetadata:
        """Register a new adapter.

        Args:
            name: Unique identifier for the adapter.
            task: Task type.
            base_model: Base model name/path.
            description: Human-readable description.
            lora_config: LoRA configuration dictionary.
            tags: Tags for categorization.

        Returns:
            Created AdapterMetadata.

        Raises:
            ValueError: If an adapter with this name already exists.
        """
        if name in self._adapters:
            raise ValueError(f"Adapter '{name}' already exists")

        # Create adapter directory
        adapter_dir = self.adapters_dir / name
        adapter_dir.mkdir(exist_ok=True)

        metadata = AdapterMetadata(
            name=name,
            task=task,
            base_model=base_model,
            description=description,
            lora_config=lora_config or {},
            tags=tags or [],
        )

        self._adapters[name] = metadata
        self._save_metadata()

        return metadata

    def unregister(self, name: str, delete_files: bool = False) -> bool:
        """Unregister an adapter.

        Args:
            name: Adapter name.
            delete_files: If True, also delete adapter files.

        Returns:
            True if adapter was removed, False if not found.
        """
        if name not in self._adapters:
            return False

        if delete_files:
            adapter_dir = self.adapters_dir / name
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)

        del self._adapters[name]
        self._save_metadata()
        return True

    def get(self, name: str) -> Optional[AdapterMetadata]:
        """Get adapter metadata by name."""
        return self._adapters.get(name)

    def list(
        self,
        task: Optional[str] = None,
        tags: Optional[List[str]] = None,
        base_model: Optional[str] = None,
    ) -> List[AdapterMetadata]:
        """List adapters with optional filtering.

        Args:
            task: Filter by task type.
            tags: Filter by tags (any match).
            base_model: Filter by base model.

        Returns:
            List of matching AdapterMetadata.
        """
        adapters = list(self._adapters.values())

        if task:
            adapters = [a for a in adapters if a.task == task]

        if tags:
            tag_set = set(tags)
            adapters = [a for a in adapters if tag_set & set(a.tags)]

        if base_model:
            adapters = [a for a in adapters if a.base_model == base_model]

        return adapters

    def list_names(self) -> List[str]:
        """List all adapter names."""
        return list(self._adapters.keys())

    def add_version(
        self,
        name: str,
        version: str,
        checkpoint_path: str,
        training_steps: int = 0,
        final_loss: float = 0.0,
        eval_metrics: Optional[Dict[str, float]] = None,
        notes: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> AdapterVersion:
        """Add a new version to an adapter.

        Args:
            name: Adapter name.
            version: Version string.
            checkpoint_path: Path to checkpoint files.
            training_steps: Training steps completed.
            final_loss: Final training loss.
            eval_metrics: Evaluation metrics.
            notes: Version notes.
            params: Optional parameters to compute hash from.

        Returns:
            Created AdapterVersion.

        Raises:
            KeyError: If adapter not found.
        """
        if name not in self._adapters:
            raise KeyError(f"Adapter '{name}' not found")

        # Compute hash if params provided
        commit_hash = ""
        if params is not None:
            commit_hash = self._compute_params_hash(params)

        version_obj = AdapterVersion(
            version=version,
            created_at=datetime.now().isoformat(),
            checkpoint_path=checkpoint_path,
            training_steps=training_steps,
            final_loss=final_loss,
            eval_metrics=eval_metrics or {},
            commit_hash=commit_hash,
            notes=notes,
        )

        self._adapters[name].add_version(version_obj)
        self._save_metadata()

        return version_obj

    def _compute_params_hash(self, params: Dict[str, Any]) -> str:
        """Compute a hash of parameters for integrity checking."""
        if JAX_AVAILABLE:
            flat = flatten_dict(params)
            # Create a hash from the structure and values
            hasher = hashlib.sha256()
            for key, value in sorted(flat.items()):
                key_str = "/".join(str(k) for k in key)
                hasher.update(key_str.encode())
                hasher.update(str(value.shape).encode())
                hasher.update(str(value.dtype).encode())
            return hasher.hexdigest()[:16]
        return ""

    def get_adapter_path(self, name: str, version: Optional[str] = None) -> Path:
        """Get the path to an adapter's files.

        Args:
            name: Adapter name.
            version: Optional version (uses current if not specified).

        Returns:
            Path to the adapter directory.
        """
        adapter_dir = self.adapters_dir / name

        if version:
            return adapter_dir / version

        # Use current version
        meta = self._adapters.get(name)
        if meta and meta.current_version:
            return adapter_dir / meta.current_version

        return adapter_dir

    def save_params(
        self,
        name: str,
        params: Dict[str, Any],
        version: Optional[str] = None,
    ) -> str:
        """Save adapter parameters to disk.

        Args:
            name: Adapter name.
            params: Parameters to save.
            version: Optional version (creates new if not specified).

        Returns:
            Path where parameters were saved.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is required for saving parameters")

        import orbax.checkpoint as ocp

        # Determine version
        meta = self._adapters.get(name)
        if not meta:
            raise KeyError(f"Adapter '{name}' not found")

        if version is None:
            # Create new version
            num_versions = len(meta.versions)
            version = f"v{num_versions + 1}"

        # Create version directory
        version_dir = self.adapters_dir / name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save parameters
        checkpointer = ocp.PyTreeCheckpointer()
        save_path = version_dir / "params"
        checkpointer.save(str(save_path), params)

        return str(version_dir)

    def load_params(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load adapter parameters from disk.

        Args:
            name: Adapter name.
            version: Optional version (uses current if not specified).

        Returns:
            Loaded parameters.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is required for loading parameters")

        import orbax.checkpoint as ocp

        adapter_path = self.get_adapter_path(name, version)
        params_path = adapter_path / "params"

        if not params_path.exists():
            raise FileNotFoundError(f"Parameters not found at {params_path}")

        checkpointer = ocp.PyTreeCheckpointer()
        return checkpointer.restore(str(params_path))

    def update_metadata(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
    ) -> AdapterMetadata:
        """Update adapter metadata.

        Args:
            name: Adapter name.
            description: New description.
            tags: New tags (replaces existing).
            performance_metrics: New performance metrics (merges with existing).

        Returns:
            Updated metadata.
        """
        if name not in self._adapters:
            raise KeyError(f"Adapter '{name}' not found")

        meta = self._adapters[name]

        if description is not None:
            meta.description = description
        if tags is not None:
            meta.tags = tags
        if performance_metrics:
            meta.performance_metrics.update(performance_metrics)

        meta.updated_at = datetime.now().isoformat()
        self._save_metadata()

        return meta

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Tuple[AdapterMetadata, float]]:
        """Search adapters by name, description, or tags.

        Args:
            query: Search query.
            limit: Maximum results to return.

        Returns:
            List of (metadata, score) tuples, sorted by relevance.
        """
        query_lower = query.lower()
        results = []

        for meta in self._adapters.values():
            score = 0.0

            # Name match (highest weight)
            if query_lower in meta.name.lower():
                score += 3.0
            if meta.name.lower() == query_lower:
                score += 2.0

            # Task match
            if query_lower in meta.task.lower():
                score += 2.0

            # Description match
            if query_lower in meta.description.lower():
                score += 1.0

            # Tag match
            for tag in meta.tags:
                if query_lower in tag.lower():
                    score += 1.5

            if score > 0:
                results.append((meta, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get_by_task(self, task: str) -> List[AdapterMetadata]:
        """Get all adapters for a specific task."""
        return self.list(task=task)

    def get_best_adapter(
        self,
        task: str,
        metric: str = "loss",
        lower_is_better: bool = True,
    ) -> Optional[AdapterMetadata]:
        """Get the best adapter for a task based on a metric.

        Args:
            task: Task type.
            metric: Metric name to compare.
            lower_is_better: Whether lower metric values are better.

        Returns:
            Best adapter metadata, or None if no adapters found.
        """
        adapters = self.get_by_task(task)

        if not adapters:
            return None

        # Filter to adapters with the metric
        adapters_with_metric = [
            a for a in adapters
            if metric in a.performance_metrics
        ]

        if not adapters_with_metric:
            return adapters[0]  # Return first if no metrics

        # Sort by metric
        return sorted(
            adapters_with_metric,
            key=lambda a: a.performance_metrics[metric],
            reverse=not lower_is_better,
        )[0]

    def export_catalog(self, output_path: Union[str, Path]) -> None:
        """Export a catalog of all adapters to a JSON file.

        Args:
            output_path: Path for the output file.
        """
        catalog = {
            "generated_at": datetime.now().isoformat(),
            "num_adapters": len(self._adapters),
            "adapters": [
                {
                    "name": meta.name,
                    "task": meta.task,
                    "description": meta.description,
                    "base_model": meta.base_model,
                    "num_versions": len(meta.versions),
                    "current_version": meta.current_version,
                    "performance_metrics": meta.performance_metrics,
                    "tags": meta.tags,
                }
                for meta in self._adapters.values()
            ],
        }

        with open(output_path, "w") as f:
            json.dump(catalog, f, indent=2)

    def __len__(self) -> int:
        return len(self._adapters)

    def __contains__(self, name: str) -> bool:
        return name in self._adapters

    def __iter__(self):
        return iter(self._adapters.values())
