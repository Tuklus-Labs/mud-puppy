"""Adapter merging and manipulation operations.

This module provides functionality for merging, averaging, and interpolating
between LoRA adapters to create hybrid specialized adapters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import jax
    import jax.numpy as jnp
    from flax.core import freeze, unfreeze
    from flax.traverse_util import flatten_dict, unflatten_dict
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import orbax.checkpoint as ocp
    ORBAX_AVAILABLE = True
except ImportError:
    ORBAX_AVAILABLE = False


class MergeStrategy(Enum):
    """Strategy for merging adapters."""
    AVERAGE = "average"  # Simple average of weights
    WEIGHTED = "weighted"  # Weighted average
    TIES = "ties"  # TIES merging (trim, elect sign, merge)
    DARE = "dare"  # DARE (drop and rescale) merging
    TASK_ARITHMETIC = "task_arithmetic"  # Task arithmetic
    LINEAR_INTERPOLATION = "linear_interpolation"  # Linear interpolation between two adapters


@dataclass
class MergeResult:
    """Result of merging adapters.

    Attributes:
        merged_params: Merged parameters.
        source_adapters: List of source adapter names.
        strategy: Merge strategy used.
        weights: Weights used (if applicable).
        metadata: Additional metadata.
    """
    merged_params: Dict[str, Any]
    source_adapters: List[str]
    strategy: MergeStrategy
    weights: Optional[List[float]] = None
    metadata: Dict[str, Any] = None

    def save(self, path: Union[str, Path]) -> None:
        """Save merged params to disk.

        Args:
            path: Output path.
        """
        if not ORBAX_AVAILABLE:
            raise RuntimeError("Orbax required for saving")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(str(path / "params"), self.merged_params)

        # Save metadata
        import json
        meta = {
            "source_adapters": self.source_adapters,
            "strategy": self.strategy.value,
            "weights": self.weights,
            "metadata": self.metadata or {},
        }
        with open(path / "merge_info.json", "w") as f:
            json.dump(meta, f, indent=2)


class AdapterMerger:
    """Merges multiple LoRA adapters using various strategies.

    Supports:
    - Simple averaging (LoRA soups)
    - Weighted averaging
    - TIES merging
    - DARE merging
    - Task arithmetic
    - Linear interpolation
    """

    def __init__(self, base_params: Optional[Dict[str, Any]] = None):
        """Initialize the merger.

        Args:
            base_params: Optional base model parameters for reference.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX required for AdapterMerger")

        self.base_params = base_params

    def merge(
        self,
        adapters: List[Dict[str, Any]],
        adapter_names: List[str],
        strategy: MergeStrategy = MergeStrategy.AVERAGE,
        weights: Optional[List[float]] = None,
        **kwargs,
    ) -> MergeResult:
        """Merge multiple adapters.

        Args:
            adapters: List of adapter parameters.
            adapter_names: Names of the adapters.
            strategy: Merge strategy.
            weights: Optional weights for weighted strategies.
            **kwargs: Strategy-specific arguments.

        Returns:
            MergeResult with merged parameters.
        """
        if len(adapters) < 2:
            raise ValueError("At least 2 adapters required for merging")

        if weights is not None and len(weights) != len(adapters):
            raise ValueError("Weights must match number of adapters")

        # Normalize weights if provided
        if weights is not None:
            total = sum(weights)
            weights = [w / total for w in weights]

        # Dispatch to appropriate merge function
        if strategy == MergeStrategy.AVERAGE:
            merged = self._average_merge(adapters, weights)
        elif strategy == MergeStrategy.WEIGHTED:
            if weights is None:
                weights = [1.0 / len(adapters)] * len(adapters)
            merged = self._weighted_merge(adapters, weights)
        elif strategy == MergeStrategy.TIES:
            merged = self._ties_merge(adapters, weights, **kwargs)
        elif strategy == MergeStrategy.DARE:
            merged = self._dare_merge(adapters, weights, **kwargs)
        elif strategy == MergeStrategy.TASK_ARITHMETIC:
            merged = self._task_arithmetic_merge(adapters, weights, **kwargs)
        elif strategy == MergeStrategy.LINEAR_INTERPOLATION:
            if len(adapters) != 2:
                raise ValueError("Linear interpolation requires exactly 2 adapters")
            alpha = kwargs.get("alpha", 0.5)
            merged = self._linear_interpolation(adapters[0], adapters[1], alpha)
            weights = [1 - alpha, alpha]
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

        return MergeResult(
            merged_params=merged,
            source_adapters=adapter_names,
            strategy=strategy,
            weights=weights,
            metadata=kwargs,
        )

    def _average_merge(
        self,
        adapters: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Simple average of adapter weights.

        Args:
            adapters: List of adapter parameters.
            weights: Optional weights (defaults to equal).

        Returns:
            Averaged parameters.
        """
        if weights is None:
            weights = [1.0 / len(adapters)] * len(adapters)

        return self._weighted_merge(adapters, weights)

    def _weighted_merge(
        self,
        adapters: List[Dict[str, Any]],
        weights: List[float],
    ) -> Dict[str, Any]:
        """Weighted average of adapter weights.

        Args:
            adapters: List of adapter parameters.
            weights: Weights for each adapter.

        Returns:
            Weighted averaged parameters.
        """
        # Flatten all adapters
        flat_adapters = [flatten_dict(unfreeze(a)) for a in adapters]

        # Get all keys
        all_keys = set()
        for flat in flat_adapters:
            all_keys.update(flat.keys())

        # Merge each key
        merged = {}
        for key in all_keys:
            values = []
            ws = []

            for i, flat in enumerate(flat_adapters):
                if key in flat:
                    values.append(flat[key])
                    ws.append(weights[i])

            if values:
                # Normalize weights for this key
                total_w = sum(ws)
                ws = [w / total_w for w in ws]

                # Weighted average
                merged[key] = sum(v * w for v, w in zip(values, ws))

        return freeze(unflatten_dict(merged))

    def _ties_merge(
        self,
        adapters: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        density: float = 0.9,
        **kwargs,
    ) -> Dict[str, Any]:
        """TIES merging: Trim, Elect Sign, Merge.

        This method:
        1. Trims small weights (keeps top density %)
        2. Resolves sign conflicts by majority vote
        3. Merges with disjoint means

        Args:
            adapters: List of adapter parameters.
            weights: Optional weights.
            density: Fraction of weights to keep (0.0 to 1.0).

        Returns:
            TIES-merged parameters.
        """
        if weights is None:
            weights = [1.0] * len(adapters)

        # Flatten adapters
        flat_adapters = [flatten_dict(unfreeze(a)) for a in adapters]

        # Get deltas from base if available
        if self.base_params is not None:
            flat_base = flatten_dict(unfreeze(self.base_params))
            deltas = []
            for flat in flat_adapters:
                delta = {}
                for key in flat:
                    if key in flat_base:
                        delta[key] = flat[key] - flat_base[key]
                    else:
                        delta[key] = flat[key]
                deltas.append(delta)
        else:
            deltas = flat_adapters

        # Get all keys
        all_keys = set()
        for delta in deltas:
            all_keys.update(delta.keys())

        merged = {}
        for key in all_keys:
            values = []
            ws = []

            for i, delta in enumerate(deltas):
                if key in delta:
                    values.append(delta[key])
                    ws.append(weights[i])

            if not values:
                continue

            # Stack values for vectorized operations
            stacked = jnp.stack(values)

            # 1. Trim: keep top density% by magnitude
            if density < 1.0:
                magnitudes = jnp.abs(stacked)
                threshold = jnp.percentile(magnitudes, (1 - density) * 100)
                mask = magnitudes >= threshold
                stacked = jnp.where(mask, stacked, 0.0)

            # 2. Elect sign: majority vote
            signs = jnp.sign(stacked)
            sign_sum = jnp.sum(signs, axis=0)
            elected_sign = jnp.sign(sign_sum)
            elected_sign = jnp.where(elected_sign == 0, 1.0, elected_sign)

            # 3. Merge: disjoint mean (only values with matching sign)
            matching = stacked * (signs == elected_sign)
            counts = jnp.sum(signs == elected_sign, axis=0)
            merged_value = jnp.sum(matching, axis=0) / jnp.maximum(counts, 1)

            merged[key] = merged_value

        # Add back to base if we have it
        if self.base_params is not None:
            for key in flat_base:
                if key in merged:
                    merged[key] = flat_base[key] + merged[key]
                else:
                    merged[key] = flat_base[key]

        return freeze(unflatten_dict(merged))

    def _dare_merge(
        self,
        adapters: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        drop_rate: float = 0.1,
        rescale: bool = True,
        seed: int = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        """DARE merging: Drop and Rescale.

        Randomly drops weights and rescales to compensate.

        Args:
            adapters: List of adapter parameters.
            weights: Optional weights.
            drop_rate: Fraction of weights to drop.
            rescale: Whether to rescale remaining weights.
            seed: Random seed.

        Returns:
            DARE-merged parameters.
        """
        if weights is None:
            weights = [1.0 / len(adapters)] * len(adapters)

        rng = jax.random.PRNGKey(seed)

        # Flatten adapters
        flat_adapters = [flatten_dict(unfreeze(a)) for a in adapters]

        # Get all keys
        all_keys = set()
        for flat in flat_adapters:
            all_keys.update(flat.keys())

        merged = {}
        for key in all_keys:
            rng, drop_rng = jax.random.split(rng)

            values = []
            ws = []

            for i, flat in enumerate(flat_adapters):
                if key in flat:
                    value = flat[key]

                    # Drop random weights
                    drop_mask = jax.random.bernoulli(
                        jax.random.fold_in(drop_rng, i),
                        p=drop_rate,
                        shape=value.shape,
                    )

                    value = jnp.where(drop_mask, 0.0, value)

                    # Rescale
                    if rescale and drop_rate < 1.0:
                        value = value / (1.0 - drop_rate)

                    values.append(value)
                    ws.append(weights[i])

            if values:
                # Weighted average
                total_w = sum(ws)
                ws = [w / total_w for w in ws]
                merged[key] = sum(v * w for v, w in zip(values, ws))

        return freeze(unflatten_dict(merged))

    def _task_arithmetic_merge(
        self,
        adapters: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        scaling: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Task arithmetic merging.

        Computes task vectors and adds them to base model.

        Args:
            adapters: List of adapter parameters.
            weights: Coefficients for each task vector.
            scaling: Global scaling factor.

        Returns:
            Task-arithmetic merged parameters.
        """
        if self.base_params is None:
            raise ValueError("Base params required for task arithmetic")

        if weights is None:
            weights = [1.0] * len(adapters)

        # Flatten
        flat_base = flatten_dict(unfreeze(self.base_params))
        flat_adapters = [flatten_dict(unfreeze(a)) for a in adapters]

        # Compute task vectors (delta from base)
        task_vectors = []
        for flat in flat_adapters:
            tv = {}
            for key in flat:
                if key in flat_base:
                    tv[key] = flat[key] - flat_base[key]
            task_vectors.append(tv)

        # Combine task vectors
        all_keys = set()
        for tv in task_vectors:
            all_keys.update(tv.keys())

        combined_tv = {}
        for key in all_keys:
            values = []
            ws = []
            for i, tv in enumerate(task_vectors):
                if key in tv:
                    values.append(tv[key])
                    ws.append(weights[i])

            if values:
                combined_tv[key] = sum(v * w for v, w in zip(values, ws)) * scaling

        # Add combined task vector to base
        merged = {}
        for key in flat_base:
            if key in combined_tv:
                merged[key] = flat_base[key] + combined_tv[key]
            else:
                merged[key] = flat_base[key]

        return freeze(unflatten_dict(merged))

    def _linear_interpolation(
        self,
        adapter_a: Dict[str, Any],
        adapter_b: Dict[str, Any],
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """Linear interpolation between two adapters.

        result = (1 - alpha) * adapter_a + alpha * adapter_b

        Args:
            adapter_a: First adapter parameters.
            adapter_b: Second adapter parameters.
            alpha: Interpolation factor (0 = all A, 1 = all B).

        Returns:
            Interpolated parameters.
        """
        flat_a = flatten_dict(unfreeze(adapter_a))
        flat_b = flatten_dict(unfreeze(adapter_b))

        merged = {}
        all_keys = set(flat_a.keys()) | set(flat_b.keys())

        for key in all_keys:
            if key in flat_a and key in flat_b:
                merged[key] = (1 - alpha) * flat_a[key] + alpha * flat_b[key]
            elif key in flat_a:
                merged[key] = (1 - alpha) * flat_a[key]
            else:
                merged[key] = alpha * flat_b[key]

        return freeze(unflatten_dict(merged))

    def create_lora_soup(
        self,
        adapters: List[Dict[str, Any]],
        adapter_names: List[str],
    ) -> MergeResult:
        """Create a LoRA soup by averaging adapters.

        This is a convenient alias for average merging.

        Args:
            adapters: List of adapter parameters.
            adapter_names: Names of the adapters.

        Returns:
            MergeResult with averaged parameters.
        """
        return self.merge(adapters, adapter_names, MergeStrategy.AVERAGE)

    def interpolate_adapters(
        self,
        adapter_a: Dict[str, Any],
        adapter_b: Dict[str, Any],
        name_a: str,
        name_b: str,
        num_points: int = 5,
    ) -> List[Tuple[float, MergeResult]]:
        """Generate interpolations between two adapters.

        Args:
            adapter_a: First adapter.
            adapter_b: Second adapter.
            name_a: Name of first adapter.
            name_b: Name of second adapter.
            num_points: Number of interpolation points.

        Returns:
            List of (alpha, MergeResult) tuples.
        """
        results = []
        alphas = jnp.linspace(0, 1, num_points)

        for alpha in alphas:
            result = self.merge(
                [adapter_a, adapter_b],
                [name_a, name_b],
                strategy=MergeStrategy.LINEAR_INTERPOLATION,
                alpha=float(alpha),
            )
            results.append((float(alpha), result))

        return results

    def subtract_adapter(
        self,
        adapter: Dict[str, Any],
        to_subtract: Dict[str, Any],
        scale: float = 1.0,
    ) -> Dict[str, Any]:
        """Subtract one adapter from another.

        Useful for removing unwanted behaviors.

        Args:
            adapter: Base adapter parameters.
            to_subtract: Adapter to subtract.
            scale: Scaling factor for subtraction.

        Returns:
            Result parameters.
        """
        flat_a = flatten_dict(unfreeze(adapter))
        flat_b = flatten_dict(unfreeze(to_subtract))

        result = {}
        for key in flat_a:
            if key in flat_b:
                result[key] = flat_a[key] - scale * flat_b[key]
            else:
                result[key] = flat_a[key]

        return freeze(unflatten_dict(result))

    def scale_adapter(
        self,
        adapter: Dict[str, Any],
        scale: float,
    ) -> Dict[str, Any]:
        """Scale adapter weights by a factor.

        Args:
            adapter: Adapter parameters.
            scale: Scaling factor.

        Returns:
            Scaled parameters.
        """
        flat = flatten_dict(unfreeze(adapter))
        scaled = {key: value * scale for key, value in flat.items()}
        return freeze(unflatten_dict(scaled))
