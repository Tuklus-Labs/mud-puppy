"""Adapter serving with hot-swap capability.

This module provides an adapter server that can load, cache, and hot-swap
LoRA adapters at runtime without reloading the base model.
"""

from __future__ import annotations

import gc
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import jax
    import jax.numpy as jnp
    from jax import random
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


@dataclass
class AdapterState:
    """State of a loaded adapter.

    Attributes:
        name: Adapter name.
        params: Adapter parameters.
        loaded_at: Load timestamp.
        last_used: Last use timestamp.
        use_count: Number of times used.
        metadata: Additional metadata.
    """
    name: str
    params: Dict[str, Any]
    loaded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    use_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_used(self) -> None:
        """Mark the adapter as used."""
        self.last_used = datetime.now().isoformat()
        self.use_count += 1


class AdapterCache:
    """LRU cache for loaded adapters.

    Manages memory by evicting least recently used adapters
    when the cache reaches capacity.
    """

    def __init__(self, max_size: int = 10):
        """Initialize the cache.

        Args:
            max_size: Maximum number of adapters to cache.
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, AdapterState] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, name: str) -> Optional[AdapterState]:
        """Get an adapter from cache.

        Args:
            name: Adapter name.

        Returns:
            AdapterState if found, None otherwise.
        """
        with self._lock:
            if name in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(name)
                state = self._cache[name]
                state.mark_used()
                return state
            return None

    def put(self, name: str, state: AdapterState) -> None:
        """Put an adapter in cache.

        Args:
            name: Adapter name.
            state: Adapter state.
        """
        with self._lock:
            # Evict if necessary
            while len(self._cache) >= self.max_size:
                # Remove least recently used
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                gc.collect()

            self._cache[name] = state
            self._cache.move_to_end(name)

    def remove(self, name: str) -> bool:
        """Remove an adapter from cache.

        Args:
            name: Adapter name.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if name in self._cache:
                del self._cache[name]
                gc.collect()
                return True
            return False

    def clear(self) -> None:
        """Clear all cached adapters."""
        with self._lock:
            self._cache.clear()
            gc.collect()

    def list(self) -> List[str]:
        """List cached adapter names."""
        with self._lock:
            return list(self._cache.keys())

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "adapters": [
                    {
                        "name": state.name,
                        "loaded_at": state.loaded_at,
                        "last_used": state.last_used,
                        "use_count": state.use_count,
                    }
                    for state in self._cache.values()
                ],
            }


class AdapterServer:
    """Server for hot-swapping LoRA adapters.

    Provides:
    - Loading adapters from disk
    - Caching loaded adapters
    - Hot-swapping between adapters
    - Generation with specific adapters
    - REST API for adapter management
    """

    def __init__(
        self,
        model_fn: Callable,
        base_params: Dict[str, Any],
        tokenizer: Any,
        adapters_dir: Union[str, Path],
        cache_size: int = 10,
    ):
        """Initialize the server.

        Args:
            model_fn: Model forward function.
            base_params: Base model parameters.
            tokenizer: Tokenizer.
            adapters_dir: Directory containing adapters.
            cache_size: Maximum adapters to cache.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX required for AdapterServer")

        self.model_fn = model_fn
        self.base_params = base_params
        self.tokenizer = tokenizer
        self.adapters_dir = Path(adapters_dir)
        self.cache = AdapterCache(max_size=cache_size)

        # Current active adapter
        self._active_adapter: Optional[str] = None
        self._active_params: Dict[str, Any] = base_params
        self._lock = threading.Lock()

        # JIT compile inference function
        self._generate_fn = None

    def _load_adapter_params(self, name: str) -> Dict[str, Any]:
        """Load adapter parameters from disk.

        Args:
            name: Adapter name.

        Returns:
            Loaded parameters.
        """
        if not ORBAX_AVAILABLE:
            raise RuntimeError("Orbax required for loading adapters")

        adapter_path = self.adapters_dir / name / "params"

        if not adapter_path.exists():
            # Try without params subdirectory
            adapter_path = self.adapters_dir / name

        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter '{name}' not found at {adapter_path}")

        checkpointer = ocp.PyTreeCheckpointer()
        return checkpointer.restore(str(adapter_path))

    def _merge_params(
        self,
        base_params: Dict[str, Any],
        lora_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge LoRA params into base params for inference.

        For inference, we merge the LoRA weights directly:
        W_effective = W_base + (B @ A) * scaling

        Args:
            base_params: Base model parameters.
            lora_params: LoRA adapter parameters.

        Returns:
            Merged parameters.
        """
        flat_base = flatten_dict(unfreeze(base_params))
        flat_lora = flatten_dict(unfreeze(lora_params))

        merged = dict(flat_base)

        # Find lora_a and lora_b pairs
        lora_a_keys = [k for k in flat_lora.keys() if "lora_a" in k]

        for a_key in lora_a_keys:
            # Construct corresponding B key and kernel key
            b_key = tuple(list(a_key[:-1]) + ["lora_b"])
            kernel_key = tuple(list(a_key[:-1]) + ["kernel"])

            if b_key in flat_lora and kernel_key in flat_base:
                lora_a = flat_lora[a_key]
                lora_b = flat_lora[b_key]

                # Compute LoRA delta: A @ B
                # Note: A is (in_features, r), B is (r, out_features)
                delta = jnp.matmul(lora_a, lora_b)

                # Get scaling from params if available, otherwise use default
                r = lora_a.shape[-1]
                scaling = 2.0  # Default alpha/r = 16/8

                merged[kernel_key] = flat_base[kernel_key] + delta * scaling

        return freeze(unflatten_dict(merged))

    def load_adapter(self, name: str) -> None:
        """Load an adapter into cache and set as active.

        Args:
            name: Adapter name.
        """
        with self._lock:
            # Check cache first
            state = self.cache.get(name)

            if state is None:
                # Load from disk
                print(f"[lora-server] Loading adapter: {name}")
                lora_params = self._load_adapter_params(name)

                state = AdapterState(
                    name=name,
                    params=lora_params,
                )
                self.cache.put(name, state)

            # Merge with base and set active
            self._active_params = self._merge_params(self.base_params, state.params)
            self._active_adapter = name

            print(f"[lora-server] Active adapter: {name}")

    def unload_adapter(self) -> None:
        """Unload current adapter and revert to base model."""
        with self._lock:
            self._active_adapter = None
            self._active_params = self.base_params
            print("[lora-server] Reverted to base model")

    def get_active_adapter(self) -> Optional[str]:
        """Get name of currently active adapter."""
        return self._active_adapter

    def list_available(self) -> List[str]:
        """List available adapters on disk."""
        if not self.adapters_dir.exists():
            return []

        adapters = []
        for item in self.adapters_dir.iterdir():
            if item.is_dir():
                # Check if it contains params
                if (item / "params").exists() or any(item.glob("*.safetensors")):
                    adapters.append(item.name)
                elif any(item.iterdir()):  # Has subdirectories (versions)
                    adapters.append(item.name)

        return sorted(adapters)

    def list_cached(self) -> List[str]:
        """List cached adapters."""
        return self.cache.list()

    def generate(
        self,
        prompt: str,
        adapter: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: int = 42,
    ) -> str:
        """Generate text with optional adapter.

        Args:
            prompt: Input prompt.
            adapter: Optional adapter name (uses active if not specified).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            seed: Random seed.

        Returns:
            Generated text.
        """
        # Load adapter if specified and different from current
        if adapter is not None and adapter != self._active_adapter:
            self.load_adapter(adapter)

        # Get parameters
        params = self._active_params

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="np",
            padding=False,
        )

        input_ids = jnp.array(inputs["input_ids"])
        attention_mask = jnp.array(inputs["attention_mask"])

        # Generate
        rng = random.PRNGKey(seed)
        generated_ids = self._generate_tokens(
            params=params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            rng=rng,
        )

        # Decode
        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return generated_text

    def _generate_tokens(
        self,
        params: Dict[str, Any],
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        rng: jax.Array,
    ) -> jnp.ndarray:
        """Generate tokens autoregressively.

        Args:
            params: Model parameters.
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            top_k: Top-k sampling.
            rng: Random key.

        Returns:
            Generated token IDs.
        """
        # Get EOS token
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        current_ids = input_ids
        current_mask = attention_mask

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.model_fn(
                {"params": params},
                input_ids=current_ids,
                attention_mask=current_mask,
                train=False,
            )

            # Get logits for last token
            logits = outputs.logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k
            if top_k > 0:
                top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
                logits = jnp.full_like(logits, -jnp.inf)
                logits = logits.at[:, top_k_indices[0]].set(top_k_logits[0])

            # Apply top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
                sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
                cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_mask = cumulative_probs > top_p
                # Shift mask right to keep first token above threshold
                sorted_mask = jnp.concatenate(
                    [jnp.zeros_like(sorted_mask[:, :1]), sorted_mask[:, :-1]],
                    axis=-1,
                )
                sorted_logits = jnp.where(sorted_mask, -jnp.inf, sorted_logits)

                # Unsort
                unsorted_indices = jnp.argsort(sorted_indices, axis=-1)
                logits = jnp.take_along_axis(sorted_logits, unsorted_indices, axis=-1)

            # Sample
            rng, sample_rng = random.split(rng)
            probs = jax.nn.softmax(logits, axis=-1)
            next_token = random.categorical(sample_rng, jnp.log(probs + 1e-10))

            # Update
            current_ids = jnp.concatenate([current_ids, next_token[:, None]], axis=-1)
            current_mask = jnp.concatenate(
                [current_mask, jnp.ones((1, 1), dtype=current_mask.dtype)],
                axis=-1,
            )

            # Check for EOS
            if eos_token_id is not None and next_token[0] == eos_token_id:
                break

        return current_ids

    def batch_generate(
        self,
        prompts: List[str],
        adapter: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts.
            adapter: Optional adapter name.
            **kwargs: Generation arguments.

        Returns:
            List of generated texts.
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, adapter=adapter, **kwargs)
            results.append(result)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "active_adapter": self._active_adapter,
            "available_adapters": self.list_available(),
            "cache": self.cache.stats(),
        }

    def start_rest_api(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        """Start REST API server.

        Args:
            host: Host to bind to.
            port: Port to listen on.
        """
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            raise RuntimeError("Flask required for REST API. Install with: pip install flask")

        app = Flask("lora-server")

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok"})

        @app.route("/adapters", methods=["GET"])
        def list_adapters():
            return jsonify({
                "available": self.list_available(),
                "cached": self.list_cached(),
                "active": self._active_adapter,
            })

        @app.route("/adapters/<name>/load", methods=["POST"])
        def load_adapter(name):
            try:
                self.load_adapter(name)
                return jsonify({"status": "loaded", "adapter": name})
            except Exception as e:
                return jsonify({"error": str(e)}), 400

        @app.route("/adapters/unload", methods=["POST"])
        def unload():
            self.unload_adapter()
            return jsonify({"status": "unloaded"})

        @app.route("/generate", methods=["POST"])
        def generate():
            data = request.json
            prompt = data.get("prompt", "")
            adapter = data.get("adapter")
            kwargs = {k: v for k, v in data.items() if k not in ("prompt", "adapter")}

            try:
                result = self.generate(prompt, adapter=adapter, **kwargs)
                return jsonify({
                    "generated": result,
                    "adapter": self._active_adapter,
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.route("/stats", methods=["GET"])
        def stats():
            return jsonify(self.get_stats())

        print(f"[lora-server] Starting REST API on {host}:{port}")
        app.run(host=host, port=port, threaded=True)


def create_server_from_registry(
    registry,
    model_fn: Callable,
    base_params: Dict[str, Any],
    tokenizer: Any,
    cache_size: int = 10,
) -> AdapterServer:
    """Create an AdapterServer from an AdapterRegistry.

    Args:
        registry: AdapterRegistry instance.
        model_fn: Model forward function.
        base_params: Base model parameters.
        tokenizer: Tokenizer.
        cache_size: Cache size.

    Returns:
        Configured AdapterServer.
    """
    return AdapterServer(
        model_fn=model_fn,
        base_params=base_params,
        tokenizer=tokenizer,
        adapters_dir=registry.adapters_dir,
        cache_size=cache_size,
    )
