"""Batch training orchestrator for training multiple LoRA adapters.

This module provides the BatchTrainer class for efficiently training
multiple specialized adapters in sequence with proper resource management.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    from jax.sharding import Mesh
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import optax
    from flax.training import train_state
    import orbax.checkpoint as ocp
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False

from .config import LoraLibraryConfig, AdapterConfig
from .registry import AdapterRegistry
from .datasets import TaskDatasetLoader


class TrainingStatus(Enum):
    """Status of adapter training."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TrainingProgress:
    """Progress tracking for adapter training.

    Attributes:
        adapter_name: Name of the adapter.
        status: Current training status.
        current_step: Current training step.
        total_steps: Total training steps.
        current_epoch: Current epoch.
        total_epochs: Total epochs.
        loss: Current loss value.
        learning_rate: Current learning rate.
        start_time: Training start timestamp.
        elapsed_seconds: Elapsed time in seconds.
        eta_seconds: Estimated time remaining.
    """
    adapter_name: str
    status: TrainingStatus = TrainingStatus.PENDING
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    start_time: Optional[str] = None
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0

    @property
    def progress_pct(self) -> float:
        """Get progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return 100.0 * self.current_step / self.total_steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "status": self.status.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "start_time": self.start_time,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "progress_pct": self.progress_pct,
        }


@dataclass
class TrainingResult:
    """Result of training an adapter.

    Attributes:
        adapter_name: Name of the adapter.
        status: Final training status.
        final_loss: Final training loss.
        final_eval_loss: Final evaluation loss.
        best_checkpoint_path: Path to best checkpoint.
        training_steps: Total training steps completed.
        training_time_seconds: Total training time.
        metrics: Additional metrics.
        error_message: Error message if failed.
    """
    adapter_name: str
    status: TrainingStatus
    final_loss: float = 0.0
    final_eval_loss: float = 0.0
    best_checkpoint_path: str = ""
    training_steps: int = 0
    training_time_seconds: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "status": self.status.value,
            "final_loss": self.final_loss,
            "final_eval_loss": self.final_eval_loss,
            "best_checkpoint_path": self.best_checkpoint_path,
            "training_steps": self.training_steps,
            "training_time_seconds": self.training_time_seconds,
            "metrics": self.metrics,
            "error_message": self.error_message,
        }


class BatchTrainer:
    """Orchestrator for training multiple LoRA adapters.

    Handles:
    - Sequential training of multiple adapters
    - Resource management and memory cleanup
    - Checkpointing and resumption
    - Progress tracking and logging
    """

    def __init__(
        self,
        config: LoraLibraryConfig,
        registry: AdapterRegistry,
        model_loader: Optional[Callable] = None,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
    ):
        """Initialize the batch trainer.

        Args:
            config: Library configuration.
            registry: Adapter registry.
            model_loader: Optional custom model loader function.
            progress_callback: Optional callback for progress updates.
        """
        if not JAX_AVAILABLE or not FLAX_AVAILABLE:
            raise RuntimeError("JAX and Flax are required for BatchTrainer")

        self.config = config
        self.registry = registry
        self.model_loader = model_loader
        self.progress_callback = progress_callback

        # State
        self._model = None
        self._tokenizer = None
        self._base_params = None
        self._mesh: Optional[Mesh] = None

        # Training state
        self._results: Dict[str, TrainingResult] = {}
        self._progress: Dict[str, TrainingProgress] = {}

        # Setup directories
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create necessary directory structure."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "logs"), exist_ok=True)

    def _load_base_model(self) -> None:
        """Load the base model and tokenizer."""
        if self._model is not None:
            return

        print(f"[lora-library] Loading base model: {self.config.base_model_path}")

        if self.model_loader:
            self._model, self._base_params, self._tokenizer = self.model_loader(
                self.config.base_model_path
            )
        else:
            # Default loading using transformers + flax
            from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_path,
                trust_remote_code=self.config.trust_remote_code,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            model = FlaxAutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                trust_remote_code=self.config.trust_remote_code,
            )
            self._model = model.module
            self._base_params = model.params

        print("[lora-library] Base model loaded")

    def _create_mesh(self) -> Mesh:
        """Create device mesh for training."""
        if self._mesh is not None:
            return self._mesh

        from jax.experimental import mesh_utils

        devices = mesh_utils.create_device_mesh(self.config.tpu_mesh.mesh_shape)
        self._mesh = Mesh(devices, axis_names=self.config.tpu_mesh.axis_names)

        return self._mesh

    def _cleanup_memory(self) -> None:
        """Clean up GPU/TPU memory between adapters."""
        if not self.config.memory_cleanup_between_adapters:
            return

        print("[lora-library] Cleaning up memory...")

        # Clear JAX caches
        jax.clear_caches()

        # Python garbage collection
        gc.collect()

        # For GPU, we can try to clear the memory allocator
        try:
            jax.device_put(jnp.zeros(1))  # Small allocation to trigger cleanup
        except Exception:
            pass

    def _init_lora_params(
        self,
        adapter_config: AdapterConfig,
        rng: jax.Array,
    ) -> Dict[str, Any]:
        """Initialize LoRA parameters for an adapter.

        Args:
            adapter_config: Adapter configuration.
            rng: Random key.

        Returns:
            Combined base + LoRA parameters.
        """
        # Import the LoRA utilities from mud_puppy
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from mud_puppy.jax_lora import LoRAConfig, init_lora_from_pretrained

        lora_config = LoRAConfig(
            r=adapter_config.r,
            alpha=adapter_config.alpha,
            dropout=adapter_config.dropout,
            target_modules=adapter_config.target_modules,
            use_rslora=adapter_config.use_rslora,
            bias=adapter_config.bias,
        )

        return init_lora_from_pretrained(self._base_params, lora_config, rng)

    def _create_optimizer(
        self,
        adapter_config: AdapterConfig,
        num_train_steps: int,
    ) -> optax.GradientTransformation:
        """Create optimizer for training.

        Args:
            adapter_config: Adapter configuration.
            num_train_steps: Total training steps.

        Returns:
            Optax optimizer.
        """
        hp = adapter_config.hyperparameters
        warmup_steps = int(num_train_steps * hp.warmup_ratio)

        # Build learning rate schedule
        if hp.lr_scheduler == "cosine":
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=hp.learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=num_train_steps,
                end_value=hp.learning_rate * 0.1,
            )
        elif hp.lr_scheduler == "linear":
            schedule = optax.join_schedules(
                schedules=[
                    optax.linear_schedule(0.0, hp.learning_rate, warmup_steps),
                    optax.linear_schedule(hp.learning_rate, 0.0, num_train_steps - warmup_steps),
                ],
                boundaries=[warmup_steps],
            )
        else:
            schedule = optax.constant_schedule(hp.learning_rate)

        return optax.chain(
            optax.clip_by_global_norm(hp.max_grad_norm),
            optax.adamw(learning_rate=schedule, weight_decay=hp.weight_decay),
        )

    def _load_dataset(self, adapter_config: AdapterConfig):
        """Load dataset for an adapter.

        Args:
            adapter_config: Adapter configuration.

        Returns:
            Loaded dataset.
        """
        if adapter_config.dataset is None:
            raise ValueError(f"No dataset configured for adapter {adapter_config.name}")

        dataset = TaskDatasetLoader.load(
            task_type=adapter_config.task,
            path=adapter_config.dataset.path,
        )

        return dataset.create_hf_dataset(
            tokenizer=self._tokenizer,
            max_length=adapter_config.dataset.max_length,
            num_proc=adapter_config.dataset.preprocessing_workers,
        )

    def _train_single_adapter(
        self,
        adapter_config: AdapterConfig,
        rng: jax.Array,
    ) -> TrainingResult:
        """Train a single adapter.

        Args:
            adapter_config: Adapter configuration.
            rng: Random key.

        Returns:
            Training result.
        """
        adapter_name = adapter_config.name
        hp = adapter_config.hyperparameters

        # Initialize progress
        progress = TrainingProgress(
            adapter_name=adapter_name,
            status=TrainingStatus.IN_PROGRESS,
            start_time=datetime.now().isoformat(),
            total_epochs=hp.num_epochs,
        )
        self._progress[adapter_name] = progress

        start_time = time.time()

        try:
            print(f"[lora-library] Training adapter: {adapter_name}")
            print(f"[lora-library]   Task: {adapter_config.task}")
            print(f"[lora-library]   Rank: {adapter_config.r}, Alpha: {adapter_config.alpha}")

            # Load dataset
            print(f"[lora-library]   Loading dataset...")
            dataset = self._load_dataset(adapter_config)
            num_examples = len(dataset)
            print(f"[lora-library]   Dataset: {num_examples} examples")

            # Calculate steps
            steps_per_epoch = num_examples // hp.batch_size
            num_train_steps = steps_per_epoch * hp.num_epochs
            if hp.max_steps:
                num_train_steps = min(num_train_steps, hp.max_steps)

            progress.total_steps = num_train_steps

            # Initialize LoRA parameters
            rng, init_rng = random.split(rng)
            params = self._init_lora_params(adapter_config, init_rng)

            # Create optimizer
            optimizer = self._create_optimizer(adapter_config, num_train_steps)

            # Create train state
            state = train_state.TrainState.create(
                apply_fn=self._model.apply,
                params=params,
                tx=optimizer,
            )

            # Check for checkpoint to resume
            checkpoint_dir = self.config.get_checkpoint_dir(adapter_name)
            os.makedirs(checkpoint_dir, exist_ok=True)

            ckpt_manager = ocp.CheckpointManager(
                checkpoint_dir,
                options=ocp.CheckpointManagerOptions(
                    max_to_keep=self.config.save_total_limit,
                ),
            )

            start_step = 0
            if self.config.resume_from_checkpoint:
                latest = ckpt_manager.latest_step()
                if latest is not None:
                    print(f"[lora-library]   Resuming from step {latest}")
                    restored = ckpt_manager.restore(latest)
                    state = state.replace(
                        params=restored["params"],
                        opt_state=restored["opt_state"],
                        step=restored["step"],
                    )
                    start_step = int(restored["step"])

            # Define training step
            @jax.jit
            def train_step(state, batch, dropout_rng):
                def loss_fn(params):
                    outputs = state.apply_fn(
                        {"params": params},
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        train=True,
                        rngs={"dropout": dropout_rng},
                    )
                    logits = outputs.logits[:, :-1, :]
                    labels = batch["labels"][:, 1:]
                    mask = batch["attention_mask"][:, 1:]

                    vocab_size = logits.shape[-1]
                    loss = optax.softmax_cross_entropy_with_integer_labels(
                        logits.reshape(-1, vocab_size),
                        labels.reshape(-1),
                    )
                    loss = (loss * mask.reshape(-1)).sum() / mask.sum()
                    return loss

                loss, grads = jax.value_and_grad(loss_fn)(state.params)
                state = state.apply_gradients(grads=grads)
                return state, loss

            # Training loop
            rng, train_rng = random.split(rng)
            best_loss = float("inf")
            global_step = start_step

            for epoch in range(hp.num_epochs):
                progress.current_epoch = epoch + 1

                # Shuffle dataset
                epoch_rng = random.fold_in(train_rng, epoch)
                indices = random.permutation(epoch_rng, num_examples)

                epoch_loss = 0.0
                epoch_steps = 0

                for batch_start in range(0, num_examples, hp.batch_size):
                    if global_step >= num_train_steps:
                        break

                    batch_indices = indices[batch_start:batch_start + hp.batch_size]
                    if len(batch_indices) < hp.batch_size:
                        continue

                    batch = dataset.select(batch_indices.tolist())
                    batch = {
                        "input_ids": jnp.array(batch["input_ids"]),
                        "attention_mask": jnp.array(batch["attention_mask"]),
                        "labels": jnp.array(batch["labels"]),
                    }

                    rng, step_rng = random.split(rng)
                    state, loss = train_step(state, batch, step_rng)

                    epoch_loss += float(loss)
                    epoch_steps += 1
                    global_step += 1

                    # Update progress
                    progress.current_step = global_step
                    progress.loss = float(loss)
                    elapsed = time.time() - start_time
                    progress.elapsed_seconds = elapsed
                    if global_step > 0:
                        progress.eta_seconds = (elapsed / global_step) * (num_train_steps - global_step)

                    # Log
                    if global_step % self.config.log_interval == 0:
                        print(
                            f"[lora-library]   Step {global_step}/{num_train_steps} | "
                            f"Loss: {loss:.4f} | Epoch: {epoch + 1}/{hp.num_epochs}"
                        )

                        if self.progress_callback:
                            self.progress_callback(progress)

                    # Checkpoint
                    if global_step % self.config.checkpoint_interval == 0:
                        ckpt_manager.save(
                            global_step,
                            args=ocp.args.PyTreeSave({
                                "params": state.params,
                                "opt_state": state.opt_state,
                                "step": global_step,
                            }),
                        )

                        if epoch_loss / max(epoch_steps, 1) < best_loss:
                            best_loss = epoch_loss / max(epoch_steps, 1)

                avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
                print(f"[lora-library]   Epoch {epoch + 1} complete | Avg Loss: {avg_epoch_loss:.4f}")

            # Save final checkpoint
            final_dir = self.config.get_adapter_output_dir(adapter_name)
            os.makedirs(final_dir, exist_ok=True)

            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(os.path.join(final_dir, "params"), state.params)

            # Register version in registry
            if adapter_name not in self.registry:
                self.registry.register(
                    name=adapter_name,
                    task=adapter_config.task,
                    base_model=self.config.base_model_path,
                    description=adapter_config.description,
                    lora_config={
                        "r": adapter_config.r,
                        "alpha": adapter_config.alpha,
                        "dropout": adapter_config.dropout,
                        "target_modules": adapter_config.target_modules,
                    },
                )

            version = f"v{len(self.registry.get(adapter_name).versions) + 1}"
            self.registry.add_version(
                name=adapter_name,
                version=version,
                checkpoint_path=final_dir,
                training_steps=global_step,
                final_loss=best_loss,
            )

            # Create result
            training_time = time.time() - start_time
            result = TrainingResult(
                adapter_name=adapter_name,
                status=TrainingStatus.COMPLETED,
                final_loss=best_loss,
                best_checkpoint_path=final_dir,
                training_steps=global_step,
                training_time_seconds=training_time,
            )

            progress.status = TrainingStatus.COMPLETED
            print(f"[lora-library] Adapter {adapter_name} completed in {training_time:.1f}s")

            return result

        except Exception as e:
            print(f"[lora-library] Training failed for {adapter_name}: {e}")
            progress.status = TrainingStatus.FAILED

            return TrainingResult(
                adapter_name=adapter_name,
                status=TrainingStatus.FAILED,
                error_message=str(e),
                training_time_seconds=time.time() - start_time,
            )

    def train_adapter(
        self,
        adapter_name: str,
        seed: int = 42,
    ) -> TrainingResult:
        """Train a single adapter by name.

        Args:
            adapter_name: Name of the adapter to train.
            seed: Random seed.

        Returns:
            Training result.
        """
        adapter_config = self.config.get_adapter(adapter_name)
        if adapter_config is None:
            raise KeyError(f"Adapter '{adapter_name}' not found in config")

        # Load base model if needed
        self._load_base_model()

        # Train
        rng = random.PRNGKey(seed)
        result = self._train_single_adapter(adapter_config, rng)

        self._results[adapter_name] = result
        return result

    def train_all(
        self,
        adapters: Optional[List[str]] = None,
        seed: int = 42,
    ) -> Dict[str, TrainingResult]:
        """Train all configured adapters (or a subset).

        Args:
            adapters: Optional list of adapter names to train.
            seed: Random seed.

        Returns:
            Dictionary of adapter name to training result.
        """
        # Load base model once
        self._load_base_model()

        # Get adapters to train
        if adapters:
            configs = [self.config.get_adapter(name) for name in adapters]
            configs = [c for c in configs if c is not None and c.enabled]
        else:
            configs = self.config.get_enabled_adapters()

        print(f"[lora-library] Training {len(configs)} adapters")

        # Train each adapter
        rng = random.PRNGKey(seed)

        for i, adapter_config in enumerate(configs):
            print(f"\n[lora-library] === Adapter {i + 1}/{len(configs)}: {adapter_config.name} ===")

            rng, adapter_rng = random.split(rng)
            result = self._train_single_adapter(adapter_config, adapter_rng)
            self._results[adapter_config.name] = result

            # Cleanup between adapters
            if i < len(configs) - 1:
                self._cleanup_memory()

        # Save summary
        self._save_training_summary()

        return self._results

    def _save_training_summary(self) -> None:
        """Save a summary of training results."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "base_model": self.config.base_model_path,
            "total_adapters": len(self._results),
            "successful": sum(1 for r in self._results.values() if r.status == TrainingStatus.COMPLETED),
            "failed": sum(1 for r in self._results.values() if r.status == TrainingStatus.FAILED),
            "results": {name: r.to_dict() for name, r in self._results.items()},
        }

        summary_path = os.path.join(self.config.output_dir, "logs", "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[lora-library] Training summary saved to {summary_path}")

    def get_progress(self, adapter_name: Optional[str] = None) -> Union[TrainingProgress, Dict[str, TrainingProgress]]:
        """Get training progress.

        Args:
            adapter_name: Optional specific adapter name.

        Returns:
            Progress for one adapter or all adapters.
        """
        if adapter_name:
            return self._progress.get(adapter_name)
        return self._progress

    def get_results(self, adapter_name: Optional[str] = None) -> Union[TrainingResult, Dict[str, TrainingResult]]:
        """Get training results.

        Args:
            adapter_name: Optional specific adapter name.

        Returns:
            Result for one adapter or all adapters.
        """
        if adapter_name:
            return self._results.get(adapter_name)
        return self._results

    def resume_failed(self, seed: int = 42) -> Dict[str, TrainingResult]:
        """Resume training for failed adapters.

        Args:
            seed: Random seed.

        Returns:
            Results for retrained adapters.
        """
        failed = [
            name for name, result in self._results.items()
            if result.status == TrainingStatus.FAILED
        ]

        if not failed:
            print("[lora-library] No failed adapters to resume")
            return {}

        print(f"[lora-library] Resuming {len(failed)} failed adapters")
        return self.train_all(adapters=failed, seed=seed)
