"""JAX/Flax training module for mud-puppy.

This module provides a JAX-native training implementation using:
- Flax TrainState for model state management
- Optax for optimizers and learning rate schedules
- Orbax for checkpointing
- shard_map for distributed training across TPU/GPU devices
"""

import os
import time
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

import flax
from flax import linen as nn
from flax.training import train_state
from flax.training import orbax_utils

import optax
import orbax.checkpoint as ocp

from datasets import load_dataset
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

from .config import TrainingConfig


# Type aliases
PRNGKey = jax.Array
Params = Dict[str, Any]
Batch = Dict[str, jax.Array]
Metrics = Dict[str, float]


def get_dtype(precision: str) -> jnp.dtype:
    """Get JAX dtype from precision string."""
    dtype_map = {
        "fp16": jnp.float16,
        "bf16": jnp.bfloat16,
        "fp32": jnp.float32,
    }
    return dtype_map.get(precision, jnp.bfloat16)


def create_mesh(
    device_type: str = "auto",
    mesh_shape: Optional[Tuple[int, ...]] = None,
) -> Mesh:
    """Create a device mesh for distributed training.

    Args:
        device_type: 'tpu', 'gpu', or 'auto' to detect
        mesh_shape: Shape of the device mesh (data, model) parallelism

    Returns:
        JAX Mesh for sharding
    """
    devices = jax.devices()
    num_devices = len(devices)

    if mesh_shape is None:
        # Default: pure data parallelism
        mesh_shape = (num_devices, 1)

    # Reshape devices into mesh
    devices_array = jnp.array(devices).reshape(mesh_shape)

    return Mesh(devices_array, axis_names=("data", "model"))


@dataclass
class TrainMetrics:
    """Container for training metrics."""
    loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    step: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
            "tokens_per_second": self.tokens_per_second,
            "step": self.step,
        }


class TrainStateWithMetrics(train_state.TrainState):
    """Extended TrainState with gradient accumulation and metrics support."""

    # Gradient accumulation buffer
    grad_accum: Optional[Params] = None
    accum_steps: int = 0

    # Dropout key
    dropout_rng: Optional[PRNGKey] = None

    def apply_gradients(self, *, grads: Params, **kwargs) -> "TrainStateWithMetrics":
        """Apply gradients with optional accumulation."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )


def create_learning_rate_schedule(
    config: TrainingConfig,
    num_train_steps: int,
    warmup_ratio: float = 0.1,
) -> optax.Schedule:
    """Create learning rate schedule based on config.

    Args:
        config: Training configuration
        num_train_steps: Total number of training steps
        warmup_ratio: Fraction of steps for warmup

    Returns:
        Optax schedule function
    """
    warmup_steps = int(num_train_steps * warmup_ratio)

    if config.lr_scheduler == "linear":
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=config.learning_rate,
                    transition_steps=warmup_steps,
                ),
                optax.linear_schedule(
                    init_value=config.learning_rate,
                    end_value=0.0,
                    transition_steps=num_train_steps - warmup_steps,
                ),
            ],
            boundaries=[warmup_steps],
        )
    elif config.lr_scheduler == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=num_train_steps,
            end_value=config.learning_rate * 0.1,
        )
    elif config.lr_scheduler == "cosine_with_restarts":
        # Cosine annealing with warm restarts
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=config.learning_rate,
                    transition_steps=warmup_steps,
                ),
                optax.cosine_decay_schedule(
                    init_value=config.learning_rate,
                    decay_steps=(num_train_steps - warmup_steps) // 3,
                    alpha=0.1,
                ),
            ],
            boundaries=[warmup_steps],
        )
    elif config.lr_scheduler == "polynomial":
        schedule = optax.polynomial_schedule(
            init_value=config.learning_rate,
            end_value=0.0,
            power=2.0,
            transition_steps=num_train_steps,
        )
    else:
        # Fallback to constant
        schedule = optax.constant_schedule(config.learning_rate)

    return schedule


def create_optimizer(
    config: TrainingConfig,
    num_train_steps: int,
) -> optax.GradientTransformation:
    """Create optimizer with learning rate schedule and gradient clipping.

    Args:
        config: Training configuration
        num_train_steps: Total number of training steps

    Returns:
        Optax gradient transformation (optimizer)
    """
    schedule = create_learning_rate_schedule(config, num_train_steps)

    # Build optimizer chain
    optimizer = optax.chain(
        # Gradient clipping
        optax.clip_by_global_norm(config.max_grad_norm),
        # AdamW with weight decay
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.01,
        ),
    )

    # Wrap with gradient accumulation if needed
    if config.gradient_accumulation > 1:
        optimizer = optax.MultiSteps(
            optimizer,
            every_k_schedule=config.gradient_accumulation,
        )

    return optimizer


def create_train_state(
    model: nn.Module,
    params: Params,
    config: TrainingConfig,
    num_train_steps: int,
    rng: PRNGKey,
) -> TrainStateWithMetrics:
    """Create initial training state.

    Args:
        model: Flax model
        params: Model parameters
        config: Training configuration
        num_train_steps: Total training steps
        rng: Random key

    Returns:
        Initialized TrainState
    """
    tx = create_optimizer(config, num_train_steps)

    # Split RNG for dropout
    dropout_rng = random.fold_in(rng, 0)

    return TrainStateWithMetrics.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dropout_rng=dropout_rng,
    )


def compute_loss(
    params: Params,
    apply_fn: Callable,
    batch: Batch,
    dropout_rng: PRNGKey,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute causal language modeling loss.

    Args:
        params: Model parameters
        apply_fn: Model forward function
        batch: Input batch with input_ids, attention_mask, labels
        dropout_rng: RNG for dropout
        dtype: Computation dtype

    Returns:
        Tuple of (loss, metrics dict)
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    # Forward pass
    outputs = apply_fn(
        {"params": params},
        input_ids=input_ids,
        attention_mask=attention_mask,
        train=True,
        rngs={"dropout": dropout_rng},
    )

    logits = outputs.logits.astype(jnp.float32)

    # Shift logits and labels for causal LM
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = attention_mask[:, 1:]

    # Cross-entropy loss
    vocab_size = shift_logits.shape[-1]
    one_hot_labels = jax.nn.one_hot(shift_labels, vocab_size)

    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    loss_per_token = -jnp.sum(one_hot_labels * log_probs, axis=-1)

    # Mask padding tokens
    loss_per_token = loss_per_token * shift_mask

    # Average loss
    total_tokens = jnp.sum(shift_mask)
    loss = jnp.sum(loss_per_token) / jnp.maximum(total_tokens, 1)

    metrics = {
        "loss": loss,
        "num_tokens": total_tokens,
    }

    return loss, metrics


def train_step(
    state: TrainStateWithMetrics,
    batch: Batch,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Tuple[TrainStateWithMetrics, Metrics]:
    """Single training step.

    Args:
        state: Current training state
        batch: Input batch
        dtype: Computation dtype

    Returns:
        Tuple of (new state, metrics)
    """
    # Get new dropout key
    dropout_rng, new_dropout_rng = random.split(state.dropout_rng)

    # Compute gradients
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, aux_metrics), grads = grad_fn(
        state.params,
        state.apply_fn,
        batch,
        dropout_rng,
        dtype,
    )

    # Compute gradient norm before clipping
    grad_norm = optax.global_norm(grads)

    # Apply gradients
    state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)

    metrics = {
        "loss": loss,
        "grad_norm": grad_norm,
        **aux_metrics,
    }

    return state, metrics


def eval_step(
    state: TrainStateWithMetrics,
    batch: Batch,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Metrics:
    """Single evaluation step.

    Args:
        state: Current training state
        batch: Input batch
        dtype: Computation dtype

    Returns:
        Metrics dictionary
    """
    # No dropout during eval
    dummy_rng = random.PRNGKey(0)
    loss, metrics = compute_loss(
        state.params,
        state.apply_fn,
        batch,
        dummy_rng,
        dtype,
    )
    return metrics


def create_sharded_train_step(
    mesh: Mesh,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Callable:
    """Create a sharded training step for distributed training.

    Args:
        mesh: Device mesh
        dtype: Computation dtype

    Returns:
        JIT-compiled, sharded train step function
    """
    # Define partition specs
    data_spec = P("data", None)  # Batch dimension sharded across data axis
    param_spec = P(None, "model")  # Params can be model-parallel

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), data_spec),  # state is replicated, batch is sharded
        out_specs=(P(), P()),  # outputs are replicated
        check_rep=False,
    )
    def sharded_step(state, batch):
        return train_step(state, batch, dtype)

    return jax.jit(sharded_step)


def create_sharded_eval_step(
    mesh: Mesh,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Callable:
    """Create a sharded evaluation step.

    Args:
        mesh: Device mesh
        dtype: Computation dtype

    Returns:
        JIT-compiled, sharded eval step function
    """
    data_spec = P("data", None)

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), data_spec),
        out_specs=P(),
        check_rep=False,
    )
    def sharded_step(state, batch):
        return eval_step(state, batch, dtype)

    return jax.jit(sharded_step)


class CheckpointManager:
    """Manages model checkpointing with Orbax.

    Handles saving and restoring of training state, including:
    - Model parameters
    - Optimizer state
    - Training step
    - RNG state
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 3,
        save_interval_steps: int = 1000,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to retain
            save_interval_steps: Save checkpoint every N steps
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_interval_steps = save_interval_steps

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create Orbax checkpoint manager
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
        )

        self.manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=options,
        )

        # Checkpointer for TrainState
        self.checkpointer = ocp.PyTreeCheckpointer()

    def save(
        self,
        state: TrainStateWithMetrics,
        step: int,
        metrics: Optional[Metrics] = None,
    ) -> None:
        """Save a checkpoint.

        Args:
            state: Training state to save
            step: Current training step
            metrics: Optional metrics to store
        """
        ckpt = {
            "params": state.params,
            "opt_state": state.opt_state,
            "step": state.step,
            "dropout_rng": state.dropout_rng,
        }

        if metrics:
            ckpt["metrics"] = metrics

        self.manager.save(
            step,
            args=ocp.args.PyTreeSave(ckpt),
        )
        print(f"[mud-puppy] Checkpoint saved at step {step}")

    def restore(
        self,
        state: TrainStateWithMetrics,
        step: Optional[int] = None,
    ) -> Tuple[TrainStateWithMetrics, int]:
        """Restore from checkpoint.

        Args:
            state: Template state with correct structure
            step: Specific step to restore, or None for latest

        Returns:
            Tuple of (restored state, restored step)
        """
        if step is None:
            step = self.manager.latest_step()

        if step is None:
            print("[mud-puppy] No checkpoint found, starting fresh")
            return state, 0

        # Create restore args with state structure
        restore_args = ocp.args.PyTreeRestore(
            item={
                "params": state.params,
                "opt_state": state.opt_state,
                "step": state.step,
                "dropout_rng": state.dropout_rng,
            }
        )

        ckpt = self.manager.restore(step, args=restore_args)

        restored_state = state.replace(
            params=ckpt["params"],
            opt_state=ckpt["opt_state"],
            step=ckpt["step"],
            dropout_rng=ckpt["dropout_rng"],
        )

        print(f"[mud-puppy] Restored checkpoint from step {step}")
        return restored_state, step

    def should_save(self, step: int) -> bool:
        """Check if we should save at this step."""
        return step % self.save_interval_steps == 0


class DataPipeline:
    """Data loading and preprocessing pipeline.

    Handles:
    - Loading datasets
    - Tokenization
    - Batching
    - Conversion to JAX arrays
    """

    def __init__(
        self,
        config: TrainingConfig,
        tokenizer: Any,
    ):
        """Initialize data pipeline.

        Args:
            config: Training configuration
            tokenizer: Tokenizer for preprocessing
        """
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = min(
            getattr(tokenizer, "model_max_length", 2048),
            2048
        )

    def load_and_preprocess(self) -> Any:
        """Load and preprocess the training dataset.

        Returns:
            Preprocessed dataset
        """
        # Load dataset
        dataset = load_dataset("json", data_files=self.config.dataset_path, split="train")
        columns = dataset.column_names

        def tokenize(examples):
            texts = []

            for i in range(len(examples[columns[0]])):
                if "messages" in columns:
                    messages = examples["messages"][i]
                    if (
                        self.config.use_chat_template
                        and hasattr(self.tokenizer, "apply_chat_template")
                        and self.tokenizer.chat_template is not None
                    ):
                        try:
                            text = self.tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=False
                            )
                        except Exception:
                            text = "\n".join(
                                f"{m.get('role', 'user')}: {m.get('content', '')}"
                                for m in messages
                            )
                    else:
                        text = "\n".join(
                            f"{m.get('role', 'user')}: {m.get('content', '')}"
                            for m in messages
                        )
                elif "text" in columns:
                    text = examples["text"][i]
                elif "input" in columns and "output" in columns:
                    text = f"{examples['input'][i]}\n{examples['output'][i]}"
                elif "prompt" in columns and "completion" in columns:
                    text = f"{examples['prompt'][i]}{examples['completion'][i]}"
                elif "instruction" in columns:
                    text = examples["instruction"][i]
                    if "response" in columns:
                        text = f"{text}\n{examples['response'][i]}"
                else:
                    text = str(examples[columns[0]][i])

                texts.append(text)

            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="np",
            )

            # Labels are the same as input_ids for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        dataset = dataset.map(
            tokenize,
            batched=True,
            num_proc=self.config.preprocessing_workers,
            remove_columns=columns,
            desc="Tokenizing dataset",
        )

        return dataset

    def create_batches(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = True,
        rng: Optional[PRNGKey] = None,
    ) -> Iterator[Batch]:
        """Create batches from dataset.

        Args:
            dataset: Preprocessed dataset
            batch_size: Batch size
            shuffle: Whether to shuffle
            rng: Random key for shuffling

        Yields:
            Batches as JAX arrays
        """
        num_examples = len(dataset)
        indices = list(range(num_examples))

        if shuffle and rng is not None:
            # Use JAX random for consistent shuffling across devices
            perm = jax.random.permutation(rng, num_examples)
            indices = [int(i) for i in perm]

        for start_idx in range(0, num_examples, batch_size):
            end_idx = min(start_idx + batch_size, num_examples)
            batch_indices = indices[start_idx:end_idx]

            # Handle incomplete batches
            if len(batch_indices) < batch_size:
                # Pad with repeated samples
                while len(batch_indices) < batch_size:
                    batch_indices.extend(indices[:batch_size - len(batch_indices)])
                batch_indices = batch_indices[:batch_size]

            batch = dataset.select(batch_indices)

            yield {
                "input_ids": jnp.array(batch["input_ids"]),
                "attention_mask": jnp.array(batch["attention_mask"]),
                "labels": jnp.array(batch["labels"]),
            }


class ProgressLogger:
    """Logging and progress tracking."""

    def __init__(
        self,
        total_steps: int,
        log_interval: int = 10,
        log_with: str = "none",
    ):
        """Initialize logger.

        Args:
            total_steps: Total training steps
            log_interval: Log every N steps
            log_with: Logging backend (none, tensorboard, wandb)
        """
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.log_with = log_with

        self.start_time = None
        self.step_times = []

        # Initialize logging backend
        if log_with == "tensorboard":
            try:
                from flax.metrics import tensorboard
                self.writer = tensorboard.SummaryWriter("./logs")
            except ImportError:
                print("[mud-puppy] tensorboard not available, falling back to stdout")
                self.log_with = "none"
        elif log_with == "wandb":
            try:
                import wandb
                if not wandb.run:
                    wandb.init(project="mud-puppy")
            except ImportError:
                print("[mud-puppy] wandb not available, falling back to stdout")
                self.log_with = "none"

    def start(self) -> None:
        """Mark training start."""
        self.start_time = time.time()

    def log_step(self, step: int, metrics: Metrics, lr: float) -> None:
        """Log metrics for a training step.

        Args:
            step: Current step
            metrics: Step metrics
            lr: Current learning rate
        """
        if step % self.log_interval != 0:
            return

        elapsed = time.time() - self.start_time
        steps_per_sec = step / elapsed if elapsed > 0 else 0

        loss = float(metrics.get("loss", 0))
        grad_norm = float(metrics.get("grad_norm", 0))
        num_tokens = int(metrics.get("num_tokens", 0))

        # Console output
        print(
            f"[mud-puppy] step {step}/{self.total_steps} | "
            f"loss={loss:.4f} | grad_norm={grad_norm:.4f} | "
            f"lr={lr:.2e} | {steps_per_sec:.2f} steps/s"
        )

        # Backend logging
        if self.log_with == "tensorboard":
            self.writer.scalar("train/loss", loss, step)
            self.writer.scalar("train/grad_norm", grad_norm, step)
            self.writer.scalar("train/learning_rate", lr, step)
            self.writer.scalar("train/tokens", num_tokens, step)
        elif self.log_with == "wandb":
            import wandb
            wandb.log({
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/learning_rate": lr,
                "train/tokens": num_tokens,
                "train/step": step,
            })

    def log_eval(self, step: int, metrics: Metrics) -> None:
        """Log evaluation metrics.

        Args:
            step: Current step
            metrics: Evaluation metrics
        """
        loss = float(metrics.get("loss", 0))

        print(f"[mud-puppy] eval @ step {step} | loss={loss:.4f}")

        if self.log_with == "tensorboard":
            self.writer.scalar("eval/loss", loss, step)
        elif self.log_with == "wandb":
            import wandb
            wandb.log({"eval/loss": loss, "eval/step": step})

    def finish(self) -> None:
        """Finish logging."""
        if self.log_with == "tensorboard":
            self.writer.close()
        elif self.log_with == "wandb":
            import wandb
            wandb.finish()


def load_model(config: TrainingConfig) -> Tuple[Any, Any, Any]:
    """Load Flax model and tokenizer.

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, params, tokenizer)
    """
    print(f"[mud-puppy] Loading model: {config.model_name_or_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine dtype
    dtype = get_dtype(config.precision)

    # Load Flax model
    model = FlaxAutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    )

    # Extract params and module
    params = model.params
    module = model.module

    print(f"[mud-puppy] Model loaded with dtype={dtype}")

    return module, params, tokenizer


def run_training(config: TrainingConfig) -> None:
    """Main JAX training entry point.

    This function orchestrates the complete training pipeline:
    1. Load model, params, and tokenizer
    2. Set up optimizer and training state
    3. Create device mesh for distributed training
    4. Set up checkpointing
    5. Run training loop
    6. Save final model

    Args:
        config: Training configuration
    """
    print(f"[mud-puppy] Starting JAX training")
    print(f"[mud-puppy] Model: {config.model_name_or_path}")
    print(f"[mud-puppy] Dataset: {config.dataset_path}")
    print(f"[mud-puppy] Output: {config.output_dir}")
    print(f"[mud-puppy] Precision: {config.precision}")
    print(f"[mud-puppy] Devices: {jax.device_count()}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize RNG
    rng = random.PRNGKey(42)
    rng, init_rng, data_rng = random.split(rng, 3)

    # Load model and tokenizer
    module, params, tokenizer = load_model(config)

    # Set up data pipeline
    data_pipeline = DataPipeline(config, tokenizer)
    dataset = data_pipeline.load_and_preprocess()
    print(f"[mud-puppy] Dataset loaded: {len(dataset)} examples")

    # Calculate training steps
    steps_per_epoch = len(dataset) // config.batch_size
    num_train_steps = steps_per_epoch * config.num_epochs
    effective_batch_size = config.batch_size * config.gradient_accumulation

    print(f"[mud-puppy] Steps per epoch: {steps_per_epoch}")
    print(f"[mud-puppy] Total steps: {num_train_steps}")
    print(f"[mud-puppy] Effective batch size: {effective_batch_size}")

    # Create training state
    state = create_train_state(
        module,
        params,
        config,
        num_train_steps,
        init_rng,
    )

    # Create device mesh for distributed training
    mesh = create_mesh()
    print(f"[mud-puppy] Device mesh: {mesh.shape}")

    # Set up dtype
    dtype = get_dtype(config.precision)

    # JIT compile training and eval steps
    if config.distributed and jax.device_count() > 1:
        p_train_step = create_sharded_train_step(mesh, dtype)
        p_eval_step = create_sharded_eval_step(mesh, dtype)
    else:
        # Single device: just JIT
        @jax.jit
        def p_train_step(state, batch):
            return train_step(state, batch, dtype)

        @jax.jit
        def p_eval_step(state, batch):
            return eval_step(state, batch, dtype)

    # Set up checkpointing
    ckpt_manager = CheckpointManager(
        checkpoint_dir=os.path.join(config.output_dir, "checkpoints"),
        max_to_keep=3,
        save_interval_steps=steps_per_epoch,  # Save every epoch
    )

    # Resume from checkpoint if requested
    start_step = 0
    if config.resume:
        state, start_step = ckpt_manager.restore(state)

    # Set up logging
    logger = ProgressLogger(
        total_steps=num_train_steps,
        log_interval=10,
        log_with=config.log_with,
    )

    # Split dataset for validation if early stopping
    train_dataset = dataset
    eval_dataset = None
    if config.early_stopping_patience > 0:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    # Training loop
    print("[mud-puppy] Starting training loop...")
    logger.start()

    global_step = start_step
    best_eval_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.num_epochs):
        print(f"[mud-puppy] Epoch {epoch + 1}/{config.num_epochs}")

        # Create epoch data iterator
        epoch_rng = random.fold_in(data_rng, epoch)
        train_batches = data_pipeline.create_batches(
            train_dataset,
            config.batch_size,
            shuffle=True,
            rng=epoch_rng,
        )

        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_batches:
            if global_step < start_step:
                global_step += 1
                continue

            # Training step
            state, metrics = p_train_step(state, batch)

            # Get current learning rate from optimizer state
            # Note: optax stores schedule info in opt_state
            lr = config.learning_rate  # Simplified; could extract from schedule

            # Log metrics
            logger.log_step(global_step, metrics, lr)

            epoch_loss += float(metrics["loss"])
            epoch_steps += 1
            global_step += 1

            # Checkpoint
            if ckpt_manager.should_save(global_step):
                ckpt_manager.save(state, global_step, metrics)

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"[mud-puppy] Epoch {epoch + 1} complete | avg_loss={avg_epoch_loss:.4f}")

        # Validation
        if eval_dataset is not None:
            eval_batches = data_pipeline.create_batches(
                eval_dataset,
                config.batch_size,
                shuffle=False,
            )

            eval_loss = 0.0
            eval_steps = 0

            for batch in eval_batches:
                metrics = p_eval_step(state, batch)
                eval_loss += float(metrics["loss"])
                eval_steps += 1

            avg_eval_loss = eval_loss / max(eval_steps, 1)
            logger.log_eval(global_step, {"loss": avg_eval_loss})

            # Early stopping check
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                patience_counter = 0
                # Save best model
                ckpt_manager.save(state, global_step, {"eval_loss": avg_eval_loss})
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"[mud-puppy] Early stopping triggered after {epoch + 1} epochs")
                    break

        # Save epoch checkpoint
        ckpt_manager.save(state, global_step, {"epoch": epoch + 1})

    # Save final model
    print("[mud-puppy] Saving final model...")

    # Save params as Flax checkpoint
    final_ckpt_path = os.path.join(config.output_dir, "final_checkpoint")
    os.makedirs(final_ckpt_path, exist_ok=True)

    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        final_ckpt_path,
        state.params,
    )

    # Save tokenizer
    tokenizer.save_pretrained(config.output_dir)

    logger.finish()
    print("[mud-puppy] JAX training complete!")


def run_validation(
    config: TrainingConfig,
    checkpoint_path: Optional[str] = None,
) -> Metrics:
    """Run validation on a trained model.

    Args:
        config: Training configuration
        checkpoint_path: Path to checkpoint, or None for latest

    Returns:
        Validation metrics
    """
    # Load model
    module, params, tokenizer = load_model(config)

    # Set up data pipeline
    data_pipeline = DataPipeline(config, tokenizer)
    dataset = data_pipeline.load_and_preprocess()

    # Create dummy state for eval
    dummy_state = create_train_state(
        module,
        params,
        config,
        num_train_steps=1,
        rng=random.PRNGKey(0),
    )

    # Load checkpoint if provided
    if checkpoint_path:
        ckpt_manager = CheckpointManager(checkpoint_path)
        dummy_state, _ = ckpt_manager.restore(dummy_state)

    dtype = get_dtype(config.precision)

    @jax.jit
    def p_eval_step(state, batch):
        return eval_step(state, batch, dtype)

    # Run evaluation
    eval_batches = data_pipeline.create_batches(
        dataset,
        config.batch_size,
        shuffle=False,
    )

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for batch in eval_batches:
        metrics = p_eval_step(dummy_state, batch)
        total_loss += float(metrics["loss"])
        total_tokens += int(metrics["num_tokens"])
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = jnp.exp(avg_loss)

    results = {
        "loss": avg_loss,
        "perplexity": float(perplexity),
        "total_tokens": total_tokens,
    }

    print(f"[mud-puppy] Validation: loss={avg_loss:.4f}, perplexity={perplexity:.4f}")

    return results


# Convenience function to match PyTorch trainer API
train = run_training
validate = run_validation
