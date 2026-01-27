from dataclasses import dataclass, field
import os
from typing import Optional, List, Tuple, Any, Dict, Callable

# JAX imports - handle gracefully if not available
try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec, NamedSharding
    from jax.experimental import mesh_utils
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

# Flax imports for training state helpers
try:
    import flax
    from flax.training import train_state
    from flax import linen as nn
    import optax
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    train_state = None
    optax = None

# PyTorch import for backward compatibility checks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class TrainingConfig:
    """Top-level configuration for a mud-puppy training run with JAX/Flax support.

    This dataclass extends the original PyTorch-based TrainingConfig to support
    JAX/Flax training while maintaining backward compatibility with PyTorch
    workflows.

    JAX-specific features:
        - TPU/GPU mesh sharding configuration
        - pjit parallelism support
        - Gradient checkpointing policies for Flax
        - Precision control (float32, bfloat16)
        - Helper methods for creating Flax TrainState

    Backward compatibility:
        - All original PyTorch settings are preserved
        - Can be used with either backend based on `backend` field
        - Validation adapts based on selected backend
    """

    # Core inputs
    model_name_or_path: str
    dataset_path: str
    output_dir: str

    # Backend selection (new)
    backend: str = "pytorch"  # pytorch, jax

    # Internal flag to skip validation (used by copy operations)
    _skip_validation: bool = field(default=False, repr=False)

    # High-level method selection
    finetuning_method: str = "full"  # full, lora, qlora, gptq, qat, preference, rl, multimodal, rm, prm
    precision: str = "bf16"  # fp16, bf16, fp8 (PyTorch) / float32, bfloat16 (JAX)

    # Optimization hyperparameters
    batch_size: int = 1
    gradient_accumulation: int = 1
    learning_rate: float = 2e-5
    num_epochs: int = 1
    use_gradient_checkpointing: bool = True

    # LoRA / QLoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Method-specific switches
    qat: bool = False
    grpo: bool = False
    preference: Optional[str] = None
    multimodal: bool = False
    reward_modeling: bool = False

    # Data / preprocessing
    use_chat_template: bool = True
    trust_remote_code: bool = False
    dataloader_workers: int = 0
    preprocessing_workers: int = 1

    # System / runtime behavior (PyTorch-focused, still valid)
    compile: bool = False
    resume: bool = False
    log_with: str = "none"  # none, tensorboard, wandb
    tokens_per_batch: int = 0
    lr_scheduler: str = "linear"
    early_stopping_patience: int = 0
    device_map: str = "auto"  # auto, pipeline, or HF accelerate-style map
    stream: bool = False
    zero_offload: bool = False
    max_grad_norm: float = 1.0
    distributed: bool = False
    local_rank: int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", 0)))

    # LoRA merge behavior
    merge_lora: bool = False
    merge_precision: str = "bf16"  # fp16, bf16, fp32

    # ==================== JAX-Specific Options ====================

    # TPU/GPU mesh configuration
    tpu_mesh_shape: Tuple[int, ...] = field(default_factory=lambda: (1, 1))
    mesh_axis_names: Tuple[str, ...] = field(default_factory=lambda: ("data", "model"))

    # Parallelism
    use_pjit: bool = False
    data_parallel_axis: str = "data"
    model_parallel_axis: str = "model"
    fsdp_enabled: bool = False  # Fully Sharded Data Parallel for JAX

    # JAX precision (separate from PyTorch precision for clarity)
    jax_precision: str = "bfloat16"  # float32, bfloat16, float16

    # Gradient checkpointing policy for Flax
    gradient_checkpointing_policy: str = "nothing_saveable"
    # Options: nothing_saveable, everything_saveable, checkpoint_dots,
    #          checkpoint_dots_with_no_batch_dims, custom

    # JAX-specific optimization
    use_scan: bool = True  # Use lax.scan for layer iteration
    donation_argnums: Tuple[int, ...] = field(default_factory=tuple)
    jit_compile: bool = True

    # XLA flags
    xla_flags: Optional[str] = None

    # Sharding specifications for model parameters
    param_sharding: Optional[Dict[str, Any]] = None

    def _validate_paths(self) -> None:
        """Validate that input paths exist where required."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

    def _validate_combinations(self) -> None:
        """Validate combinations of options that are known to be invalid."""
        if self.finetuning_method == "qlora" and self.stream:
            raise ValueError("Streaming is not supported with QLoRA")
        if self.tokens_per_batch < 0:
            raise ValueError("tokens_per_batch must be non-negative")

        # PyTorch-specific validation
        if self.backend == "pytorch":
            if TORCH_AVAILABLE and self.distributed and torch.cuda.device_count() < 2:
                raise ValueError("Distributed training requires at least 2 GPUs")

    def _validate_jax_options(self) -> None:
        """Validate JAX-specific configuration options."""
        if self.backend != "jax":
            return

        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX backend selected but JAX is not installed. "
                "Install with: pip install jax jaxlib"
            )

        if not FLAX_AVAILABLE:
            raise ImportError(
                "JAX backend requires Flax. Install with: pip install flax optax"
            )

        # Validate JAX precision
        valid_jax_precisions = {"float32", "bfloat16", "float16"}
        if self.jax_precision not in valid_jax_precisions:
            raise ValueError(
                f"Unsupported jax_precision: {self.jax_precision}. "
                f"Must be one of: {valid_jax_precisions}"
            )

        # Validate mesh shape
        if len(self.tpu_mesh_shape) != len(self.mesh_axis_names):
            raise ValueError(
                f"tpu_mesh_shape length ({len(self.tpu_mesh_shape)}) must match "
                f"mesh_axis_names length ({len(self.mesh_axis_names)})"
            )

        # Validate mesh shape product against available devices
        mesh_size = 1
        for dim in self.tpu_mesh_shape:
            if dim < 1:
                raise ValueError(f"Mesh dimensions must be >= 1, got {dim}")
            mesh_size *= dim

        device_count = jax.device_count()
        if mesh_size > device_count:
            raise ValueError(
                f"Mesh shape {self.tpu_mesh_shape} requires {mesh_size} devices "
                f"but only {device_count} available"
            )

        # Validate gradient checkpointing policy
        valid_policies = {
            "nothing_saveable",
            "everything_saveable",
            "checkpoint_dots",
            "checkpoint_dots_with_no_batch_dims",
            "custom",
        }
        if self.gradient_checkpointing_policy not in valid_policies:
            raise ValueError(
                f"Unsupported gradient_checkpointing_policy: {self.gradient_checkpointing_policy}. "
                f"Must be one of: {valid_policies}"
            )

        # Validate parallelism axes exist in mesh
        if self.use_pjit:
            if self.data_parallel_axis not in self.mesh_axis_names:
                raise ValueError(
                    f"data_parallel_axis '{self.data_parallel_axis}' not in mesh_axis_names"
                )
            if self.model_parallel_axis not in self.mesh_axis_names:
                raise ValueError(
                    f"model_parallel_axis '{self.model_parallel_axis}' not in mesh_axis_names"
                )

        # Warn about unsupported finetuning methods in JAX
        jax_unsupported = {"qlora", "gptq", "fp8"}
        if self.finetuning_method in jax_unsupported:
            raise ValueError(
                f"Finetuning method '{self.finetuning_method}' is not yet supported "
                f"with JAX backend. Supported methods: full, lora, preference, rl"
            )

    def __post_init__(self) -> None:
        # Skip validation if flag is set (used internally for copy operations)
        if self._skip_validation:
            # Reset the flag after skipping
            object.__setattr__(self, '_skip_validation', False)
            return

        # Validate backend
        if self.backend not in {"pytorch", "jax"}:
            raise ValueError(f"Unsupported backend: {self.backend}. Must be 'pytorch' or 'jax'")

        supported = {
            "full",
            "lora",
            "qlora",
            "gptq",
            "qat",
            "preference",
            "rl",
            "multimodal",
            "rm",
            "prm",
        }
        if self.finetuning_method not in supported:
            raise ValueError(f"Unsupported finetuning method: {self.finetuning_method}")

        # Precision validation depends on backend
        if self.backend == "pytorch":
            if self.precision not in {"fp16", "bf16", "fp8"}:
                raise ValueError(f"Unsupported precision for PyTorch: {self.precision}")
        # JAX precision is validated separately in _validate_jax_options

        schedulers = {"linear", "cosine", "cosine_with_restarts", "polynomial"}
        if self.lr_scheduler not in schedulers:
            raise ValueError(f"Unsupported lr_scheduler: {self.lr_scheduler}")

        if self.merge_precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError(f"Unsupported merge_precision: {self.merge_precision}")

        if self.log_with not in {"none", "tensorboard", "wandb"}:
            raise ValueError(f"Unsupported logging backend: {self.log_with}")

        self._validate_paths()
        self._validate_combinations()
        self._validate_jax_options()

    # ==================== JAX Helper Methods ====================

    def get_jax_dtype(self) -> Any:
        """Get the JAX dtype corresponding to the configured precision."""
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not available")

        dtype_map = {
            "float32": jnp.float32,
            "bfloat16": jnp.bfloat16,
            "float16": jnp.float16,
        }
        return dtype_map[self.jax_precision]

    def create_mesh(self) -> "Mesh":
        """Create a JAX device mesh based on configuration.

        Returns:
            jax.sharding.Mesh configured for the specified topology
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not available")

        devices = mesh_utils.create_device_mesh(self.tpu_mesh_shape)
        return Mesh(devices, axis_names=self.mesh_axis_names)

    def get_partition_spec(
        self,
        param_name: str,
        param_shape: Tuple[int, ...],
    ) -> "PartitionSpec":
        """Get sharding specification for a parameter.

        Args:
            param_name: Name of the parameter (for custom sharding rules)
            param_shape: Shape of the parameter

        Returns:
            PartitionSpec for the parameter
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not available")

        # Check for custom sharding specification
        if self.param_sharding and param_name in self.param_sharding:
            return PartitionSpec(*self.param_sharding[param_name])

        # Default sharding heuristics
        if not self.use_pjit:
            return PartitionSpec()  # Replicated

        # Shard large matrices along model axis
        if len(param_shape) >= 2:
            # For 2D weights, shard along the larger dimension
            if param_shape[-1] >= param_shape[-2]:
                return PartitionSpec(None, self.model_parallel_axis)
            else:
                return PartitionSpec(self.model_parallel_axis, None)

        return PartitionSpec()  # Replicate 1D params

    def get_gradient_checkpoint_policy(self) -> Optional[Callable]:
        """Get the Flax remat policy based on configuration.

        Returns:
            A callable policy for jax.checkpoint/flax.linen.remat, or None
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not available")

        if not self.use_gradient_checkpointing:
            return None

        from jax.ad_checkpoint import checkpoint_policies

        policy_map = {
            "nothing_saveable": checkpoint_policies.nothing_saveable,
            "everything_saveable": checkpoint_policies.everything_saveable,
            "checkpoint_dots": checkpoint_policies.checkpoint_dots,
            "checkpoint_dots_with_no_batch_dims": checkpoint_policies.checkpoint_dots_with_no_batch_dims,
            "custom": None,  # User provides custom policy
        }
        return policy_map.get(self.gradient_checkpointing_policy)

    def create_optimizer(self) -> Any:
        """Create an Optax optimizer chain based on configuration.

        Returns:
            optax.GradientTransformation optimizer
        """
        if not FLAX_AVAILABLE:
            raise RuntimeError("Flax/Optax is not available")

        # Build learning rate schedule
        if self.lr_scheduler == "linear":
            schedule = optax.linear_schedule(
                init_value=self.learning_rate,
                end_value=0.0,
                transition_steps=self.num_epochs * 1000,  # Placeholder, should use actual steps
            )
        elif self.lr_scheduler == "cosine":
            schedule = optax.cosine_decay_schedule(
                init_value=self.learning_rate,
                decay_steps=self.num_epochs * 1000,
            )
        elif self.lr_scheduler == "cosine_with_restarts":
            schedule = optax.cosine_onecycle_schedule(
                transition_steps=self.num_epochs * 1000,
                peak_value=self.learning_rate,
            )
        elif self.lr_scheduler == "polynomial":
            schedule = optax.polynomial_schedule(
                init_value=self.learning_rate,
                end_value=0.0,
                power=1.0,
                transition_steps=self.num_epochs * 1000,
            )
        else:
            schedule = self.learning_rate

        # Build optimizer chain
        optimizer_chain = []

        # Gradient clipping
        if self.max_grad_norm > 0:
            optimizer_chain.append(optax.clip_by_global_norm(self.max_grad_norm))

        # AdamW optimizer with schedule
        optimizer_chain.append(optax.adamw(learning_rate=schedule))

        return optax.chain(*optimizer_chain)

    def create_train_state(
        self,
        model: Any,
        params: Dict[str, Any],
        tx: Optional[Any] = None,
    ) -> Any:
        """Create a Flax TrainState for training.

        Args:
            model: The Flax model (nn.Module)
            params: Model parameters (PyTree)
            tx: Optional Optax optimizer. If None, creates one from config.

        Returns:
            flax.training.train_state.TrainState
        """
        if not FLAX_AVAILABLE:
            raise RuntimeError("Flax is not available")

        if tx is None:
            tx = self.create_optimizer()

        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
        )

    def create_sharded_train_state(
        self,
        model: Any,
        params: Dict[str, Any],
        mesh: Optional["Mesh"] = None,
        tx: Optional[Any] = None,
    ) -> Any:
        """Create a sharded Flax TrainState for distributed training.

        Args:
            model: The Flax model (nn.Module)
            params: Model parameters (PyTree)
            mesh: Optional device mesh. If None, creates one from config.
            tx: Optional Optax optimizer. If None, creates one from config.

        Returns:
            Sharded flax.training.train_state.TrainState
        """
        if not JAX_AVAILABLE or not FLAX_AVAILABLE:
            raise RuntimeError("JAX and Flax are required for sharded training")

        if mesh is None:
            mesh = self.create_mesh()

        if tx is None:
            tx = self.create_optimizer()

        # Create base train state
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
        )

        # Apply sharding to parameters
        def get_sharding(path: Tuple[str, ...], x: Any) -> NamedSharding:
            param_name = "/".join(str(p) for p in path)
            pspec = self.get_partition_spec(param_name, x.shape)
            return NamedSharding(mesh, pspec)

        # Shard the state
        from jax.tree_util import tree_map_with_path

        param_shardings = tree_map_with_path(get_sharding, params)

        # Create sharded state using jax.device_put with shardings
        sharded_params = jax.tree_util.tree_map(
            lambda x, s: jax.device_put(x, s),
            params,
            param_shardings,
        )

        return state.replace(params=sharded_params)

    def setup_xla_flags(self) -> None:
        """Set XLA flags from configuration."""
        if self.xla_flags:
            current = os.environ.get("XLA_FLAGS", "")
            if current:
                os.environ["XLA_FLAGS"] = f"{current} {self.xla_flags}"
            else:
                os.environ["XLA_FLAGS"] = self.xla_flags

    def to_pytorch_config(self) -> "TrainingConfig":
        """Return a copy of this config with backend set to pytorch.

        Useful for converting JAX configs to PyTorch for comparison or fallback.
        Note: This skips validation to allow conversion even without PyTorch installed.
        """
        from dataclasses import replace
        return replace(self, backend="pytorch", _skip_validation=True)

    def to_jax_config(self) -> "TrainingConfig":
        """Return a copy of this config with backend set to jax.

        Also performs precision mapping from PyTorch conventions to JAX.
        Note: This skips validation to allow conversion even without JAX installed.
        The config will be validated when actually used for JAX operations.
        """
        from dataclasses import replace

        # Map PyTorch precision to JAX precision if not explicitly set
        precision_map = {
            "fp16": "float16",
            "bf16": "bfloat16",
            "fp8": "bfloat16",  # fp8 not directly supported, fallback to bf16
        }

        new_jax_precision = precision_map.get(self.precision, self.jax_precision)

        return replace(
            self,
            backend="jax",
            jax_precision=new_jax_precision,
            _skip_validation=True,
        )


# Convenience factory functions
def from_pytorch_config(
    model_name_or_path: str,
    dataset_path: str,
    output_dir: str,
    **kwargs,
) -> TrainingConfig:
    """Create a PyTorch-backend TrainingConfig."""
    return TrainingConfig(
        model_name_or_path=model_name_or_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        backend="pytorch",
        **kwargs,
    )


def from_jax_config(
    model_name_or_path: str,
    dataset_path: str,
    output_dir: str,
    tpu_mesh_shape: Tuple[int, ...] = (1, 1),
    jax_precision: str = "bfloat16",
    use_pjit: bool = False,
    **kwargs,
) -> TrainingConfig:
    """Create a JAX-backend TrainingConfig with sensible defaults."""
    return TrainingConfig(
        model_name_or_path=model_name_or_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        backend="jax",
        tpu_mesh_shape=tpu_mesh_shape,
        jax_precision=jax_precision,
        use_pjit=use_pjit,
        **kwargs,
    )
