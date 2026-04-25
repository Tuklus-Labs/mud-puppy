from dataclasses import dataclass, field
import math
import os
from typing import Optional, List
import torch


def _env_int(name: str, default: int) -> int:
    """int(env) with sane handling of unset/empty values."""
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


@dataclass
class TrainingConfig:
    """Top-level configuration for a mud-puppy training run.

    This dataclass is used by both the CLI entrypoints and the Python API.
    It performs basic validation of paths and incompatible options in
    ``__post_init__`` so that most user errors are caught early.
    """

    # Core inputs
    model_name_or_path: str
    dataset_path: str
    output_dir: str

    # High-level method selection
    finetuning_method: str = "full"  # full, lora, qlora, gptq, qat, preference, rl, multimodal, rm, prm, embedding
    precision: str = "bf16"  # fp16, bf16, fp8, fp32

    # Optimization hyperparameters
    batch_size: int = 1
    gradient_accumulation: int = 1
    learning_rate: float = 2e-5
    num_epochs: int = 1
    use_gradient_checkpointing: bool = True

    # LoRA / QLoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
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

    # Quantization backend for QLoRA
    quant_backend: str = "int4"  # int4 (bnb_rocm) or mxfp4 (block-scaled)
    quant_block_size: int = 32   # Block size for mxfp4 quantization

    # GPTQ hyperparameters (used when finetuning_method == "gptq")
    gptq_group_size: int = 128
    gptq_actorder: bool = True
    gptq_damp_percent: float = 0.01

    # Method-specific switches (currently mostly informational)
    qat: bool = False
    grpo: bool = False
    preference: Optional[str] = None
    multimodal: bool = False
    reward_modeling: bool = False

    # Data / preprocessing
    max_seq_length: int = 0  # 0 = use model default (capped at 2048)
    response_only: bool = True  # Mask prompt tokens in loss (train on response only)
    use_chat_template: bool = True
    trust_remote_code: bool = False
    dataloader_workers: int = 0
    preprocessing_workers: int = 1

    # Sequence packing
    pack_sequences: bool = False  # Greedy bin-packing collator (v0.4, default-on v0.5)

    # System / runtime behavior
    compile: bool = False
    compile_mode: str = "reduce-overhead"  # torch.compile mode: reduce-overhead, default, max-autotune
    resume: bool = False
    log_with: str = "none"  # none, tensorboard, wandb
    tokens_per_batch: int = 0
    lr_scheduler: str = "linear"
    early_stopping_patience: int = 0
    device_map: str = "auto"  # auto, pipeline, or HF accelerate-style map
    stream: bool = False
    prefetch_layers: int = 2  # Number of layers to keep resident in GPU ring
    zero_offload: bool = False
    max_grad_norm: float = 1.0
    distributed: bool = False
    local_rank: int = _env_int("LOCAL_RANK", 0)

    # --- FSDP / multi-GPU ----------------------------------------------------
    #
    # FSDP is the supported path for multi-GPU training on MI300 nodes.
    # DDP still works (``distributed=True, fsdp=False``) but won't fit a
    # 70B model on 8x192GB without it.
    #
    # When ``fsdp_mode`` is non-empty, ``create_training_args`` forwards
    # the matching HuggingFace FSDP options. The launcher sets
    # LOCAL_RANK / WORLD_SIZE from torchrun; we infer ``distributed``
    # automatically when WORLD_SIZE > 1, but an explicit flag always wins.
    # "full_shard" / "shard_grad_op" / "no_shard" all work on a single node.
    # "hybrid_shard" REQUIRES multi-node (set --nnodes > 1 via launcher); a
    # single-node hybrid_shard run is a misconfiguration and only emits a
    # warning because the launcher owns the topology.
    fsdp_mode: str = ""                 # "", "full_shard", "shard_grad_op", "no_shard", "hybrid_shard"
    fsdp_cpu_offload: bool = False      # offload sharded params to CPU (MI300 rarely needs this)
    fsdp_activation_checkpointing: bool = False
    fsdp_min_num_params: int = 1_000_000  # auto-wrap threshold for transformer blocks
    fsdp_transformer_layer_cls: str = ""  # e.g. "LlamaDecoderLayer"; empty = size-based wrap
    distributed_backend: str = "nccl"     # "nccl"(=rccl on ROCm), "gloo" for CPU tests

    # LoRA merge behavior
    merge_lora: bool = False
    merge_precision: str = "bf16"  # fp16, bf16, fp32

    # Monitor (web dashboard via WebSocket)
    monitor: bool = False
    monitor_port: int = 5980

    def _validate_paths(self) -> None:
        """Validate that input paths exist where required."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

    def _validate_combinations(self) -> None:
        """Validate combinations of options that are known to be invalid."""
        if self.tokens_per_batch < 0:
            raise ValueError("tokens_per_batch must be non-negative")
        if self.max_seq_length < 0:
            raise ValueError("max_seq_length must be non-negative")
        if self.prefetch_layers < 1:
            raise ValueError(
                f"prefetch_layers must be >= 1, got {self.prefetch_layers}"
            )
        # Validate fsdp_mode enum FIRST so we fail fast on a bad value.
        if self.fsdp_mode and self.fsdp_mode not in {
            "full_shard", "shard_grad_op", "no_shard", "hybrid_shard",
        }:
            raise ValueError(
                f"Unsupported fsdp_mode: {self.fsdp_mode!r}. "
                "Must be one of: full_shard, shard_grad_op, no_shard, hybrid_shard"
            )
        # Auto-promote BEFORE the device-count check so users who pass only
        # ``--fsdp full_shard`` (without ``--distributed``) don't bypass the
        # GPU-count guard and then crash inside NCCL.
        if self.fsdp_mode and not self.distributed:
            # FSDP implies distributed. Flip it on so users don't have to
            # pass both --fsdp full_shard --distributed every time.
            self.distributed = True
        if self.fsdp_mode == "hybrid_shard":
            # hybrid_shard only makes sense with >1 node. TrainingConfig has
            # no nnodes field (the launcher controls topology), so we can't
            # hard-check it here. Warn so launcher users who legitimately
            # want hybrid_shard aren't blocked.
            import logging
            logging.getLogger(__name__).warning(
                "fsdp_mode='hybrid_shard' is intended for multi-node runs; "
                "make sure the launcher sets --nnodes > 1."
            )
        if self.distributed and torch.cuda.device_count() < 2:
            # Note: torch.distributed on ROCm still reports as CUDA in PyTorch.
            # Allow skipping this check under pytest / torchrun dry-run where
            # CUDA isn't visible but the user is legitimately constructing
            # a config for a remote multi-GPU run.
            if os.environ.get("MUD_PUPPY_SKIP_GPU_COUNT_CHECK") != "1":
                raise ValueError("Distributed training requires at least 2 GPUs")
        if self.distributed_backend not in {"nccl", "gloo", "mpi"}:
            raise ValueError(
                f"Unsupported distributed_backend: {self.distributed_backend!r}"
            )

    def __post_init__(self) -> None:
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
            "embedding",
        }
        if self.finetuning_method not in supported:
            raise ValueError(f"Unsupported finetuning method: {self.finetuning_method}")

        if self.precision not in {"fp16", "bf16", "fp8", "fp32"}:
            raise ValueError(f"Unsupported precision: {self.precision}")

        schedulers = {"linear", "cosine", "cosine_with_restarts", "polynomial"}
        if self.lr_scheduler not in schedulers:
            raise ValueError(f"Unsupported lr_scheduler: {self.lr_scheduler}")

        if self.merge_precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError(
                f"Unsupported merge_precision: {self.merge_precision}"
            )

        if self.log_with not in {"none", "tensorboard", "wandb"}:
            raise ValueError(f"Unsupported logging backend: {self.log_with}")
        if not math.isfinite(self.lora_dropout):
            raise ValueError(f"lora_dropout must be finite, got {self.lora_dropout}")
        if not 0.0 <= self.lora_dropout <= 1.0:
            raise ValueError("lora_dropout must be between 0.0 and 1.0")

        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError(
                f"Unsupported compile_mode: {self.compile_mode}. "
                "Must be one of: default, reduce-overhead, max-autotune"
            )

        # G1: numeric sanity checks -- catch obvious mis-configs before any
        # GPU memory is allocated.
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be > 0, got {self.num_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not math.isfinite(self.learning_rate):
            raise ValueError(f"learning_rate must be finite, got {self.learning_rate}")
        if self.finetuning_method in {"lora", "qlora"} and self.lora_r <= 0:
            raise ValueError(
                f"lora_r must be > 0 for method={self.finetuning_method!r}, "
                f"got {self.lora_r}"
            )
        if not 0 < self.gptq_damp_percent < 1:
            raise ValueError(
                f"gptq_damp_percent must be in (0, 1), got {self.gptq_damp_percent}"
            )
        if self.max_seq_length < 0:
            raise ValueError(
                f"max_seq_length must be >= 0 (0 means use model default), "
                f"got {self.max_seq_length}"
            )
        if not math.isfinite(self.max_grad_norm):
            raise ValueError(
                f"max_grad_norm must be finite, got {self.max_grad_norm}"
            )
        if self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be > 0 (use the default 1.0 to enable "
                f"standard gradient clipping); got {self.max_grad_norm}"
            )

        self._validate_paths()
        self._validate_combinations()
