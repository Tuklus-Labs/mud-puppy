from dataclasses import dataclass, field
import os
from typing import Optional, List
import torch


@dataclass
class TrainingConfig:
    model_name_or_path: str
    dataset_path: str
    output_dir: str
    finetuning_method: str = "full"
    precision: str = "bf16"
    batch_size: int = 1
    gradient_accumulation: int = 1
    learning_rate: float = 2e-5
    num_epochs: int = 1
    use_gradient_checkpointing: bool = True
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
    qat: bool = False
    grpo: bool = False
    preference: Optional[str] = None
    multimodal: bool = False
    reward_modeling: bool = False
    use_chat_template: bool = True
    trust_remote_code: bool = False
    dataloader_workers: int = 0
    preprocessing_workers: int = 1
    compile: bool = False
    resume: bool = False
    log_with: str = "tensorboard"
    tokens_per_batch: int = 0
    lr_scheduler: str = "linear"
    early_stopping_patience: int = 0
    device_map: str = "auto"
    stream: bool = False
    zero_offload: bool = False
    max_grad_norm: float = 1.0
    distributed: bool = False
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    merge_lora: bool = False
    merge_precision: str = "bf16"

    def _validate_paths(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

    def _validate_combinations(self):
        if self.finetuning_method == "qlora" and self.stream:
            raise ValueError("Streaming is not supported with QLoRA")
        if self.tokens_per_batch < 0:
            raise ValueError("tokens_per_batch must be non-negative")

    def __post_init__(self):
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

        if self.precision not in {"fp16", "bf16", "fp8"}:
            raise ValueError(f"Unsupported precision: {self.precision}")

        schedulers = {"linear", "cosine", "cosine_with_restarts", "polynomial"}
        if self.lr_scheduler not in schedulers:
            raise ValueError(f"Unsupported lr_scheduler: {self.lr_scheduler}")

        if self.merge_precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError(
                f"Unsupported merge_precision: {self.merge_precision}"
            )

        self._validate_paths()
        self._validate_combinations()

        if self.distributed and torch.cuda.device_count() < 2:
            raise ValueError("Distributed training requires at least 2 GPUs")
