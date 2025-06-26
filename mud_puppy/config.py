from dataclasses import dataclass, field
from typing import Optional

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
    qat: bool = False
    grpo: bool = False
    preference: Optional[str] = None
    multimodal: bool = False
    reward_modeling: bool = False

    def __post_init__(self):
        supported = {
            "full", "lora", "qlora", "gptq", "qat",
            "preference", "rl", "multimodal", "rm"
        }
        if self.finetuning_method not in supported:
            raise ValueError(
                f"Unsupported finetuning method: {self.finetuning_method}"
            )

        if self.precision not in {"fp16", "bf16", "fp8"}:
            raise ValueError(f"Unsupported precision: {self.precision}")

