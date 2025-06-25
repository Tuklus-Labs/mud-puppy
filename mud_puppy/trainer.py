import os
from typing import Optional, Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from .config import TrainingConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(config: TrainingConfig):
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if config.precision == "fp8":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("FP8 precision is not supported in this PyTorch build")
        model.to(torch.float8_e4m3fn)

    return model, tokenizer


def prepare_lora(model, config: TrainingConfig):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise RuntimeError("peft is required for LoRA training") from e

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    return model


def create_training_args(config: TrainingConfig) -> TrainingArguments:
    precision = config.precision
    fp16 = precision == "fp16"
    bf16 = precision == "bf16"
    return TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=config.use_gradient_checkpointing,
        optim="adamw_torch",
    )


class FP8Trainer(Trainer):
    """Trainer subclass enabling FP8 autocast."""

    def training_step(self, model, inputs):
        with torch.autocast("cuda", dtype=torch.float8_e4m3fn):
            return super().training_step(model, inputs)



def configure_rocm():
    """Apply ROCm-specific environment settings."""
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:128")
    torch.backends.cuda.matmul.allow_tf32 = True


def apply_chat_template(examples: Dict[str, List[Dict[str, str]]], tokenizer):
    text = tokenizer.apply_chat_template(
        examples["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


ddef load_and_preprocess_dataset(config: TrainingConfig, tokenizer):
    """Load a JSONL chat dataset and tokenize it."""
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=config.dataset_path)["train"]

    # Apply chat template to get plain text
    dataset = dataset.map(
        lambda ex: apply_chat_template(ex, tokenizer),
        remove_columns=dataset.column_names,
    )

    # Tokenize WITHOUT return_tensors="pt" in batched mode
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding=False,  # Let DataCollator handle padding
            max_length=2048
        )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return dataset


def run_training(config: TrainingConfig):
    configure_rocm()
    device = get_device()
    model, tokenizer = load_model(config)

    if config.finetuning_method in {"lora", "qlora"}:
        model = prepare_lora(model, config)

    training_args = create_training_args(config)

    dataset = load_and_preprocess_dataset(config, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer_cls = FP8Trainer if config.precision == "fp8" else Trainer
    trainer = trainer_cls(

        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


