import os
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .config import TrainingConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(config: TrainingConfig):
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    return model, tokenizer


def prepare_lora(model, config: TrainingConfig):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise RuntimeError("peft is required for LoRA training") from e

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj"],
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


def run_training(config: TrainingConfig):
    device = get_device()
    model, tokenizer = load_model(config)

    if config.finetuning_method in {"lora", "qlora"}:
        model = prepare_lora(model, config)

    training_args = create_training_args(config)

    # Placeholder dataset loading
    from datasets import load_dataset

    dataset = load_dataset(config.dataset_path)
    if "train" not in dataset:
        raise ValueError("Dataset must contain a train split")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )

    trainer.train()


