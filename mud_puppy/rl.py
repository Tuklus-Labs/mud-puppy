"""Reinforcement learning utilities for ROCm-first training."""

from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
)


from .config import TrainingConfig


def run_grpo_training(config: TrainingConfig):
    """Run a simplified Guided Reinforcement Policy Optimization loop."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name_or_path)

    dataset = load_dataset("json", data_files=config.dataset_path)["train"]

    ppo_config = PPOConfig(
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation,
        mini_batch_size=config.batch_size,
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        fp16=config.precision == "fp16",
        bf16=config.precision == "bf16",
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
