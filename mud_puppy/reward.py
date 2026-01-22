"""Reward modeling utilities.

This module supports training scalar reward models for RM (Reward Modeling)
and PRM (Process Reward Modeling) using TRL's specialized trainers when
available, with a fallback to standard Hugging Face Trainer.

Expected dataset formats (JSON/JSONL):

- For standard RM with pairwise preferences::

    {"prompt": "...", "chosen": "...", "rejected": "..."}

- For standard RM with scalar labels::

    {"text": "response text", "label": 0.87}

- For PRM, multiple token- or step-level labels may be provided::

    {"prompt": "...", "completions": ["step1", "step2"], "labels": [1, 0]}
"""

from __future__ import annotations

from typing import Dict

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from .config import TrainingConfig

# Try to import TRL's specialized trainers
RewardTrainer = None
PRMTrainer = None
RewardConfig = None

try:
    from trl import RewardTrainer, RewardConfig
except ImportError:
    pass

try:
    from trl import PRMTrainer
except ImportError:
    pass


def _prepare_reward_dataset(path: str):
    """Prepare a reward modeling dataset."""
    dataset = load_dataset("json", data_files=path)["train"]

    columns = dataset.column_names

    # Check for pairwise format (prompt, chosen, rejected)
    if "chosen" in columns and "rejected" in columns:
        return dataset, "pairwise"

    # Check for scalar label format
    if "label" not in columns and "labels" not in columns:
        # attempt to infer label column
        possible = [c for c in columns if c not in {"text", "prompt", "input"}]
        if len(possible) == 1:
            dataset = dataset.rename_column(possible[0], "label")
        else:
            raise ValueError("Dataset must contain a label/labels column or be in pairwise format")

    if "text" not in columns:
        # Attempt a simple heuristic: use the first non-label column as text
        non_label_cols = [c for c in columns if c not in {"label", "labels"}]
        if not non_label_cols:
            raise ValueError("Dataset must contain a 'text' column or an obvious text field")
        dataset = dataset.rename_column(non_label_cols[0], "text")

    return dataset, "scalar"


def _prepare_prm_dataset(path: str):
    """Prepare a process reward modeling dataset."""
    dataset = load_dataset("json", data_files=path)["train"]

    columns = dataset.column_names

    # PRM datasets should have prompt, completions (list of steps), and labels
    if "completions" not in columns and "steps" not in columns:
        # Try to use standard scalar format as fallback
        return _prepare_reward_dataset(path)

    if "steps" in columns and "completions" not in columns:
        dataset = dataset.rename_column("steps", "completions")

    return dataset, "prm"


def train_reward_model(config: TrainingConfig):
    """Train a reward model for RM or PRM.

    This function uses TRL's specialized trainers when available:
    - RewardTrainer for standard reward modeling (RM)
    - PRMTrainer for process reward modeling (PRM)

    Falls back to a standard HF Trainer with regression head if TRL trainers
    are not available.
    """
    print(f"[mud-puppy] Starting {config.finetuning_method.upper()} training")
    print(f"[mud-puppy] Model: {config.model_name_or_path}")
    print(f"[mud-puppy] Dataset: {config.dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine if this is PRM or standard RM
    is_prm = config.finetuning_method == "prm"

    if is_prm:
        dataset, data_format = _prepare_prm_dataset(config.dataset_path)
    else:
        dataset, data_format = _prepare_reward_dataset(config.dataset_path)

    print(f"[mud-puppy] Dataset format: {data_format}")

    # Use appropriate trainer based on format and availability
    if data_format == "pairwise" and RewardTrainer is not None:
        _train_with_reward_trainer(config, tokenizer, dataset)
    elif data_format == "prm" and PRMTrainer is not None:
        _train_with_prm_trainer(config, tokenizer, dataset)
    else:
        _train_with_hf_trainer(config, tokenizer, dataset, data_format)

    print(f"[mud-puppy] {config.finetuning_method.upper()} training complete!")


def _train_with_reward_trainer(config: TrainingConfig, tokenizer, dataset):
    """Train using TRL's RewardTrainer for pairwise preferences."""
    from trl import RewardConfig

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=1,
        trust_remote_code=config.trust_remote_code,
    )

    reward_config = RewardConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation,
        fp16=config.precision == "fp16",
        bf16=config.precision == "bf16",
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


def _train_with_prm_trainer(config: TrainingConfig, tokenizer, dataset):
    """Train using TRL's PRMTrainer for process reward modeling."""
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=1,
        trust_remote_code=config.trust_remote_code,
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation,
        fp16=config.precision == "fp16",
        bf16=config.precision == "bf16",
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = PRMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


def _train_with_hf_trainer(config: TrainingConfig, tokenizer, dataset, data_format: str):
    """Fallback training using standard HF Trainer."""
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=1,
        trust_remote_code=config.trust_remote_code,
    )

    def tokenize(batch: Dict[str, str]):
        if "text" in batch:
            return tokenizer(batch["text"], truncation=True, padding=True, max_length=2048)
        elif "chosen" in batch and "rejected" in batch:
            # For pairwise, concatenate prompt+chosen and prompt+rejected
            chosen_texts = [
                f"{p} {c}" for p, c in zip(batch.get("prompt", [""]*len(batch["chosen"])), batch["chosen"])
            ]
            rejected_texts = [
                f"{p} {r}" for p, r in zip(batch.get("prompt", [""]*len(batch["rejected"])), batch["rejected"])
            ]
            return tokenizer(chosen_texts + rejected_texts, truncation=True, padding=True, max_length=2048)
        else:
            raise ValueError("Unsupported dataset format for HF Trainer fallback")

    dataset = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation,
        fp16=config.precision == "fp16",
        bf16=config.precision == "bf16",
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
