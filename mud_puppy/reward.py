"""Reward modeling utilities."""

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


from .config import TrainingConfig


def train_reward_model(config: TrainingConfig):
    """Train a reward model for RM or PRM."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=config.trust_remote_code
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path, num_labels=1, trust_remote_code=config.trust_remote_code
    )

    dataset = load_dataset("json", data_files=config.dataset_path)["train"]

    if "label" not in dataset.column_names and "labels" not in dataset.column_names:
        # attempt to infer label column
        possible = [c for c in dataset.column_names if c != "text"]
        if len(possible) == 1:
            dataset = dataset.rename_column(possible[0], "label")
        else:
            raise ValueError("Dataset must contain a label column")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=2048)

    dataset = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation,
        fp16=config.precision == "fp16",
        bf16=config.precision == "bf16",
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
