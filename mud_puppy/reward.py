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
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path, num_labels=1
    )

    dataset = load_dataset("json", data_files=config.dataset_path)["train"]

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

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


