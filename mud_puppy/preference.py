"""Preference-based fine tuning algorithms."""

from typing import Dict

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer


from .config import TrainingConfig


SUPPORTED_PREFERENCES = {"dpo"}


def run_preference_training(config: TrainingConfig):
    """Run preference tuning via DPO/IPO/KTO/ORPO."""
    if config.preference not in SUPPORTED_PREFERENCES:
        raise ValueError(f"Unsupported preference method: {config.preference}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=config.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path, trust_remote_code=config.trust_remote_code
    )

    dataset = load_dataset("json", data_files=config.dataset_path)["train"]

    required_columns = {"prompt", "chosen", "rejected"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset missing columns: {', '.join(sorted(missing))}")

    def preprocess(batch: Dict[str, str]):
        return {
            "prompt": batch["prompt"],
            "chosen": batch["chosen"],
            "rejected": batch["rejected"],
        }

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation,
        fp16=config.precision == "fp16",
        bf16=config.precision == "bf16",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
