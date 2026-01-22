"""Preference-based fine tuning algorithms.

This module implements several preference-based fine-tuning algorithms on top
of `trl`, including DPO, IPO, KTO, and ORPO where available.

Datasets are expected to be JSON/JSONL files with at least the following
columns for pairwise preference training:

- ``prompt``: the input text / question
- ``chosen``: the preferred response
- ``rejected``: the less preferred response

Some algorithms (e.g. ORPO) may also use single-response columns; for now we
stick to the common pairwise schema used by TRL's trainers.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from .config import TrainingConfig

# Import available trainers from TRL
DPOTrainer = None
KTOTrainer = None
ORPOTrainer = None

try:  # pragma: no cover - optional dependencies
    from trl import DPOTrainer
except ImportError:
    pass

try:  # pragma: no cover
    from trl import KTOTrainer
except ImportError:
    pass

try:  # pragma: no cover
    from trl import ORPOTrainer
except ImportError:
    pass


SUPPORTED_PREFERENCES = {"dpo", "ipo", "kto", "orpo"}


def _load_pairwise_dataset(path: str):
    """Load a pairwise preference dataset with (prompt, chosen, rejected)."""
    dataset = load_dataset("json", data_files=path)["train"]

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
    return dataset


def _build_training_args(config: TrainingConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation,
        fp16=config.precision == "fp16",
        bf16=config.precision == "bf16",
        remove_unused_columns=False,
        logging_steps=10,
        save_strategy="epoch",
    )


def run_preference_training(config: TrainingConfig):
    """Run preference tuning via DPO/IPO/KTO/ORPO.

    The specific algorithm is selected via ``config.preference`` and must be
    one of ``{"dpo", "ipo", "kto", "orpo"}``.

    Note: IPO (Identity Preference Optimization) is implemented as a loss_type
    variant of DPO in modern TRL versions.
    """
    if config.preference not in SUPPORTED_PREFERENCES:
        raise ValueError(f"Unsupported preference method: {config.preference}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path, trust_remote_code=config.trust_remote_code
    )

    dataset = _load_pairwise_dataset(config.dataset_path)
    training_args = _build_training_args(config)

    pref = config.preference.lower()

    if pref == "dpo":
        if DPOTrainer is None:
            raise RuntimeError("DPOTrainer is required but not available in trl")
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # DPOTrainer will create a copy
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            beta=0.1,
            loss_type="sigmoid",  # Standard DPO loss
        )
    elif pref == "ipo":
        # IPO is implemented as a DPO variant with loss_type="ipo"
        if DPOTrainer is None:
            raise RuntimeError("DPOTrainer is required for IPO (as loss_type variant)")
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            beta=0.1,
            loss_type="ipo",  # IPO loss variant
        )
    elif pref == "kto":
        if KTOTrainer is None:
            raise RuntimeError("KTOTrainer is required but not available in trl")
        trainer = KTOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    elif pref == "orpo":
        if ORPOTrainer is None:
            raise RuntimeError("ORPOTrainer is required but not available in trl")
        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:  # pragma: no cover - guarded by SUPPORTED_PREFERENCES
        raise ValueError(f"Unsupported preference method: {config.preference}")

    print(f"[mud-puppy] Starting {pref.upper()} preference training...")
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"[mud-puppy] {pref.upper()} training complete!")
