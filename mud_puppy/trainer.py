import os
<<<<<<< ozocca-codex/create-llm-fine-tuning-framework-with-rocm-support
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:128")
=======
>>>>>>> main
from typing import Optional, Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
<<<<<<< ozocca-codex/create-llm-fine-tuning-framework-with-rocm-support
    BitsAndBytesConfig,
=======
>>>>>>> main
)

from .config import TrainingConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(config: TrainingConfig):
<<<<<<< ozocca-codex/create-llm-fine-tuning-framework-with-rocm-support
    """Load the base model and tokenizer with optional quantization."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    if config.finetuning_method == "qlora":
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if config.precision == "bf16" else torch.float16,
            )
        except Exception as e:  # pragma: no cover - bitsandbytes may be missing
            raise RuntimeError("bitsandbytes is required for QLoRA") from e
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
    elif config.finetuning_method == "gptq":
        try:  # pragma: no cover - auto_gptq optional
            from auto_gptq import AutoGPTQForCausalLM
        except ImportError as e:  # pragma: no cover - dependency missing
            raise RuntimeError("auto-gptq is required for GPTQ mode") from e
        model = AutoGPTQForCausalLM.from_quantized(
            config.model_name_or_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)

=======
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
>>>>>>> main
    if config.precision == "fp8":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("FP8 precision is not supported in this PyTorch build")
        model.to(torch.float8_e4m3fn)
<<<<<<< ozocca-codex/create-llm-fine-tuning-framework-with-rocm-support

=======
>>>>>>> main
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
<<<<<<< ozocca-codex/create-llm-fine-tuning-framework-with-rocm-support
        dataloader_num_workers=config.dataloader_workers,
        dataloader_pin_memory=True,
=======
>>>>>>> main
        optim="adamw_torch",
    )


class FP8Trainer(Trainer):
    """Trainer subclass enabling FP8 autocast."""

    def training_step(self, model, inputs):
        with torch.autocast("cuda", dtype=torch.float8_e4m3fn):
            return super().training_step(model, inputs)


def configure_rocm():
    """Apply ROCm-specific environment settings."""
<<<<<<< ozocca-codex/create-llm-fine-tuning-framework-with-rocm-support
=======
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:128")
>>>>>>> main
    torch.backends.cuda.matmul.allow_tf32 = True


def apply_chat_template(examples: Dict[str, List[Dict[str, str]]], tokenizer):
    text = tokenizer.apply_chat_template(
        examples["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def load_and_preprocess_dataset(config: TrainingConfig, tokenizer):
    """Load a JSONL chat dataset and tokenize it."""
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=config.dataset_path)["train"]

    # Apply chat template to get plain text
    dataset = dataset.map(
        lambda ex: apply_chat_template(ex, tokenizer),
        remove_columns=dataset.column_names,
<<<<<<< ozocca-codex/create-llm-fine-tuning-framework-with-rocm-support
        num_proc=config.preprocessing_workers,
=======
>>>>>>> main
    )

    # Tokenize WITHOUT return_tensors="pt" in batched mode
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # Let DataCollator handle padding
            max_length=2048,
        )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
<<<<<<< ozocca-codex/create-llm-fine-tuning-framework-with-rocm-support
        num_proc=config.preprocessing_workers,
=======
>>>>>>> main
    )

    return dataset


def run_training(config: TrainingConfig):
    configure_rocm()
    device = get_device()
    model, tokenizer = load_model(config)

<<<<<<< ozocca-codex/create-llm-fine-tuning-framework-with-rocm-support
    if config.compile:
        model = torch.compile(model)

    if config.finetuning_method in {"lora", "qlora"}:
        model = prepare_lora(model, config)

    if config.finetuning_method == "qat":  # pragma: no cover - qat optional
        from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert
        model.qconfig = get_default_qat_qconfig("fbgemm")
        model = prepare_qat(model)

=======
    if config.finetuning_method in {"lora", "qlora"}:
        model = prepare_lora(model, config)

>>>>>>> main
    if config.use_gradient_checkpointing:
        # Enable checkpointing and ensure inputs require grad so backward works
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
<<<<<<< ozocca-codex/create-llm-fine-tuning-framework-with-rocm-support
        model.config.use_cache = False
=======
>>>>>>> main

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

<<<<<<< ozocca-codex/create-llm-fine-tuning-framework-with-rocm-support
    if config.finetuning_method == "gptq":  # pragma: no cover - auto_gptq optional
        try:
            from auto_gptq import AutoGPTQForCausalLM
            quantized_model = AutoGPTQForCausalLM.from_pretrained(model.config._name_or_path)
            quantized_model.quantize(
                tokenizer,
                dataset,
                use_triton=False,
            )
            quantized_model.save_quantized(config.output_dir)
        except Exception as e:
            raise RuntimeError("GPTQ quantization failed") from e

    if config.finetuning_method == "qat":  # pragma: no cover - qat optional
        model = convert(model)

=======
>>>>>>> main
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


