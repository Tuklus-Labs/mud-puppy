import os

os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:128")

from typing import Dict, List

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
    """Load the base model and tokenizer with optional quantization."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=config.trust_remote_code
    )

    if config.finetuning_method == "qlora":
        from .bnb_rocm import quantize_model_4bit

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=(
                torch.bfloat16 if config.precision == "bf16" else torch.float16
            ),
            trust_remote_code=config.trust_remote_code,
        )
        model = quantize_model_4bit(
            model,
            dtype=torch.bfloat16 if config.precision == "bf16" else torch.float16,
        )
        try:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
        except Exception:
            pass
    elif config.finetuning_method == "gptq":
        if torch.version.hip is not None:
            from .gptq_rocm import load_quantized

            model = load_quantized(
                AutoModelForCausalLM,
                config.model_name_or_path,
                trust_remote_code=config.trust_remote_code,
            )
        else:
            try:  # pragma: no cover - auto_gptq optional
                from auto_gptq import AutoGPTQForCausalLM
            except ImportError as e:  # pragma: no cover - dependency missing
                raise RuntimeError("auto-gptq is required for GPTQ mode") from e
            model = AutoGPTQForCausalLM.from_quantized(
                config.model_name_or_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=config.trust_remote_code,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, trust_remote_code=config.trust_remote_code
        )

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
        target_modules=config.lora_target_modules,
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
        dataloader_num_workers=config.dataloader_workers,
        dataloader_pin_memory=True,
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
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            examples["messages"], tokenize=False, add_generation_prompt=False
        )
    else:
        # Fallback: concatenate messages
        text = "\n".join(m["content"] for m in examples["messages"])
    return {"text": text}


def load_and_preprocess_dataset(config: TrainingConfig, tokenizer):
    """Load a JSONL chat dataset and tokenize it."""
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=config.dataset_path)["train"]

    if config.use_chat_template:
        dataset = dataset.map(
            lambda ex: apply_chat_template(ex, tokenizer),
            remove_columns=dataset.column_names,
            num_proc=config.preprocessing_workers,
        )
    else:
        dataset = dataset.map(
            lambda ex: {"text": "\n".join(m["content"] for m in ex["messages"])},
            remove_columns=dataset.column_names,
            num_proc=config.preprocessing_workers,
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
        num_proc=config.preprocessing_workers,
    )

    return dataset


def run_training(config: TrainingConfig):
    configure_rocm()
    device = get_device()
    model, tokenizer = load_model(config)

    if config.compile:
        model = torch.compile(model)

    if config.finetuning_method in {"lora", "qlora"}:
        model = prepare_lora(model, config)

    if config.finetuning_method == "qat":
        from .qat_rocm import apply_qat

        model = apply_qat(model)

    if config.use_gradient_checkpointing:
        # Enable checkpointing and ensure inputs require grad so backward works
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    training_args = create_training_args(config)

    dataset = load_and_preprocess_dataset(config, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer_cls = FP8Trainer if config.precision == "fp8" else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    if config.finetuning_method == "gptq":  # pragma: no cover - auto_gptq optional
        if torch.version.hip is not None:
            from .gptq_rocm import quantize_model_gptq, save_quantized

            quantized_model = quantize_model_gptq(trainer.model)
            os.makedirs(os.path.join(config.output_dir, "gptq"), exist_ok=True)
            save_quantized(
                quantized_model, os.path.join(config.output_dir, "gptq", "model.pt")
            )
        else:
            try:
                from auto_gptq import AutoGPTQForCausalLM

                quantized_model = AutoGPTQForCausalLM.from_pretrained(
                    config.output_dir, trust_remote_code=config.trust_remote_code
                )
                quantized_model.quantize(
                    tokenizer,
                    dataset,
                    use_triton=False,
                )
                quantized_model.save_quantized(
                    os.path.join(config.output_dir, "gptq")
                )
            except Exception:
                print("Warning: GPTQ quantization failed")

    if config.finetuning_method == "qat":
        from .qat_rocm import convert_qat

        trainer.model = convert_qat(trainer.model)

    # Save fine-tuned (and possibly quantized) model
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
