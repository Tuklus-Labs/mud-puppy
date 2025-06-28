import os

os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:128")

from typing import Dict, List
import random
from torch.utils.data import Sampler, DataLoader

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipeline.sync import Pipe
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback,
)

from .config import TrainingConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(config: TrainingConfig):
    """Load the base model and tokenizer with optional quantization."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=config.trust_remote_code
        )
    except OSError as e:
        raise RuntimeError(
            f"Failed to load tokenizer from {config.model_name_or_path}"
        ) from e

    if config.finetuning_method == "qlora":
        from .bnb_rocm import quantize_model_4bit

        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                torch_dtype=(
                    torch.bfloat16 if config.precision == "bf16" else torch.float16
                ),
                trust_remote_code=config.trust_remote_code,
                device_map=config.device_map,
            )
        except OSError as e:
            raise RuntimeError(
                f"Failed to load model from {config.model_name_or_path}"
            ) from e
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
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=config.trust_remote_code,
                device_map=None if config.stream else config.device_map,
            )
        except OSError as e:
            raise RuntimeError(
                f"Failed to load model from {config.model_name_or_path}"
            ) from e

    if config.stream:
        model.to("cpu")
        model = StreamWrapper(model)

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
        lr_scheduler_type=config.lr_scheduler,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        report_to=[config.log_with],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_grad_norm=config.max_grad_norm,
        local_rank=config.local_rank,
        ddp_find_unused_parameters=False,
    )


class FP8Trainer(DynamicBatchTrainer):
    """Trainer subclass enabling FP8 autocast."""

    def training_step(self, model, inputs):
        with torch.autocast("cuda", dtype=torch.float8_e4m3fn):
            return super().training_step(model, inputs)


def compute_metrics(eval_pred):
    """Compute evaluation loss and perplexity."""
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[..., 1:].contiguous().view(-1)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits, shift_labels)
    perplexity = torch.exp(loss)
    return {"eval_loss": loss.item(), "perplexity": perplexity.item()}


class MemoryCallback(TrainerCallback):
    """Log GPU memory usage after each step."""

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"[memory] step {state.global_step}: {mem:.2f} GB")


class GradientClipCallback(TrainerCallback):
    """Clip gradients to avoid exploding values."""

    def __init__(self, max_norm: float):
        self.max_norm = max_norm

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.max_norm > 0 and model is not None:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
            if state.is_local_process_zero:
                print(f"[grad_norm] step {state.global_step}: {norm:.2f}")


class TokenBucketSampler(Sampler[List[int]]):
    """Batch sampler that groups sequences by a token budget."""

    def __init__(self, lengths: List[int], tokens_per_batch: int, shuffle: bool = True):
        self.lengths = lengths
        self.tokens_per_batch = tokens_per_batch
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)
        batch = []
        total = 0
        for idx in indices:
            l = int(self.lengths[idx])
            if batch and total + l > self.tokens_per_batch:
                yield batch
                batch = []
                total = 0
            batch.append(idx)
            total += l
        if batch:
            yield batch

    def __len__(self):
        return len(self.lengths)


class DistributedBatchSampler(Sampler[List[int]]):
    """Distribute batches across multiple processes."""

    def __init__(self, batch_sampler: Sampler[List[int]]):
        self.batches = list(batch_sampler)
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()

    def __iter__(self):
        for i in range(self.rank, len(self.batches), self.num_replicas):
            yield self.batches[i]

    def __len__(self):
        return (len(self.batches) + self.num_replicas - 1) // self.num_replicas


class DynamicBatchTrainer(Trainer):
    """Trainer with dynamic batching via TokenBucketSampler."""

    def __init__(self, *args, tokens_per_batch: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens_per_batch = tokens_per_batch

    def get_train_dataloader(self) -> DataLoader:
        if self.tokens_per_batch > 0:
            lengths = self.train_dataset["length"]
            sampler = TokenBucketSampler(lengths, self.tokens_per_batch, shuffle=True)
            if dist.is_initialized():
                sampler = DistributedBatchSampler(sampler)
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        return super().get_train_dataloader()


class ZeroOffloadOptimizer(torch.optim.AdamW):
    """AdamW variant that keeps optimizer states on the CPU."""

    def step(self, closure=None):  # pragma: no cover - simple device moves
        orig_devices = []
        for group in self.param_groups:
            for p in group["params"]:
                orig_devices.append(p.device)
                if p.grad is not None and p.grad.device != torch.device("cpu"):
                    p.grad = p.grad.cpu()
                if p.device != torch.device("cpu"):
                    p.data = p.data.cpu()
        loss = super().step(closure)
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                device = orig_devices[idx]
                if device != torch.device("cpu"):
                    p.data = p.data.to(device)
                idx += 1
        return loss


class ZeroOffloadTrainer(DynamicBatchTrainer):
    """Trainer that uses the ZeroOffloadOptimizer."""

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = ZeroOffloadOptimizer(
                self.model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer


class StreamWrapper(nn.Module):
    """Wrapper that streams individual modules from CPU to GPU."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def pre_hook(mod, inp):
            mod.to(self.device)

        def post_hook(mod, inp, out):
            mod.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for m in self.model.modules():
            if len(list(m.children())) == 0:
                m.register_forward_pre_hook(pre_hook)
                m.register_forward_hook(post_hook)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, mode: bool = True):  # pragma: no cover - passthrough
        self.model.train(mode)
        return super().train(mode)

    def eval(self):  # pragma: no cover - passthrough
        self.model.eval()
        return super().eval()


def init_distributed(config: TrainingConfig):
    """Initialize torch.distributed if requested."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if config.distributed or world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(config.local_rank)
    return dist.get_rank() if dist.is_initialized() else 0, world_size


def configure_rocm():
    """Apply ROCm-specific environment settings."""
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:128")
    torch.backends.cuda.matmul.allow_tf32 = True


def apply_pipeline_parallel(model: nn.Module, devices: List[int]) -> nn.Module:
    """Naively split a Sequential model across GPUs using Pipe."""
    if not isinstance(model, nn.Sequential):
        return model
    splits = torch.linspace(0, len(model), len(devices) + 1, dtype=torch.int64)
    stages = []
    for i, dev in enumerate(devices):
        start, end = int(splits[i]), int(splits[i + 1])
        stages.append(nn.Sequential(*model[start:end]).to(f"cuda:{dev}"))
    return Pipe(nn.Sequential(*stages))


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
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    if "messages" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'messages' column")

    def _validate(example):
        if not isinstance(example["messages"], list):
            raise ValueError("'messages' must be a list")
        for m in example["messages"]:
            if "content" not in m:
                raise ValueError("All messages must contain a 'content' field")
        return example

    dataset = dataset.map(_validate, num_proc=config.preprocessing_workers)

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

    dataset = dataset.map(
        lambda ex: {"length": len(ex["input_ids"])},
        num_proc=config.preprocessing_workers,
    )

    # Split off 10% of data for evaluation
    split = int(0.1 * len(dataset))
    eval_dataset = dataset.select(range(split))
    train_dataset = dataset.select(range(split, len(dataset)))

    return train_dataset, eval_dataset


def run_training(config: TrainingConfig):
    configure_rocm()
    rank, world_size = init_distributed(config)
    model, tokenizer = load_model(config)

    if config.device_map == "pipeline" and world_size > 1:
        model = apply_pipeline_parallel(model, list(range(world_size)))
    elif world_size > 1 and not config.stream:
        model.to(f"cuda:{config.local_rank}")
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
            find_unused_parameters=False,
        )

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

    train_dataset, eval_dataset = load_and_preprocess_dataset(config, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    base_cls = FP8Trainer if config.precision == "fp8" else DynamicBatchTrainer
    trainer_cls = base_cls
    if config.zero_offload:

        class ZeroTrainer(ZeroOffloadTrainer, base_cls):
            pass

        trainer_cls = ZeroTrainer
    callbacks = [MemoryCallback(), GradientClipCallback(config.max_grad_norm)]
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        tokens_per_batch=config.tokens_per_batch,
    )

    checkpoint = None
    if config.resume:
        from transformers.trainer_utils import get_last_checkpoint

        checkpoint = get_last_checkpoint(config.output_dir)

    try:
        trainer.train(resume_from_checkpoint=checkpoint)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and torch.cuda.is_available():
            print(
                f"CUDA OOM at step {trainer.state.global_step}."
                f" Max memory {torch.cuda.max_memory_reserved() / 1e9:.2f} GB"
            )
            raise
        else:
            raise

    metrics = trainer.evaluate()
    print(metrics)

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
                    train_dataset,
                    use_triton=False,
                )
                quantized_model.save_quantized(os.path.join(config.output_dir, "gptq"))
            except Exception:
                print("Warning: GPTQ quantization failed")

    if config.finetuning_method == "qat":
        from .qat_rocm import convert_qat

        trainer.model = convert_qat(trainer.model)

    if not dist.is_initialized() or rank == 0:
        trainer.save_model(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
    if dist.is_initialized():
        dist.destroy_process_group()
