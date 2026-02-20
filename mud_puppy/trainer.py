"""Core training module for mud-puppy.

This module contains the main training loop, data loading utilities, and
specialized trainers for features like dynamic batching and layer streaming.
"""

import os

# Favor ROCm-friendly allocator behavior by default
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:128")

from typing import Dict, List, Optional, Iterator, Any
import random
from torch.utils.data import Sampler, DataLoader, Dataset

import torch
import torch.nn as nn
import torch.distributed as dist

try:
    from torch.distributed.pipeline.sync import Pipe
except Exception:  # pragma: no cover - optional dependency
    Pipe = None

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Optional vision import (not available in all transformers versions)
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None

from .config import TrainingConfig


def is_rocm() -> bool:
    """Return True if this PyTorch build has HIP/ROCm support."""
    return getattr(torch.version, "hip", None) is not None


def get_device() -> torch.device:
    """Select the primary compute device (ROCm/CUDA/CPU)."""
    if torch.cuda.is_available():  # includes ROCm builds
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_model_for_kbit_training_rocm(model: nn.Module) -> None:
    """Prepare a quantized model for k-bit training (ROCm-native).

    This replaces peft's prepare_model_for_kbit_training which requires bitsandbytes.
    It performs the following:
    1. Casts LayerNorm and embedding layers to float32 for training stability
    2. Ensures quantized base weights are frozen
    3. Enables input gradients for LoRA to work
    """
    # Cast normalization layers to float32 for stability
    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            module.float()

        # Also cast embedding layers
        if isinstance(module, nn.Embedding):
            module.float()

    # Ensure quantized weights are frozen (Linear4bit/LinearMX4 already do this, but be safe)
    from .bnb_rocm import Linear4bit
    try:
        from .mxfp4_rocm import LinearMX4
    except ImportError:
        LinearMX4 = None

    for module in model.modules():
        if isinstance(module, Linear4bit):
            # Base weight should be frozen, LoRA adapters will be trainable
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
        elif LinearMX4 is not None and isinstance(module, LinearMX4):
            # MXFP4: packed weights are buffers (already frozen), bias is trainable
            if module.bias is not None:
                module.bias.requires_grad = False  # Will be enabled by LoRA if needed

    # Enable input gradients so LoRA can receive gradients
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        # Manual fallback
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


class StreamWrapper(nn.Module):
    """Wrapper that streams model layers to GPU on demand.

    This allows training models that don't fit entirely in GPU memory by
    keeping most layers on CPU and moving them to GPU only when needed.

    Strategy:
    - Keep embedding/lm_head on GPU always (small, always needed)
    - Move transformer layers to GPU for forward pass
    - Keep layers on GPU through backward pass
    - Offload to CPU after optimizer step (via offload_to_cpu method)

    Note: This provides memory savings between training steps, not during
    the forward/backward pass itself. For true streaming during forward/backward,
    use gradient checkpointing with this wrapper.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._device = get_device()
        self._cpu = torch.device("cpu")
        self._on_gpu = False

        # Find the transformer layers (supports LLaMA, GPT-2, etc.)
        self._layers = self._find_layers()

        # Move everything to CPU initially
        self.model.to(self._cpu)

        # Keep embedding and output layers on GPU (small and always needed)
        self._move_endpoints_to_gpu()

    def _find_layers(self) -> List[nn.Module]:
        """Find the main transformer layers that can be streamed."""
        # LLaMA-style: model.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return list(self.model.model.layers)
        # GPT-2 style: transformer.h
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return list(self.model.transformer.h)
        # Fallback: no streaming possible
        return []

    def _move_endpoints_to_gpu(self):
        """Move embedding and output projection to GPU (always needed)."""
        # Move embeddings
        if hasattr(self.model, 'get_input_embeddings'):
            emb = self.model.get_input_embeddings()
            if emb is not None:
                emb.to(self._device)

        # Move output projection (lm_head)
        if hasattr(self.model, 'lm_head'):
            self.model.lm_head.to(self._device)

        # Move final layer norm
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
            self.model.model.norm.to(self._device)
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'ln_f'):
            self.model.transformer.ln_f.to(self._device)

    def _ensure_on_gpu(self):
        """Move all layers to GPU for forward/backward pass."""
        if not self._on_gpu:
            for layer in self._layers:
                layer.to(self._device)
            self._on_gpu = True

    def offload_to_cpu(self):
        """Move transformer layers back to CPU to free VRAM.

        Call this after optimizer.step() to free memory between training steps.
        """
        if self._on_gpu:
            for layer in self._layers:
                layer.to(self._cpu)
            self._on_gpu = False
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def forward(self, *args, **kwargs):
        # Ensure all layers are on GPU for forward pass
        self._ensure_on_gpu()

        # Move inputs to GPU
        args = tuple(
            a.to(self._device) if isinstance(a, torch.Tensor) else a for a in args
        )
        kwargs = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }

        return self.model(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class DynamicBatchSampler(Sampler):
    """Sampler that creates batches based on a token budget.

    Instead of fixed batch sizes, this sampler groups sequences such that
    each batch contains approximately `tokens_per_batch` tokens total.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokens_per_batch: int,
        length_column: str = "length",
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.tokens_per_batch = tokens_per_batch
        self.length_column = length_column
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Extract sequence lengths
        if hasattr(dataset, "features") and length_column in dataset.features:
            self.lengths = dataset[length_column]
        elif hasattr(dataset, "__getitem__"):
            # Try to compute lengths from input_ids
            self.lengths = []
            for i in range(len(dataset)):
                item = dataset[i]
                if "input_ids" in item:
                    self.lengths.append(len(item["input_ids"]))
                else:
                    self.lengths.append(512)  # Default estimate
        else:
            self.lengths = [512] * len(dataset)

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        # Sort by length for more efficient batching (within shuffled chunks)
        chunk_size = min(10000, len(indices))
        sorted_indices = []
        for i in range(0, len(indices), chunk_size):
            chunk = indices[i : i + chunk_size]
            chunk.sort(key=lambda x: self.lengths[x])
            sorted_indices.extend(chunk)

        # Create batches based on token budget
        batches = []
        current_batch = []
        current_tokens = 0

        for idx in sorted_indices:
            seq_len = self.lengths[idx]

            if current_tokens + seq_len > self.tokens_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(idx)
            current_tokens += seq_len

        # Handle last batch
        if current_batch and not self.drop_last:
            batches.append(current_batch)

        # Shuffle batches to avoid always training on similar-length sequences
        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        # Approximate number of batches
        total_tokens = sum(self.lengths)
        return max(1, total_tokens // self.tokens_per_batch)


class ZeroOffloadCallback(TrainerCallback):
    """Callback to offload optimizer states to CPU memory.

    This is a ROCm-native implementation of ZeRO-Offload that doesn't
    require DeepSpeed (which is CUDA-only).

    The approach:
    - After each optimizer step, move optimizer states to CPU
    - Before each step, states are automatically moved back as needed
    - This saves VRAM between steps (optimizer states are 2x model size for Adam)
    """

    def __init__(self):
        self._optimizer_wrapped = False
        self._cpu_states = {}
        self._cpu = torch.device("cpu")
        self._gpu = torch.device("cuda") if torch.cuda.is_available() else self._cpu

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize CPU offload structures."""
        print("[mud-puppy] ZeRO-Offload enabled (ROCm-native)")

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        """After optimizer step, offload states to CPU."""
        if optimizer is None:
            return

        # Move optimizer states to CPU to free VRAM
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param not in optimizer.state:
                    continue

                opt_state = optimizer.state[param]
                pid = id(param)

                # Initialize CPU storage for this param
                if pid not in self._cpu_states:
                    self._cpu_states[pid] = {}

                # Move each state tensor to CPU
                for key, val in opt_state.items():
                    if isinstance(val, torch.Tensor) and val.device.type == "cuda":
                        # Move to CPU (pinned memory for faster transfer back)
                        if pid not in self._cpu_states or key not in self._cpu_states[pid]:
                            cpu_tensor = torch.empty(
                                val.shape, dtype=val.dtype, device=self._cpu,
                                pin_memory=True
                            )
                            self._cpu_states[pid][key] = cpu_tensor

                        self._cpu_states[pid][key].copy_(val, non_blocking=True)
                        opt_state[key] = self._cpu_states[pid][key]

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_step_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        """Before optimizer step, move states back to GPU."""
        if optimizer is None:
            return

        # Move optimizer states back to GPU for the step
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param not in optimizer.state:
                    continue

                opt_state = optimizer.state[param]

                for key, val in opt_state.items():
                    if isinstance(val, torch.Tensor) and val.device.type == "cpu":
                        # Move back to GPU
                        gpu_tensor = val.to(self._gpu, non_blocking=True)
                        opt_state[key] = gpu_tensor


class LoggingCallback(TrainerCallback):
    """Enhanced logging callback for mud-puppy."""

    def __init__(self, log_with: str = "none"):
        self.log_with = log_with

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step % 10 == 0:
            loss = logs.get("loss", logs.get("train_loss", "N/A"))
            lr = logs.get("learning_rate", "N/A")
            print(f"[mud-puppy] step {state.global_step}: loss={loss}, lr={lr}")


def load_model(config: TrainingConfig):
    """Load the base model and tokenizer with optional quantization.

    For ``finetuning_method == 'multimodal'``, this will attempt to load a
    vision+language model via :class:`AutoModelForVision2Seq`.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=config.trust_remote_code
        )
    except OSError as e:
        raise RuntimeError(
            f"Failed to load tokenizer from {config.model_name_or_path}"
        ) from e

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Multimodal: delegate to vision-language model class
    if config.finetuning_method == "multimodal":
        if AutoModelForVision2Seq is None:
            raise RuntimeError(
                "Multimodal training requires AutoModelForVision2Seq, "
                "which is not available in this version of transformers"
            )
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=config.trust_remote_code,
                device_map=None if config.stream else config.device_map,
            )
        except OSError as e:
            raise RuntimeError(
                "Failed to load multimodal model; ensure this checkpoint "
                "is compatible with AutoModelForVision2Seq"
            ) from e

    # QLoRA: load in 16-bit, apply 4-bit quantization, prepare for k-bit training
    elif config.finetuning_method == "qlora":
        model_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16

        # ROCm fix: Load to CPU first, then move to GPU (avoid device_map segfaults)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                dtype=model_dtype,
                trust_remote_code=config.trust_remote_code,
                device_map=None,  # Load to CPU
                low_cpu_mem_usage=True,
            )
        except OSError as e:
            raise RuntimeError(
                f"Failed to load model from {config.model_name_or_path}"
            ) from e

        # Apply 4-bit quantization based on backend choice
        quant_backend = getattr(config, "quant_backend", "int4")
        if quant_backend == "mxfp4":
            from .mxfp4_rocm import quantize_model_mx4
            block_size = getattr(config, "quant_block_size", 32)
            print(f"[mud-puppy] Applying MXFP4 quantization (block_size={block_size})...")
            model = quantize_model_mx4(model, block_size=block_size)
        else:
            from .bnb_rocm import quantize_model_4bit
            print("[mud-puppy] Applying INT4 quantization (ROCm-native)...")
            model = quantize_model_4bit(model, dtype=model_dtype)

        # Move to GPU
        if torch.cuda.is_available():
            print("[mud-puppy] Moving quantized model to GPU...")
            model = model.to("cuda")

        # Prepare model for k-bit training (ROCm-native implementation)
        # This replaces peft's prepare_model_for_kbit_training which requires bitsandbytes
        print("[mud-puppy] Preparing model for k-bit training...")
        _prepare_model_for_kbit_training_rocm(model)

    # GPTQ: either ROCm-native simple GPTQ, or auto-gptq on CUDA
    elif config.finetuning_method == "gptq":
        if is_rocm():
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
                raise RuntimeError("auto-gptq is required for GPTQ mode on CUDA") from e
            model = AutoGPTQForCausalLM.from_quantized(
                config.model_name_or_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=config.trust_remote_code,
            )

    # QAT: load and apply quantization-aware training wrappers
    elif config.finetuning_method == "qat":
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

        from .qat_rocm import apply_qat

        model = apply_qat(model, bits=8)

    # Standard full / LoRA / other methods
    else:
        # ROCm fix: Load to CPU first, then move to GPU
        # Transformers' automatic device_map causes segfaults on ROCm
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        model_dtype = dtype_map.get(config.precision, torch.float16)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=config.trust_remote_code,
                dtype=model_dtype,
                device_map=None,  # Load to CPU
                low_cpu_mem_usage=True,
            )
        except OSError as e:
            raise RuntimeError(
                f"Failed to load model from {config.model_name_or_path}"
            ) from e

        # Move to GPU if available and not streaming
        if not config.stream and torch.cuda.is_available():
            print("[mud-puppy] Moving model to GPU...")
            model = model.to("cuda")

    # Streaming: keep model on CPU and stream leaf modules to active device
    if config.stream:
        model.to("cpu")
        model = StreamWrapper(model)

    # Experimental FP8 support
    if config.precision == "fp8":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("FP8 precision is not supported in this PyTorch build")
        model.to(torch.float8_e4m3fn)

    return model, tokenizer


def prepare_lora(model: nn.Module, config: TrainingConfig) -> nn.Module:
    """Attach LoRA/QLoRA adapters to the model."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise RuntimeError("peft is required for LoRA training") from e

    # Auto-detect target modules based on model architecture
    target_modules = config.lora_target_modules

    # Get all module names to check what's available
    module_names = [name for name, _ in model.named_modules()]
    module_names_str = " ".join(module_names)

    # Check if default LLaMA-style modules exist
    if "q_proj" not in module_names_str:
        # GPT-2 style: c_attn, c_proj, c_fc
        if "c_attn" in module_names_str:
            target_modules = ["c_attn", "c_proj"]
            print(f"[mud-puppy] Auto-detected GPT-2 style model, using targets: {target_modules}")
        # GPT-Neo/J style: q_proj exists but different structure
        elif "attn.attention" in module_names_str:
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
            print(f"[mud-puppy] Auto-detected GPT-Neo style model, using targets: {target_modules}")
        else:
            # Fall back to finding any Linear layers with common attention names
            linear_modules = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    short_name = name.split(".")[-1]
                    if short_name not in linear_modules and any(
                        kw in name.lower() for kw in ["attn", "query", "key", "value", "proj"]
                    ):
                        linear_modules.append(short_name)
            if linear_modules:
                target_modules = list(set(linear_modules))[:4]
                print(f"[mud-puppy] Auto-detected target modules: {target_modules}")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def merge_lora_weights(
    model: nn.Module, config: TrainingConfig, tokenizer: PreTrainedTokenizer
) -> nn.Module:
    """Merge LoRA adapters back into the base model."""
    try:
        from peft import PeftModel
    except ImportError:
        print("[mud-puppy] Warning: peft not available, skipping LoRA merge")
        return model

    if not isinstance(model, PeftModel):
        print("[mud-puppy] Model is not a PeftModel, skipping merge")
        return model

    print("[mud-puppy] Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # Convert to specified precision
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    target_dtype = dtype_map.get(config.merge_precision, torch.bfloat16)
    model = model.to(target_dtype)

    # Save merged model
    merged_path = os.path.join(config.output_dir, "merged")
    os.makedirs(merged_path, exist_ok=True)
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"[mud-puppy] Merged model saved to {merged_path}")

    return model


def create_training_args(config: TrainingConfig) -> TrainingArguments:
    """Construct Hugging Face TrainingArguments from TrainingConfig."""
    precision = config.precision
    fp16 = precision == "fp16"
    bf16 = precision == "bf16"

    # ROCm bf16 detection fix: HF TrainingArguments doesn't properly detect ROCm support
    if bf16 and torch.cuda.is_available():
        # Check if bf16 is actually supported on this GPU
        try:
            # Try to create a bf16 tensor on GPU - this will fail if not supported
            test_tensor = torch.zeros(1, dtype=torch.bfloat16, device="cuda")
            del test_tensor
            print("[mud-puppy] bf16 support verified on GPU")
        except (RuntimeError, AssertionError):
            print("[mud-puppy] bf16 not supported on this GPU, falling back to fp16")
            bf16 = False
            fp16 = True
    elif bf16 and not torch.cuda.is_available():
        print("[mud-puppy] No GPU available, disabling bf16")
        bf16 = False

    # transformers 5.0 changed report_to behavior - empty list means no reporting
    report_to = [] if config.log_with == "none" else [config.log_with]

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
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        lr_scheduler_type=config.lr_scheduler,
        eval_strategy="epoch" if config.early_stopping_patience > 0 else "no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        report_to=report_to,
        save_total_limit=2,
        load_best_model_at_end=config.early_stopping_patience > 0,
        metric_for_best_model="eval_loss" if config.early_stopping_patience > 0 else None,
        max_grad_norm=config.max_grad_norm,
        local_rank=config.local_rank,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )


def load_and_preprocess_dataset(
    config: TrainingConfig, tokenizer: PreTrainedTokenizer
) -> Dataset:
    """Load and preprocess the training dataset.

    Supports JSONL format with chat-style messages or simple text fields.
    """
    # Load the dataset
    dataset = load_dataset("json", data_files=config.dataset_path, split="train")

    # Determine the format and preprocess accordingly
    columns = dataset.column_names

    def tokenize_chat(examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize chat-formatted examples."""
        texts = []

        for i in range(len(examples[columns[0]])):
            # Try to extract messages in various formats
            if "messages" in columns:
                messages = examples["messages"][i]
                if (
                    config.use_chat_template
                    and hasattr(tokenizer, "apply_chat_template")
                    and tokenizer.chat_template is not None
                ):
                    try:
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                    except Exception:
                        # Fallback if chat template fails
                        text = "\n".join(
                            f"{m.get('role', 'user')}: {m.get('content', '')}"
                            for m in messages
                        )
                else:
                    # Fallback: concatenate message contents
                    text = "\n".join(
                        f"{m.get('role', 'user')}: {m.get('content', '')}"
                        for m in messages
                    )
            elif "text" in columns:
                text = examples["text"][i]
            elif "input" in columns and "output" in columns:
                text = f"{examples['input'][i]}\n{examples['output'][i]}"
            elif "prompt" in columns and "completion" in columns:
                text = f"{examples['prompt'][i]}{examples['completion'][i]}"
            elif "instruction" in columns:
                text = examples["instruction"][i]
                if "response" in columns:
                    text = f"{text}\n{examples['response'][i]}"
            else:
                # Use first text-like column
                text = str(examples[columns[0]][i])

            texts.append(text)

        # Tokenize - use config max_seq_length if set, else model default (capped at 2048)
        if config.max_seq_length > 0:
            max_len = config.max_seq_length
        else:
            max_len = min(
                getattr(tokenizer, "model_max_length", 2048),
                2048
            )
            # GPT-2 and similar models have 1024 max positions
            if max_len > 1024 and hasattr(tokenizer, "name_or_path"):
                if "gpt2" in tokenizer.name_or_path.lower():
                    max_len = 1024

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors=None,
        )

        # Add labels for causal LM training
        tokenized["labels"] = tokenized["input_ids"].copy()

        # Add length column for dynamic batching
        tokenized["length"] = [len(ids) for ids in tokenized["input_ids"]]

        return tokenized

    # Apply tokenization
    dataset = dataset.map(
        tokenize_chat,
        batched=True,
        num_proc=config.preprocessing_workers,
        remove_columns=columns,
        desc="Tokenizing dataset",
    )

    # Set format for PyTorch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset


def setup_distributed(config: TrainingConfig) -> bool:
    """Initialize distributed training if requested."""
    if not config.distributed:
        return False

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(config.local_rank)
    return True


def setup_pipeline_parallelism(model: nn.Module, config: TrainingConfig) -> nn.Module:
    """Set up pipeline parallelism if requested."""
    if config.device_map != "pipeline":
        return model

    if Pipe is None:
        print(
            "[mud-puppy] Warning: torch.distributed.pipeline not available, "
            "falling back to standard model parallelism"
        )
        return model

    # Get the model's layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        print(
            "[mud-puppy] Warning: Could not identify model layers for pipeline "
            "parallelism, falling back to standard parallelism"
        )
        return model

    # Determine number of GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("[mud-puppy] Pipeline parallelism requires at least 2 GPUs")
        return model

    # Split layers across GPUs
    layers_per_gpu = len(layers) // num_gpus

    # Create sequential module for pipeline
    chunks = []
    for i in range(num_gpus):
        start_idx = i * layers_per_gpu
        end_idx = start_idx + layers_per_gpu if i < num_gpus - 1 else len(layers)
        chunk_layers = nn.Sequential(*[layers[j] for j in range(start_idx, end_idx)])
        chunks.append(chunk_layers.to(f"cuda:{i}"))

    try:
        pipe_model = Pipe(nn.Sequential(*chunks), chunks=num_gpus)
        print(f"[mud-puppy] Pipeline parallelism enabled across {num_gpus} GPUs")
        return pipe_model
    except Exception as e:
        print(f"[mud-puppy] Failed to create pipeline: {e}")
        return model


def run_training(config: TrainingConfig) -> None:
    """Main training entry point for mud-puppy.

    This function orchestrates the complete training pipeline:
    1. Load model and tokenizer
    2. Apply LoRA/QLoRA if requested
    3. Load and preprocess dataset
    4. Set up trainer with appropriate callbacks
    5. Run training
    6. Save model and optionally merge LoRA weights
    """
    print(f"[mud-puppy] Starting {config.finetuning_method} training")
    print(f"[mud-puppy] Model: {config.model_name_or_path}")
    print(f"[mud-puppy] Dataset: {config.dataset_path}")
    print(f"[mud-puppy] Output: {config.output_dir}")
    print(f"[mud-puppy] Precision: {config.precision}")

    # Warn about incompatible options
    if config.stream and config.finetuning_method in ("lora", "qlora"):
        print("[mud-puppy] WARNING: Streaming + LoRA is not supported (LoRA modules won't stream)")
        print("[mud-puppy] Disabling streaming mode for this training run")
        config = TrainingConfig(**{**config.__dict__, "stream": False})

    # Set up distributed training if requested
    is_distributed = setup_distributed(config)
    if is_distributed:
        print(f"[mud-puppy] Distributed training enabled (rank {config.local_rank})")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model(config)

    # Apply LoRA if requested
    if config.finetuning_method in ("lora", "qlora"):
        model = prepare_lora(model, config)

    # Set up pipeline parallelism if requested
    if config.device_map == "pipeline":
        model = setup_pipeline_parallelism(model, config)

    # Enable gradient checkpointing
    if config.use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Compile model if requested
    if config.compile and hasattr(torch, "compile"):
        print("[mud-puppy] Compiling model with torch.compile...")
        model = torch.compile(model)

    # Load and preprocess dataset
    dataset = load_and_preprocess_dataset(config, tokenizer)
    print(f"[mud-puppy] Dataset loaded: {len(dataset)} examples")

    # Split dataset for evaluation if early stopping is enabled
    if config.early_stopping_patience > 0:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    # Create training arguments
    training_args = create_training_args(config)

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Set up callbacks
    callbacks = [LoggingCallback(config.log_with)]

    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        )

    if config.zero_offload:
        callbacks.append(ZeroOffloadCallback())

    # Create trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "processing_class": tokenizer,  # transformers 5.0 renamed tokenizer to processing_class
        "callbacks": callbacks,
    }

    # Use dynamic batching if tokens_per_batch is specified
    if config.tokens_per_batch > 0:
        print(f"[mud-puppy] Using dynamic batching with {config.tokens_per_batch} tokens/batch")
        # Note: The Trainer will use its own batching, but we can influence it
        # through custom data collator or by passing pre-batched data

    trainer = Trainer(**trainer_kwargs)

    # Resume from checkpoint if requested
    resume_from = None
    if config.resume:
        checkpoints = [
            d for d in os.listdir(config.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_from = os.path.join(config.output_dir, latest)
            print(f"[mud-puppy] Resuming from {resume_from}")

    # Run training
    print("[mud-puppy] Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    print("[mud-puppy] Saving model...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Post-training: QAT conversion
    if config.finetuning_method == "qat":
        from .qat_rocm import convert_qat

        print("[mud-puppy] Converting QAT model to int8...")
        model = convert_qat(model, bits=8)
        qat_path = os.path.join(config.output_dir, "qat_int8")
        os.makedirs(qat_path, exist_ok=True)
        model.save_pretrained(qat_path)
        tokenizer.save_pretrained(qat_path)
        print(f"[mud-puppy] QAT model saved to {qat_path}")

    # Post-training: GPTQ quantization
    if config.finetuning_method == "gptq":
        from .gptq_rocm import quantize_model_gptq, save_quantized

        print("[mud-puppy] Applying GPTQ quantization...")
        model = quantize_model_gptq(model, bits=4)
        gptq_path = os.path.join(config.output_dir, "gptq")
        os.makedirs(gptq_path, exist_ok=True)
        save_quantized(model, gptq_path)
        tokenizer.save_pretrained(gptq_path)
        print(f"[mud-puppy] GPTQ model saved to {gptq_path}")

    # Merge LoRA weights if requested
    if config.merge_lora and config.finetuning_method in ("lora", "qlora"):
        model = merge_lora_weights(model, config, tokenizer)

    print("[mud-puppy] Training complete!")
