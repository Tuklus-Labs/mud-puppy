"""Core training module for mud-puppy.

This module contains the main training loop, data loading utilities, and
specialized trainers for features like dynamic batching and layer streaming.
"""

import os
import logging
import warnings

# Favor ROCm-friendly allocator behavior by default
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

from typing import Dict, List, Optional, Iterator, Any
import random

log = logging.getLogger(__name__)
from torch.utils.data import Sampler, DataLoader, Dataset

import torch
import torch.nn as nn
import torch.distributed as dist

try:
    from torch.distributed.pipeline.sync import Pipe
except ImportError:  # pragma: no cover - optional dependency
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

# Optional vision imports (not available in all transformers versions)
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq
        AutoProcessor = None
    except ImportError:
        AutoModelForVision2Seq = None
        AutoProcessor = None

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
    1. Casts non-quantized layers (norms, embeddings, lm_head) to float32
       for training stability and dtype consistency
    2. Quantized layers (Linear4bit/LinearMX4) are already frozen (packed buffers)
    3. Enables input gradients for LoRA to work
    """
    from .bnb_rocm import Linear4bit
    try:
        from .mxfp4_rocm import LinearMX4
    except ImportError:
        LinearMX4 = None

    quant_types = (Linear4bit,) if LinearMX4 is None else (Linear4bit, LinearMX4)

    for name, module in model.named_modules():
        # Cast normalization layers to float32 for stability
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            module.float()
        # Cast embedding layers to float32
        elif isinstance(module, nn.Embedding):
            module.float()

    # Add a forward pre-hook on non-quantized Linear layers (lm_head, etc.)
    # to cast input to match weight dtype. This avoids the fp32 VRAM cost of
    # upcasting the weight (e.g. 2GB for lm_head on 128K-vocab models) while
    # fixing the dtype mismatch from fp32 norm outputs.
    def _cast_input_hook(module, args):
        x = args[0]
        if x.dtype != module.weight.dtype:
            return (x.to(module.weight.dtype),) + args[1:]
        return args

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, quant_types):
            module.register_forward_pre_hook(_cast_input_hook)

    # Enable input gradients so LoRA can receive gradients
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        # Manual fallback
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


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
        self.lengths = None
        if hasattr(dataset, "features") and length_column in dataset.features:
            try:
                self.lengths = list(dataset[length_column])
            except Exception:
                self.lengths = None

        if self.lengths is None:
            # Fallback: compute lengths from input_ids
            self.lengths = []
            for i in range(len(dataset)):
                item = dataset[i]
                if "input_ids" in item:
                    self.lengths.append(len(item["input_ids"]))
                else:
                    self.lengths.append(512)  # Default estimate

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


class LoggingCallback(TrainerCallback):
    """Enhanced logging callback for mud-puppy."""

    def __init__(self, log_with: str = "none"):
        self.log_with = log_with

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step % 10 == 0:
            loss = logs.get("loss", logs.get("train_loss", "N/A"))
            lr = logs.get("learning_rate", "N/A")
            print(f"[mud-puppy] step {state.global_step}: loss={loss}, lr={lr}")


class MudPuppyTrainer(Trainer):
    """Trainer with optional token-budget dynamic batching and CPU offload."""

    def __init__(self, *args, tokens_per_batch: int = 0,
                 mudpuppy_zero_offload: bool = False, **kwargs):
        self.tokens_per_batch = tokens_per_batch
        self._mudpuppy_zero_offload = mudpuppy_zero_offload
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """Create optimizer, wrapping with CPUOffloadOptimizer when requested."""
        opt = super().create_optimizer()
        if self._mudpuppy_zero_offload:
            from .zero_offload import wrap_optimizer_for_offload
            opt = wrap_optimizer_for_offload(opt)
            print("[mud-puppy] ZeRO-Offload enabled via CPUOffloadOptimizer")
        return opt

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.tokens_per_batch <= 0:
            return super().get_train_dataloader()

        sampler = DynamicBatchSampler(
            dataset=self.train_dataset,
            tokens_per_batch=self.tokens_per_batch,
            length_column="length",
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
        )

        num_workers = self.args.dataloader_num_workers
        extra = {}
        if num_workers > 0:
            extra["persistent_workers"] = True
            extra["prefetch_factor"] = 4

        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            **extra,
        )


def load_model(config: TrainingConfig, calibration_data: Optional[List[torch.Tensor]] = None):
    """Load the base model and tokenizer with optional quantization.

    For ``finetuning_method == 'multimodal'``, this will attempt to load a
    vision+language model via :class:`AutoModelForVision2Seq`.

    Parameters
    ----------
    config:
        Training configuration.
    calibration_data:
        Optional list of input tensors used for GPTQ calibration on ROCm.
        Typically the first 128 batches of the training set, each of shape
        [batch, seq_len] (token ids) or [batch, in_features] for raw layers.
        If None and GPTQ+ROCm is requested, attempts to load a pre-quantized
        checkpoint from config.model_name_or_path instead.
    """
    # Common UX trap: user passes a path ending in a weight-file name
    # (model.safetensors, pytorch_model.bin, etc) instead of the repo
    # directory. HF `from_pretrained` expects a directory containing
    # config.json + tokenizer files, not the weight blob itself. Catch
    # this before we get a cryptic "config.json not found" error.
    _weight_suffixes = (
        ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf",
    )
    mp = str(config.model_name_or_path)
    if os.path.isfile(mp) and mp.lower().endswith(_weight_suffixes):
        parent = os.path.dirname(mp) or "."
        raise RuntimeError(
            f"model_name_or_path points at a weight file ({mp}). "
            f"HuggingFace from_pretrained needs the model DIRECTORY, not "
            f"the weight blob. Try: {parent}"
        )

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

    # Multimodal: load AutoProcessor + AutoModelForVision2Seq
    if config.finetuning_method == "multimodal":
        if AutoModelForVision2Seq is None:
            raise RuntimeError(
                "Multimodal training requires AutoModelForVision2Seq, "
                "which is not available in this version of transformers"
            )

        # Load processor (handles both image and text preprocessing).
        # Fall back to the plain AutoTokenizer when AutoProcessor is not
        # available (older transformers) or when the model has no image
        # processor registered.
        processor = None
        if AutoProcessor is not None:
            try:
                processor = AutoProcessor.from_pretrained(
                    config.model_name_or_path,
                    trust_remote_code=config.trust_remote_code,
                )
                print("[mud-puppy] Loaded AutoProcessor for multimodal model")
            except Exception as proc_err:
                print(
                    f"[mud-puppy] AutoProcessor load failed ({proc_err}); "
                    "falling back to AutoTokenizer"
                )

        if processor is None:
            # Plain tokenizer fallback -- pixel_values will not be produced
            processor = tokenizer
            print("[mud-puppy] Using tokenizer as processor fallback")

        # Ensure pad token is set on the processor's tokenizer
        proc_tok = getattr(processor, "tokenizer", processor)
        if getattr(proc_tok, "pad_token", None) is None:
            proc_tok.pad_token = getattr(proc_tok, "eos_token", None)

        # Replace the returned tokenizer with the full processor so that
        # callers (load_and_preprocess_dataset, run_training) receive it
        tokenizer = processor

        try:
            model_dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            model_dtype = model_dtype_map.get(config.precision, torch.bfloat16)

            model = AutoModelForVision2Seq.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=config.trust_remote_code,
                torch_dtype=model_dtype,
                device_map=None,  # load to CPU first (ROCm segfault avoidance)
                low_cpu_mem_usage=True,
            )
        except OSError as e:
            raise RuntimeError(
                "Failed to load multimodal model; ensure this checkpoint "
                "is compatible with AutoModelForVision2Seq. "
                "Models known to work: LLaVA-1.5, Phi-3-vision, Qwen2-VL, InternVL2. "
                "For models requiring custom code, set trust_remote_code=True."
            ) from e

        # Move to GPU if available and not streaming
        if not config.stream and torch.cuda.is_available():
            print("[mud-puppy] Moving multimodal model to GPU...")
            model = model.to("cuda")

    # QLoRA: load in 16-bit, apply 4-bit quantization, prepare for k-bit training
    elif config.finetuning_method == "qlora":
        model_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16

        # Graceful loader: tier 1 (CausalLM) -> tier 2 (VL-extract) -> tier 3
        # (synthesize from unknown arch). Lets QLoRA work on Mistral3/LLaVA/etc
        # by extracting the text backbone, and on future unreleased models by
        # structural pattern matching.
        from .model_loader import load_model_graceful
        try:
            result = load_model_graceful(
                config.model_name_or_path,
                dtype=model_dtype,
                trust_remote_code=config.trust_remote_code,
                device_map=None,  # load to CPU first (ROCm segfault avoidance)
                low_cpu_mem_usage=True,
            )
            model = result.model
            if result.tier > 1:
                print(f"[mud-puppy] model_loader tier={result.tier}: "
                      + "; ".join(result.notes))
        except (OSError, RuntimeError) as e:
            raise RuntimeError(
                f"Failed to load model from {config.model_name_or_path}: {e}"
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

    # GPTQ: either ROCm-native real GPTQ, or auto-gptq on CUDA
    elif config.finetuning_method == "gptq":
        if is_rocm():
            if calibration_data is not None:
                # Real GPTQ: load fp32 model, run Hessian-calibrated quantization
                from .gptq_rocm import quantize_model_gptq

                print("[mud-puppy] Loading fp32 model for GPTQ calibration...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        config.model_name_or_path,
                        trust_remote_code=config.trust_remote_code,
                        torch_dtype=torch.float32,
                        device_map=None,
                        low_cpu_mem_usage=True,
                    )
                except OSError as e:
                    raise RuntimeError(
                        f"Failed to load model from {config.model_name_or_path}"
                    ) from e

                n_calib = len(calibration_data)
                print(f"[mud-puppy] Running GPTQ with {n_calib} calibration batches...")
                model = quantize_model_gptq(
                    model,
                    calibration_data=calibration_data,
                    bits=4,
                    group_size=config.gptq_group_size,
                    actorder=config.gptq_actorder,
                    damp_percent=config.gptq_damp_percent,
                )
                print("[mud-puppy] GPTQ quantization complete")
            else:
                # Fallback: load pre-quantized model from path
                from .gptq_rocm import load_quantized

                print("[mud-puppy] Loading pre-quantized GPTQ model (no calibration data)...")
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

        # Graceful loader: tier 1 (CausalLM) -> tier 2 (VL-extract) -> tier 3
        # (synthesize). Same rationale as the QLoRA branch: any model the HF
        # ecosystem can load should be trainable, including unreleased ones.
        from .model_loader import load_model_graceful
        try:
            result = load_model_graceful(
                config.model_name_or_path,
                dtype=model_dtype,
                trust_remote_code=config.trust_remote_code,
                device_map=None,
                low_cpu_mem_usage=True,
            )
            model = result.model
            if result.tier > 1:
                print(f"[mud-puppy] model_loader tier={result.tier}: "
                      + "; ".join(result.notes))
        except (OSError, RuntimeError) as e:
            raise RuntimeError(
                f"Failed to load model from {config.model_name_or_path}: {e}"
            ) from e

        # Move to GPU if available and not streaming
        if not config.stream and torch.cuda.is_available():
            print("[mud-puppy] Moving model to GPU...")
            model = model.to("cuda")

    # Streaming: keep model on CPU and stream transformer blocks to GPU via prefetch ring.
    # LayerStreamer raises NotImplementedError for unsafe training methods
    # (anything that updates base-model weights or needs them for backward).
    if config.stream:
        model.to("cpu")
        from .stream import LayerStreamer
        model = LayerStreamer.wrap(
            model,
            prefetch_layers=config.prefetch_layers,
            training_method=config.finetuning_method,
        )
        print(f"[mud-puppy] LayerStreamer active (prefetch_layers={config.prefetch_layers})")

    # FP8 mixed-precision training via per-layer module replacement.
    # This is the delayed-scaling recipe (Transformer Engine style),
    # adapted for ROCm `_scaled_mm`. See mud_puppy/fp8_rocm.py.
    if config.precision == "fp8":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError(
                "FP8 requires a PyTorch build with float8_e4m3fn dtype "
                "(torch >= 2.1, ROCm 6.0+ or CUDA 11.8+)."
            )
        from .fp8_rocm import apply_fp8, is_fp8_hardware_available, fp8_layer_count

        apply_fp8(model)
        n_fp8 = fp8_layer_count(model)
        if is_fp8_hardware_available():
            print(
                f"[mud-puppy] FP8 training enabled on hardware path "
                f"({n_fp8} layers; torch._scaled_mm)"
            )
        else:
            print(
                "[mud-puppy] WARNING: FP8 requested but current GPU has no "
                "hardware FP8 (ROCm requires MI300+/RDNA4, CUDA requires "
                "SM 8.9+). Running emulated cast-dequant path — correct but "
                "not faster than bf16. "
                f"{n_fp8} layers wrapped."
            )

    return model, tokenizer


def _detect_lora_targets(model: nn.Module, default: List[str]) -> List[str]:
    """Pick LoRA target module short-names for a model of unknown family.

    Strategy: walk all nn.Linear modules, collect their short names
    (the last dotted segment), intersect with a curated allow-list of
    attention/MLP projection names seen across every modern transformer
    family (LLaMA, Mistral, Qwen, Phi, Gemma, GPT-2, GPT-NeoX, MPT,
    Falcon, CodeGen, OPT, StableLM, BLOOM, and the VL wrappers of
    those). Fall back to ``default`` if nothing matches.

    Returning too MANY targets is safer than too few: peft silently
    drops names that don't exist in the model, and adding a few extra
    LoRA adapters on present modules is a compute cost, not a
    correctness cost.
    """
    # Union of projection names seen in modern transformer implementations.
    # Expand freely; false positives on unusual models are cheap.
    KNOWN_TARGETS = {
        # LLaMA / Mistral / Qwen / Gemma / Phi3 / Pixtral
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # GPT-2 / GPT-NeoX / MPT family
        "c_attn", "c_proj", "c_fc",
        # GPT-Neo / GPT-J
        "out_proj",
        # Falcon / RefinedWeb
        "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
        # BLOOM
        "self_attention.query_key_value",
        # T5-family (just in case)
        "q", "k", "v", "o", "wi", "wo", "wi_0", "wi_1",
        # MPT
        "Wqkv", "out_proj",
        # StableLM, CodeGen
        "qkv_proj", "attn_out",
    }

    present_shortnames: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        short = name.rsplit(".", 1)[-1]
        if short in KNOWN_TARGETS:
            present_shortnames.add(short)

    if not present_shortnames:
        # No curated names matched. Second-pass heuristic: any Linear
        # whose parent path contains "attn"/"attention"/"mlp" is a
        # plausible target.
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            lname = name.lower()
            if any(k in lname for k in ("attn", "attention", "mlp",
                                         "feed_forward", "ffn")):
                short = name.rsplit(".", 1)[-1]
                # Skip generic short names like "linear" that collide
                # with unrelated layers.
                if short and short not in {"linear", "fc", "classifier"}:
                    present_shortnames.add(short)

    if not present_shortnames:
        print(f"[mud-puppy] Warning: no LoRA targets detected, using default: {default}")
        return list(default)

    # Sort for stability (so log lines are deterministic across runs).
    return sorted(present_shortnames)


def prepare_lora(model: nn.Module, config: TrainingConfig) -> nn.Module:
    """Attach LoRA/QLoRA adapters to the model."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise RuntimeError("peft is required for LoRA training") from e

    # Auto-detect target modules based on model architecture. Ordered
    # from most-specific to least; first hit wins.
    target_modules = config.lora_target_modules
    target_modules = _detect_lora_targets(model, default=target_modules)
    print(f"[mud-puppy] LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
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

    # -------------------- FSDP wiring --------------------
    # HuggingFace TrainingArguments accepts ``fsdp`` as a string or list
    # of tokens: {"full_shard", "shard_grad_op", "no_shard", "hybrid_shard",
    # "offload", "auto_wrap"}. We compose it from the dedicated config
    # fields so users don't have to memorize the token soup.
    fsdp_tokens: List[str] = []
    fsdp_config_dict: Dict[str, Any] = {}
    if config.fsdp_mode:
        fsdp_tokens.append(config.fsdp_mode)
        fsdp_tokens.append("auto_wrap")
        if config.fsdp_cpu_offload:
            fsdp_tokens.append("offload")
        if config.fsdp_transformer_layer_cls:
            # Class-based wrap: one FSDP unit per transformer block. This
            # is the standard pattern for LLM training and tends to give
            # much better memory behavior than min-num-params wrapping
            # because it aligns shard boundaries with activation
            # recomputation boundaries.
            fsdp_config_dict["transformer_layer_cls_to_wrap"] = [
                config.fsdp_transformer_layer_cls
            ]
        else:
            fsdp_config_dict["min_num_params"] = config.fsdp_min_num_params
        if config.fsdp_activation_checkpointing:
            fsdp_config_dict["activation_checkpointing"] = True
        # Keep the param+grad flat in bf16 on MI300 to match amp dtype and
        # let RCCL push bigger bf16 buckets. Users can still override via
        # env.
        fsdp_config_dict.setdefault(
            "backward_prefetch", "backward_pre"
        )
        fsdp_config_dict.setdefault(
            "sharding_strategy", config.fsdp_mode.upper()
        )
        fsdp_config_dict.setdefault("use_orig_params", True)

    fsdp_str = " ".join(fsdp_tokens) if fsdp_tokens else ""

    kwargs: Dict[str, Any] = dict(
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
        save_strategy=getattr(config, "save_strategy", "epoch"),
        save_steps=getattr(config, "save_steps", 500),
        logging_strategy="steps",
        logging_steps=getattr(config, "logging_steps", 10),
        report_to=report_to,
        save_total_limit=2,
        load_best_model_at_end=config.early_stopping_patience > 0,
        metric_for_best_model="eval_loss" if config.early_stopping_patience > 0 else None,
        max_grad_norm=config.max_grad_norm,
        local_rank=config.local_rank,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    if fsdp_str:
        kwargs["fsdp"] = fsdp_str
        kwargs["fsdp_config"] = fsdp_config_dict
    return TrainingArguments(**kwargs)


def load_and_preprocess_dataset(
    config: TrainingConfig, tokenizer: PreTrainedTokenizer
) -> Dataset:
    """Load and preprocess the training dataset.

    Supports JSONL format with chat-style messages or simple text fields.

    For ``finetuning_method == "multimodal"``, the ``tokenizer`` argument is
    expected to be an ``AutoProcessor`` (as returned by ``load_model`` in the
    multimodal branch). Image paths are preserved in the dataset; images are
    loaded lazily by the ``MultimodalCollator`` during training.
    """
    # Multimodal path: delegate to the dedicated multimodal tokenizer
    if config.finetuning_method == "multimodal":
        from .multimodal import tokenize_multimodal_dataset

        import os as _os
        base_dir = _os.path.dirname(_os.path.abspath(config.dataset_path))

        if config.max_seq_length > 0:
            max_len = config.max_seq_length
        else:
            # Try to get max length from the processor's tokenizer
            proc_tok = getattr(tokenizer, "tokenizer", tokenizer)
            max_len = min(getattr(proc_tok, "model_max_length", 2048), 2048)

        print(f"[mud-puppy] Tokenizing multimodal dataset (max_length={max_len})...")
        return tokenize_multimodal_dataset(
            dataset_path=config.dataset_path,
            processor=tokenizer,
            max_length=max_len,
            base_dir=base_dir,
            preprocessing_workers=config.preprocessing_workers,
        )

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
            padding=False,
            max_length=max_len,
            return_tensors=None,
        )

        # Add length column for dynamic batching
        tokenized["length"] = [len(ids) for ids in tokenized["input_ids"]]

        # Populate labels for causal LM training. PackedCollator (and the
        # standard data collator without label shifting) requires labels to
        # exist up-front; HuggingFace's Trainer shifts internally at loss
        # time, so we clone input_ids here and let the model handle the
        # shift. Downstream masking (response-only, prompt masking) can
        # overwrite these values with -100 as needed.
        tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]

        return tokenized

    # Apply tokenization
    dataset = dataset.map(
        tokenize_chat,
        batched=True,
        num_proc=config.preprocessing_workers,
        remove_columns=columns,
        desc="Tokenizing dataset",
    )

    # Set format for PyTorch. Retain "labels" (required by collators +
    # PackedCollator) and "length" (required by DynamicBatchSampler when
    # tokens_per_batch > 0).
    format_columns = ["input_ids", "attention_mask", "labels"]
    if getattr(config, "tokens_per_batch", 0) > 0:
        format_columns.append("length")
    dataset.set_format(type="torch", columns=format_columns)

    return dataset


def setup_distributed(config: TrainingConfig) -> bool:
    """Initialize distributed training if requested.

    On ROCm the ``nccl`` backend is transparently RCCL; PyTorch picks the
    right one based on the build. For MI300 nodes we additionally set a
    small number of env var defaults that materially help performance on
    xGMI fabric; users can override by exporting the same variables
    before invoking the launcher.
    """
    # Auto-promote to distributed when torchrun provided a world size.
    # This lets ``mud-puppy`` scripts that don't know about --distributed
    # still do the right thing behind torchrun.
    world_size = int(os.environ.get("WORLD_SIZE") or "1")
    if world_size > 1 and not config.distributed:
        log.info("WORLD_SIZE=%d detected; enabling distributed", world_size)
        config.distributed = True

    if not config.distributed:
        return False

    # Guard: if the user asked for distributed but there's no launcher env,
    # ``init_process_group`` would otherwise block forever on rendezvous.
    # Accept either torchrun's WORLD_SIZE>1 OR explicit MASTER_ADDR+MASTER_PORT
    # (for static init), and otherwise fail with a clear message.
    if not dist.is_initialized():
        has_torchrun = world_size > 1
        has_static = bool(os.environ.get("MASTER_ADDR")) and bool(
            os.environ.get("MASTER_PORT")
        )
        if not (has_torchrun or has_static):
            raise RuntimeError(
                "--distributed requires torchrun or MASTER_ADDR/MASTER_PORT; "
                "see scripts/mud-puppy-launch"
            )

    # CDNA-specific tuning. All of these are overridable via the shell.
    from .arch import get_arch

    # Touch the device first so get_arch can actually read gcnArchName.
    # One set_device call, placed after the guard so we don't pin a device
    # for a run that's about to error out.
    if torch.cuda.is_available():
        torch.cuda.set_device(config.local_rank)
    info = get_arch(config.local_rank) if torch.cuda.is_available() else None
    if info is not None and info.is_cdna:
        # RCCL on MI300 xGMI benefits from these defaults. They're
        # ignored on other hardware.
        os.environ.setdefault("NCCL_PROTO", "Simple,LL,LL128")
        os.environ.setdefault("NCCL_ALGO", "Tree,Ring")
        # P2P over xGMI; falls back automatically if links are absent.
        os.environ.setdefault("NCCL_P2P_DISABLE", "0")
        # RCCL-specific: rely on HSA_OVERRIDE only when user sets it.
        os.environ.setdefault("RCCL_MSCCL_ENABLE", "1")

    if not dist.is_initialized():
        dist.init_process_group(backend=config.distributed_backend)

    rank = dist.get_rank() if dist.is_initialized() else 0
    log.info(
        "distributed init: rank=%d world=%d backend=%s arch=%s",
        rank, world_size, config.distributed_backend,
        info.family.value if info else "cpu",
    )
    return True


def setup_pipeline_parallelism(model: nn.Module, config: TrainingConfig) -> nn.Module:
    """Set up pipeline parallelism if requested.

    .. deprecated::
        The ``torch.distributed.pipeline.sync.Pipe`` API this helper wraps
        is deprecated upstream in PyTorch 2.4+ (replaced by
        ``torch.distributed.pipelining``). Single-GPU is the primary
        target for mud-puppy on the 7900 XTX; real multi-GPU pipeline
        parallelism will be re-implemented against the new API in a
        future release.

    Current behavior is a hard fail if the host has fewer than 2 GPUs
    or if the deprecated API is not importable. This replaces the
    previous silent-fallback path that looked like it was doing
    something when it wasn't.
    """
    if config.device_map != "pipeline":
        return model

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        raise RuntimeError(
            "Pipeline parallelism requires >=2 GPUs; detected "
            f"{num_gpus}. On single-GPU setups use --stream for layer "
            "streaming instead."
        )

    if Pipe is None:
        raise RuntimeError(
            "Pipeline parallelism requires torch.distributed.pipeline.sync.Pipe "
            "which is not available in this PyTorch build. This API is also "
            "deprecated upstream in torch>=2.4; support for the new "
            "torch.distributed.pipelining will land in a future mud-puppy "
            "release."
        )

    warnings.warn(
        "setup_pipeline_parallelism uses the deprecated "
        "torch.distributed.pipeline.sync.Pipe API. It works on torch<2.4 but "
        "will be removed when PyTorch drops the old API. Prefer LayerStreamer "
        "(--stream) on single-GPU or wait for the torch.distributed.pipelining "
        "port.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Find the layer list.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise RuntimeError(
            "Pipeline parallelism: could not identify transformer block "
            "list (looked for model.model.layers and model.transformer.h). "
            "Architecture not supported by this path."
        )

    layers_per_gpu = len(layers) // num_gpus
    chunks = []
    for i in range(num_gpus):
        start_idx = i * layers_per_gpu
        end_idx = start_idx + layers_per_gpu if i < num_gpus - 1 else len(layers)
        chunk_layers = nn.Sequential(*[layers[j] for j in range(start_idx, end_idx)])
        chunks.append(chunk_layers.to(f"cuda:{i}"))

    pipe_model = Pipe(nn.Sequential(*chunks), chunks=num_gpus)
    print(f"[mud-puppy] Pipeline parallelism enabled across {num_gpus} GPUs "
          "(DEPRECATED API path)")
    return pipe_model


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

    # Set up distributed training if requested
    is_distributed = setup_distributed(config)
    if is_distributed:
        print(f"[mud-puppy] Distributed training enabled (rank {config.local_rank})")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Streaming + gradient_checkpointing is unsafe. During backward,
    # checkpoint recomputation re-invokes the forward hooks installed by
    # LayerStreamer; a concurrent prefetch/eviction can rip weights out
    # from under an in-flight recomputation. Fail fast rather than silently
    # training on corrupted gradients.
    if config.stream and config.use_gradient_checkpointing:
        raise RuntimeError(
            "LayerStreamer is incompatible with gradient checkpointing "
            "(recomputation races against ring eviction). "
            "Disable --use-gradient-checkpointing or --stream."
        )

    # Load model and tokenizer
    model, tokenizer = load_model(config)

    # Enable gradient checkpointing BEFORE LoRA wrapping.
    # PEFT's forward hooks need gradient checkpointing already active so that
    # checkpoint boundaries are correctly set inside the base model layers.
    if config.use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Apply LoRA if requested
    if config.finetuning_method in ("lora", "qlora"):
        model = prepare_lora(model, config)

    # Set up pipeline parallelism if requested
    if config.device_map == "pipeline":
        model = setup_pipeline_parallelism(model, config)

    # Compile model if requested (skip when --stream is active: custom hooks don't mesh with compile)
    if config.compile and hasattr(torch, "compile") and not config.stream:
        print(f"[mud-puppy] Compiling model with torch.compile (mode={config.compile_mode})...")
        model = torch.compile(model, mode=config.compile_mode)
    elif config.compile and config.stream:
        log.warning("torch.compile disabled when --stream is active")

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
    if config.finetuning_method == "multimodal":
        from .multimodal import MultimodalCollator
        import os as _os

        max_seq = config.max_seq_length if config.max_seq_length > 0 else 2048
        base_dir = _os.path.dirname(_os.path.abspath(config.dataset_path))

        # Determine pad token id from the processor's tokenizer or the tokenizer itself
        proc_tok = getattr(tokenizer, "tokenizer", tokenizer)
        pad_id = getattr(proc_tok, "pad_token_id", None) or 0

        data_collator = MultimodalCollator(
            processor=tokenizer,
            tokenizer=proc_tok,
            max_length=max_seq,
            pad_token_id=pad_id,
            base_dir=base_dir,
        )
        print("[mud-puppy] Using MultimodalCollator (image + text collation)")

    elif config.pack_sequences:
        from .packing import PackedCollator
        max_seq = config.max_seq_length if config.max_seq_length > 0 else 2048
        data_collator = PackedCollator(
            max_seq_length=max_seq,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
        print(f"[mud-puppy] Sequence packing enabled (max_seq_length={max_seq})")
    else:
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

    # zero_offload is wired via MudPuppyTrainer.create_optimizer, not a callback

    # QAT: recalibrate per-channel scales periodically during training so
    # the fake-quant range tracks weight distribution shifts.
    if config.finetuning_method == "qat":
        from .qat_rocm import QATScaleCallback

        callbacks.append(QATScaleCallback(interval=50, momentum=0.01))

    # Monitor callback (web dashboard via WebSocket)
    monitor_server = None
    if config.monitor:
        from .monitor import MonitorCallback, MonitorServer, start_gpu_telemetry

        config_data = {
            "model": config.model_name_or_path,
            "method": config.finetuning_method,
            "precision": config.precision,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "dataset_size": len(dataset),
            "lora_r": config.lora_r if config.finetuning_method in ("lora", "qlora") else None,
            "lora_alpha": config.lora_alpha if config.finetuning_method in ("lora", "qlora") else None,
            "quant_backend": config.quant_backend if config.finetuning_method == "qlora" else None,
        }

        monitor_server = MonitorServer(port=config.monitor_port)
        monitor_server.start()
        start_gpu_telemetry(monitor_server)
        # Publish the actual bound port on stdout as a single line so the
        # Phos shell (and any other parent process) can parse it without
        # needing to pre-allocate a port (avoids the bind(:0) TOCTOU race).
        bound = monitor_server.port
        print(f"MUD_PUPPY_MONITOR_PORT={bound}", flush=True)
        print(f"[mud-puppy] Training monitor: http://localhost:{bound}")

        callbacks.append(MonitorCallback(
            model=model,
            config_data=config_data,
            server=monitor_server,
            tui=None,
            lora_norm_interval=50 if config.finetuning_method in ("lora", "qlora") else 0,
        ))

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

    if config.tokens_per_batch > 0:
        print(
            f"[mud-puppy] Dynamic batching enabled: ~{config.tokens_per_batch} tokens/batch"
        )

    trainer = MudPuppyTrainer(
        tokens_per_batch=config.tokens_per_batch,
        mudpuppy_zero_offload=config.zero_offload,
        **trainer_kwargs,
    )

    # Resume from checkpoint if requested
    resume_from = None
    if config.resume:
        checkpoints = [
            d for d in os.listdir(config.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            def _checkpoint_step(name: str) -> int:
                parts = name.split("-", 1)
                if len(parts) != 2 or not parts[1].isdigit():
                    return -1  # skip non-numeric checkpoints (e.g. checkpoint-best)
                return int(parts[1])
            numeric_ckpts = [d for d in checkpoints if _checkpoint_step(d) >= 0]
            if numeric_ckpts:
                latest = max(numeric_ckpts, key=_checkpoint_step)
                resume_from = os.path.join(config.output_dir, latest)
                print(f"[mud-puppy] Resuming from {resume_from}")

    # Collect calibration batches for GPTQ before training consumes the loader.
    calibration_data: List[torch.Tensor] = []
    if config.finetuning_method == "gptq":
        loader = trainer.get_train_dataloader()
        for i, batch in enumerate(loader):
            if i >= 32:
                break
            calibration_data.append(batch["input_ids"])

    # Run training
    print("[mud-puppy] Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)

    # Stop monitor
    if monitor_server:
        monitor_server.stop()

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
        model = quantize_model_gptq(model, calibration_data, bits=4)
        gptq_path = os.path.join(config.output_dir, "gptq")
        os.makedirs(gptq_path, exist_ok=True)
        save_quantized(model, gptq_path)
        tokenizer.save_pretrained(gptq_path)
        print(f"[mud-puppy] GPTQ model saved to {gptq_path}")

    # Merge LoRA weights if requested
    if config.merge_lora and config.finetuning_method in ("lora", "qlora"):
        model = merge_lora_weights(model, config, tokenizer)

    print("[mud-puppy] Training complete!")
