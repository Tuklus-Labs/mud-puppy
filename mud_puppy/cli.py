import argparse
import os

from .config import TrainingConfig
from .trainer import run_training


def build_parser() -> argparse.ArgumentParser:
    """Build the mud-puppy CLI argument parser.

    This is factored out so it can be reused in tests or tooling.
    """
    parser = argparse.ArgumentParser(description="mud-puppy: ROCm-first fine tuning")
    parser.add_argument("model", help="base model name or path")
    parser.add_argument("dataset", help="dataset path (JSONL file)")
    parser.add_argument(
        "--method",
        default="full",
        dest="method",
        help=(
            "finetuning method: full, lora, qlora, gptq, qat, "
            "preference, rl, multimodal, rm, prm"
        ),
    )
    parser.add_argument(
        "--output", default="./outputs", dest="output", help="output directory"
    )
    parser.add_argument(
        "--quant-backend",
        dest="quant_backend",
        default="int4",
        choices=["int4", "mxfp4"],
        help="quantization backend for qlora: int4 (row-wise) or mxfp4 (block-wise)",
    )
    parser.add_argument(
        "--preference",
        dest="preference",
        help="preference method (e.g. dpo, ipo, kto, orpo)",
    )
    parser.add_argument(
        "--precision",
        dest="precision",
        default="bf16",
        choices=["fp16", "bf16", "fp8"],
        help="training precision",
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        help="enable torch.compile for extra speed",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=0,
        help="dataloader workers",
    )
    parser.add_argument(
        "--preprocess-workers",
        dest="preprocess_workers",
        type=int,
        default=1,
        help="dataset preprocessing workers",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        dest="trust_remote_code",
        help="allow loading models with custom code",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        dest="no_chat_template",
        help="disable tokenizer chat template application",
    )
    parser.add_argument(
        "--lora-targets",
        dest="lora_targets",
        help="comma separated list of LoRA target modules",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        dest="resume",
        help="resume training from the last checkpoint",
    )
    parser.add_argument(
        "--log-with",
        dest="log_with",
        default="none",
        choices=["none", "tensorboard", "wandb"],
        help="logging backend",
    )
    parser.add_argument(
        "--tokens-per-batch",
        dest="tokens_per_batch",
        type=int,
        default=0,
        help="enable dynamic batching with this many tokens",
    )
    parser.add_argument(
        "--lr-scheduler",
        dest="lr_scheduler",
        default="linear",
        help="learning rate scheduler type",
    )
    parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        type=int,
        default=0,
        help="stop training if no improvement for N evals",
    )
    parser.add_argument(
        "--device-map",
        dest="device_map",
        default="auto",
        help="model device map for model parallelism (auto, pipeline, or map)",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        help="load model into swap and stream layers to the GPU",
    )
    parser.add_argument(
        "--zero-offload",
        dest="zero_offload",
        action="store_true",
        help="offload optimizer states to CPU memory",
    )
    parser.add_argument(
        "--merge-lora",
        dest="merge_lora",
        action="store_true",
        help="merge LoRA weights into the base model when training completes",
    )
    parser.add_argument(
        "--merge-precision",
        dest="merge_precision",
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="precision for the merged model weights",
    )
    parser.add_argument("--monitor", dest="monitor", action="store_true",
        help="enable real-time web training dashboard (port 5980)")
    parser.add_argument("--monitor-tui", dest="monitor_tui", action="store_true",
        help="enable terminal (Rich) training monitor")
    parser.add_argument("--monitor-port", dest="monitor_port", type=int, default=5980,
        help="port for web training monitor (default: 5980)")
    parser.add_argument(
        "--distributed",
        dest="distributed",
        action="store_true",
        help="enable torch.distributed training",
    )
    parser.add_argument(
        "--local-rank",
        dest="local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="rank of this process for distributed training",
    )

    # Optional training hyperparameters
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="per-device batch size (overrides config default)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        dest="gradient_accumulation",
        type=int,
        help="gradient accumulation steps (overrides config default)",
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        help="learning rate (overrides config default)",
    )
    parser.add_argument(
        "--epochs",
        dest="num_epochs",
        type=int,
        help="number of training epochs (overrides config default)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_kwargs = dict(
        model_name_or_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        finetuning_method=args.method,
        precision=args.precision,
        preference=args.preference,
        compile=args.compile,
        dataloader_workers=args.num_workers,
        preprocessing_workers=args.preprocess_workers,
        trust_remote_code=args.trust_remote_code,
        use_chat_template=not args.no_chat_template,
        resume=args.resume,
        log_with=args.log_with,
        tokens_per_batch=args.tokens_per_batch,
        lr_scheduler=args.lr_scheduler,
        early_stopping_patience=args.early_stopping,
        device_map=args.device_map,
        stream=args.stream,
        zero_offload=args.zero_offload,
        merge_lora=args.merge_lora,
        merge_precision=args.merge_precision,
        monitor=args.monitor,
        monitor_tui=args.monitor_tui,
        monitor_port=args.monitor_port,
        distributed=args.distributed,
        local_rank=args.local_rank,
        quant_backend=args.quant_backend,
    )

    # Optional training hyperparameters from CLI override dataclass defaults
    if args.batch_size is not None:
        config_kwargs["batch_size"] = args.batch_size
    if args.gradient_accumulation is not None:
        config_kwargs["gradient_accumulation"] = args.gradient_accumulation
    if args.learning_rate is not None:
        config_kwargs["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        config_kwargs["num_epochs"] = args.num_epochs

    if args.lora_targets:
        config_kwargs["lora_target_modules"] = args.lora_targets.split(",")

    config = TrainingConfig(**config_kwargs)

    if config.finetuning_method == "preference":
        from .preference import run_preference_training

        run_preference_training(config)
    elif config.finetuning_method == "rl":
        from .rl import run_grpo_training

        run_grpo_training(config)
    elif config.finetuning_method in {"rm", "prm"}:
        from .reward import train_reward_model

        train_reward_model(config)
    else:
        run_training(config)


if __name__ == "__main__":
    main()
