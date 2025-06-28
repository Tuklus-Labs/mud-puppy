import argparse
import os
from .config import TrainingConfig
from .trainer import run_training
from .preference import run_preference_training
from .rl import run_grpo_training
from .reward import train_reward_model


def main():
    parser = argparse.ArgumentParser(description="mud-puppy: ROCm-first fine tuning")
    parser.add_argument("model", help="base model name or path")
    parser.add_argument("dataset", help="dataset path")
    parser.add_argument(
        "--method", default="full", dest="method", help="finetuning method"
    )
    parser.add_argument(
        "--output", default="./outputs", dest="output", help="output directory"
    )
    parser.add_argument("--preference", dest="preference", help="preference method")
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
        default="tensorboard",
        choices=["tensorboard", "wandb"],
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
        help="model device map for model parallelism",
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
        distributed=args.distributed,
        local_rank=args.local_rank,
    )
    if args.lora_targets:
        config_kwargs["lora_target_modules"] = args.lora_targets.split(",")

    config = TrainingConfig(**config_kwargs)

    if config.finetuning_method == "preference":
        run_preference_training(config)
    elif config.finetuning_method == "rl":
        run_grpo_training(config)
    elif config.finetuning_method in {"rm", "prm"}:
        train_reward_model(config)
    else:
        run_training(config)


if __name__ == "__main__":
    main()
