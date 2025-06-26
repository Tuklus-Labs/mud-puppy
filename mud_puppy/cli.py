import argparse
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
