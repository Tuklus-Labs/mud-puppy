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
    parser.add_argument("--method", default="full", dest="method", help="finetuning method")
    parser.add_argument("--output", default="./outputs", dest="output", help="output directory")
    parser.add_argument("--preference", dest="preference", help="preference method")
    parser.add_argument(
        "--precision",
        dest="precision",
        default="bf16",
        choices=["fp16", "bf16", "fp8"],
        help="training precision",
    )

    args = parser.parse_args()

    config = TrainingConfig(
        model_name_or_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        finetuning_method=args.method,
        precision=args.precision,
        preference=args.preference,
    )

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

