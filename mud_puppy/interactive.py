BANNER = (
"##::::'##:'##::::'##:'########:::::'########::'##::::'##:'########::'########::'##:::'##:\n"
" ###::'###: ##:::: ##: ##.... ##:::: ##.... ##: ##:::: ##: ##.... ##: ##.... ##:. ##:'##::\n"
" ####'####: ##:::: ##: ##:::: ##:::: ##:::: ##: ##:::: ##: ##:::: ##: ##:::: ##::. ####:::\n"
" ## ### ##: ##:::: ##: ##:::: ##:::: ########:: ##:::: ##: ########:: ########::::. ##:::\n"
" ##. #: ##: ##:::: ##: ##:::: ##:::: ##.....::: ##:::: ##: ##.....::: ##.....:::::: ##:::\n"
" ##:.:: ##: ##:::: ##: ##:::: ##:::: ##:::::::: ##:::: ##: ##:::::::: ##::::::::::: ##:::\n"
" ##:::: ##:. #######:: ########::::: ##::::::::. #######:: ##:::::::: ##::::::::::: ##:::\n"
"..:::::..:::.......:::........::::::..::::::::::.......:::..:::::::::..::::::::::::..:::::\n"
)


from .config import TrainingConfig


def prompt(message: str, default: str | None = None) -> str:
    """Prompt the user for input with an optional default."""
    if default is not None:
        message = f"{message} [{default}]"
    value = input(message + ": ")
    if not value and default is not None:
        value = default
    return value


def configure_training() -> TrainingConfig:
    """Interactively gather a TrainingConfig."""
    model = prompt("Model name or path")
    dataset = prompt("Dataset path")
    output = prompt("Output directory", "./outputs")
    method = prompt(
        "Finetuning method (full, lora, qlora, gptq, qat, preference, rl, multimodal, rm, prm)",
        "full",
    )
    precision = prompt("Precision (fp16, bf16, fp8)", "bf16")
    return TrainingConfig(
        model_name_or_path=model,
        dataset_path=dataset,
        output_dir=output,
        finetuning_method=method,
        precision=precision,
    )


def run(config: TrainingConfig):
    """Dispatch to the correct training routine based on method."""
    if config.finetuning_method == "preference":
        from .preference import run_preference_training

        config.preference = prompt("Preference method", "dpo")
        run_preference_training(config)
    elif config.finetuning_method == "rl":
        from .rl import run_grpo_training

        run_grpo_training(config)
    elif config.finetuning_method in {"rm", "prm"}:
        from .reward import train_reward_model

        train_reward_model(config)
    else:
        from .trainer import run_training


        run_training(config)


def main() -> None:
    print(BANNER)
    print("Type 'start' to configure a run or 'quit' to exit.")
    while True:
        try:
            cmd = input("mud-puppy> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cmd in {"quit", "exit"}:
            break
        if cmd == "start":
            try:
                config = configure_training()
                run(config)
            except Exception as e:  # pragma: no cover - interactive errors
                print(f"Error: {e}")
        elif cmd:
            print("Unknown command. Type 'start' or 'quit'.")


if __name__ == "__main__":
    main()
