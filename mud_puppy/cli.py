import argparse
import os

from .config import TrainingConfig, _env_int
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
        choices=["fp16", "bf16", "fp8", "fp32"],
        help="training precision",
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        help="enable torch.compile for extra speed",
    )
    parser.add_argument(
        "--compile-mode",
        dest="compile_mode",
        default="reduce-overhead",
        choices=["reduce-overhead", "default", "max-autotune"],
        help="torch.compile mode (default: reduce-overhead)",
    )
    parser.add_argument(
        "--pack-sequences",
        dest="pack_sequences",
        action="store_true",
        help="enable sequence packing (bin-packs short examples into rows, "
             "uses block-diagonal attention mask)",
    )
    parser.add_argument(
        "--prefetch-layers",
        dest="prefetch_layers",
        type=int,
        default=2,
        help="number of transformer layers to keep resident in GPU ring "
             "when --stream is active (default: 2)",
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
        "--max-seq-length",
        dest="max_seq_length",
        type=int,
        default=0,
        help="max sequence length (0 uses tokenizer default, capped at 2048)",
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
        "--lora-r",
        dest="lora_r",
        type=int,
        help="LoRA rank (overrides config default)",
    )
    parser.add_argument(
        "--lora-alpha",
        dest="lora_alpha",
        type=int,
        help="LoRA alpha (overrides config default)",
    )
    parser.add_argument(
        "--lora-dropout",
        dest="lora_dropout",
        type=float,
        help="LoRA dropout (overrides config default)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="no_gradient_checkpointing",
        action="store_true",
        help="disable gradient checkpointing",
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
    # ------------------- GGUF export + kernel-anvil -------------------
    parser.add_argument(
        "--export-gguf",
        dest="export_gguf",
        action="store_true",
        help="after training, convert the output to GGUF (llama.cpp format). "
             "Runs LoRA merge if needed, then convert_hf_to_gguf.py, "
             "optionally llama-quantize, and optionally kernel-anvil "
             "gguf-optimize for per-shape kernel tuning.",
    )
    parser.add_argument(
        "--gguf-quant",
        dest="gguf_quant",
        default="Q4_K_M",
        help="GGUF quantization type for --export-gguf (default: Q4_K_M). "
             "Empty string keeps fp16. Accepts any llama-quantize type.",
    )
    parser.add_argument(
        "--no-kernel-anvil",
        dest="optimize_with_kernel_anvil",
        action="store_false",
        default=True,
        help="skip the kernel-anvil gguf-optimize step after --export-gguf",
    )
    parser.add_argument(
        "--gptq-group-size",
        dest="gptq_group_size",
        type=int,
        default=128,
        help="GPTQ group size for per-group quantization (default: 128)",
    )
    parser.add_argument(
        "--gptq-actorder",
        dest="gptq_actorder",
        action="store_true",
        default=True,
        help="enable activation-order reordering in GPTQ (default: on)",
    )
    parser.add_argument(
        "--no-gptq-actorder",
        dest="gptq_actorder",
        action="store_false",
        help="disable GPTQ activation-order reordering",
    )
    parser.add_argument(
        "--gptq-damp-percent",
        dest="gptq_damp_percent",
        type=float,
        default=0.01,
        help="GPTQ Hessian damping percentage, must be in (0, 1) (default: 0.01)",
    )
    # ------------------- checkpoint cadence -----------------------
    parser.add_argument(
        "--save-strategy",
        dest="save_strategy",
        default="epoch",
        choices=["epoch", "steps", "no"],
        help="HuggingFace Trainer save_strategy (default: epoch). "
             "Use 'steps' for mid-run checkpoints on long runs.",
    )
    parser.add_argument(
        "--save-steps",
        dest="save_steps",
        type=int,
        default=500,
        help="save_steps value when --save-strategy=steps (default 500)",
    )
    parser.add_argument(
        "--logging-steps",
        dest="logging_steps",
        type=int,
        default=10,
        help="HF Trainer logging_steps (default 10)",
    )

    # ------------------- Heretic (abliteration) --------------------
    parser.add_argument(
        "--heretic",
        dest="heretic",
        action="store_true",
        help="after training (and LoRA merge if applicable), run heretic "
             "(p-e-w/heretic-llm) to orthogonalize the refusal direction out "
             "of the model. Output goes to <output>/heretic/ and becomes the "
             "input to --export-gguf if both are set.",
    )
    parser.add_argument(
        "--heretic-n-trials",
        dest="heretic_n_trials",
        type=int,
        default=30,
        help="Optuna trial count for heretic (default 30)",
    )
    parser.add_argument(
        "--heretic-quantization",
        dest="heretic_quantization",
        default="BNB_4BIT",
        choices=["NONE", "BNB_4BIT"],
        help="heretic quantization for the analysis pass. "
             "BNB_4BIT is required for 14B+ models on consumer 24GB GPUs.",
    )
    parser.add_argument(
        "--heretic-good-prompts-dataset",
        dest="heretic_good_prompts_dataset",
        default=None,
        help="override heretic good-prompt pool (HF dataset id)",
    )
    parser.add_argument(
        "--heretic-bad-prompts-dataset",
        dest="heretic_bad_prompts_dataset",
        default=None,
        help="override heretic bad-prompt pool (HF dataset id)",
    )
    parser.add_argument(
        "--heretic-system-prompt",
        dest="heretic_system_prompt",
        default=None,
        help="explicit system prompt during refusal analysis",
    )
    parser.add_argument(
        "--heretic-extra",
        dest="heretic_extra",
        default="",
        help="free-form pass-through to heretic CLI (space-separated, "
             "use quotes for values with spaces)",
    )
    parser.add_argument("--monitor", dest="monitor", action="store_true",
        help="enable real-time web training dashboard (port 5980)")
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
        default=_env_int("LOCAL_RANK", 0),
        help="rank of this process for distributed training",
    )
    parser.add_argument(
        "--distributed-backend",
        dest="distributed_backend",
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="distributed backend; nccl is RCCL on ROCm (default: nccl)",
    )
    # FSDP (multi-GPU, CDNA / MI300 path)
    parser.add_argument(
        "--fsdp",
        dest="fsdp_mode",
        default="",
        choices=["", "full_shard", "shard_grad_op", "no_shard", "hybrid_shard"],
        help=(
            "enable Fully Sharded Data Parallel; 'full_shard' shards "
            "params+grads+optimizer across all GPUs (recommended for 70B+ "
            "on MI300x8); 'hybrid_shard' shards within a node and "
            "replicates across nodes (multi-node only)"
        ),
    )
    parser.add_argument(
        "--fsdp-cpu-offload",
        dest="fsdp_cpu_offload",
        action="store_true",
        help="offload sharded params to CPU (rarely needed on MI300/192GB)",
    )
    parser.add_argument(
        "--fsdp-activation-checkpointing",
        dest="fsdp_activation_checkpointing",
        action="store_true",
        help="checkpoint activations inside FSDP units (complements gradient_checkpointing)",
    )
    parser.add_argument(
        "--fsdp-min-num-params",
        dest="fsdp_min_num_params",
        type=int,
        default=1_000_000,
        help="auto-wrap threshold when --fsdp-wrap-class is not set",
    )
    parser.add_argument(
        "--fsdp-wrap-class",
        dest="fsdp_transformer_layer_cls",
        default="",
        help=(
            "wrap each instance of this class as one FSDP unit, e.g. "
            "LlamaDecoderLayer, Qwen2DecoderLayer, MixtralDecoderLayer. "
            "Class-based wrap is the right pattern for transformer LLMs; "
            "min-num-params is a fallback for unfamiliar architectures."
        ),
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
        compile_mode=args.compile_mode,
        pack_sequences=args.pack_sequences,
        prefetch_layers=args.prefetch_layers,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        dataloader_workers=args.num_workers,
        preprocessing_workers=args.preprocess_workers,
        max_seq_length=args.max_seq_length,
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
        monitor_port=args.monitor_port,
        distributed=args.distributed,
        local_rank=args.local_rank,
        distributed_backend=args.distributed_backend,
        fsdp_mode=args.fsdp_mode,
        fsdp_cpu_offload=args.fsdp_cpu_offload,
        fsdp_activation_checkpointing=args.fsdp_activation_checkpointing,
        fsdp_min_num_params=args.fsdp_min_num_params,
        fsdp_transformer_layer_cls=args.fsdp_transformer_layer_cls,
        quant_backend=args.quant_backend,
        gptq_group_size=args.gptq_group_size,
        gptq_actorder=args.gptq_actorder,
        gptq_damp_percent=args.gptq_damp_percent,
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
    if args.lora_r is not None:
        config_kwargs["lora_r"] = args.lora_r
    if args.lora_alpha is not None:
        config_kwargs["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        config_kwargs["lora_dropout"] = args.lora_dropout

    if args.lora_targets:
        config_kwargs["lora_target_modules"] = args.lora_targets.split(",")

    config = TrainingConfig(**config_kwargs)

    # Optional trainer knobs added to config via attribute assignment so
    # the dataclass does not need its schema extended just for HF Trainer
    # passthroughs. trainer.py reads via getattr-with-default.
    config.save_strategy = args.save_strategy
    config.save_steps = args.save_steps
    config.logging_steps = args.logging_steps

    if config.finetuning_method == "preference":
        from .preference import run_preference_training

        run_preference_training(config)
    elif config.finetuning_method == "rl":
        from .rl import run_grpo_training

        run_grpo_training(config)
    elif config.finetuning_method in {"rm", "prm"}:
        from .reward import train_reward_model

        train_reward_model(config)
    elif config.finetuning_method == "embedding":
        from .embedding import run_embedding_training

        run_embedding_training(config)
    else:
        run_training(config)

    # Optional post-training: heretic abliteration between training and GGUF.
    heretic_out_dir: str | None = None
    if getattr(args, "heretic", False):
        from .heretic_hook import (
            HereticConfig, HereticError, run_heretic, heretic_output_dir,
            is_heretic_available,
        )

        if not is_heretic_available():
            print("[mud-puppy] --heretic requested but heretic-llm is not "
                  "installed in mud-puppy's env. Run: pip install --no-deps "
                  "heretic-llm==1.2.0 ; pip install optuna questionary "
                  "pydantic-settings kernels psutil hf-transfer. Skipping.")
        else:
            # Determine the input dir for heretic. If we ran LoRA/QLoRA with
            # --merge-lora, the merged weights live at <output>/merged/ (see
            # merge_lora_weights). Otherwise the training output is the model.
            merged_dir = os.path.join(config.output_dir, "merged")
            if os.path.isdir(merged_dir) and os.path.isfile(
                os.path.join(merged_dir, "config.json")
            ):
                heretic_input = merged_dir
            elif os.path.isfile(os.path.join(config.output_dir, "config.json")):
                heretic_input = config.output_dir
            else:
                print("[mud-puppy] --heretic requested but cannot locate an "
                      "HF-loadable model dir. Looked at merged/ and output_dir. "
                      "Skipping. (LoRA runs need --merge-lora to produce "
                      "heretic-compatible weights.)")
                heretic_input = None

            if heretic_input is not None:
                heretic_out_dir = heretic_output_dir(config.output_dir)
                extra = (args.heretic_extra or "").split()
                hcfg = HereticConfig(
                    model_dir=heretic_input,
                    out_dir=heretic_out_dir,
                    n_trials=args.heretic_n_trials,
                    quantization=args.heretic_quantization,
                    good_prompts_dataset=args.heretic_good_prompts_dataset,
                    bad_prompts_dataset=args.heretic_bad_prompts_dataset,
                    system_prompt=args.heretic_system_prompt,
                    extra_args=extra,
                )
                try:
                    print(f"[mud-puppy] Running heretic: "
                          f"input={heretic_input} -> output={heretic_out_dir}")
                    heretic_out_dir = run_heretic(hcfg)
                    print(f"[mud-puppy] Heretic output: {heretic_out_dir}")
                except HereticError as exc:
                    print(f"[mud-puppy] Heretic failed: {exc}")
                    heretic_out_dir = None

    # Optional post-training: export to GGUF (+ kernel-anvil optimize).
    # Runs regardless of method once training succeeds.
    if getattr(args, "export_gguf", False):
        from .gguf_export import export_to_gguf, ExportConfig, GgufExportError

        # Prefer the heretic output if it landed; otherwise the training dir.
        source_dir = heretic_out_dir if heretic_out_dir else config.output_dir
        # For LoRA/QLoRA the output_dir contains adapter_config.json; the
        # export path detects and merges automatically. For full/other,
        # the same directory holds the merged model already.
        export_cfg = ExportConfig(
            source_dir=source_dir,
            out_path="model.gguf",
            quant=args.gguf_quant,
            optimize_with_kernel_anvil=args.optimize_with_kernel_anvil,
        )
        try:
            print(f"[mud-puppy] Exporting to GGUF ({args.gguf_quant or 'fp16'})...")
            result = export_to_gguf(export_cfg)
            for step in result.steps:
                print(f"  - {step}")
            print(f"[mud-puppy] GGUF ready: {result.gguf_path}")
            if result.serve_command:
                print("[mud-puppy] To serve with llama.cpp:")
                print(f"    {result.serve_command}")
        except GgufExportError as exc:
            print(f"[mud-puppy] GGUF export failed: {exc}")


if __name__ == "__main__":
    main()
