"""Command line interface for the LoRA library.

Usage:
    python -m lora_library.cli train --config library.yaml
    python -m lora_library.cli train-one --adapter coding --dataset code.jsonl
    python -m lora_library.cli evaluate --adapter math
    python -m lora_library.cli merge --adapters coding,math --output hybrid
    python -m lora_library.cli list
    python -m lora_library.cli serve --port 8080
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def cmd_train(args: argparse.Namespace) -> int:
    """Train all adapters from a configuration file."""
    from .config import LoraLibraryConfig
    from .registry import AdapterRegistry
    from .batch_trainer import BatchTrainer

    print(f"[lora-library] Loading config from: {args.config}")
    config = LoraLibraryConfig.from_yaml(args.config)

    # Override base model if specified
    if args.model:
        config.base_model_path = args.model

    # Create registry
    registry = AdapterRegistry(config.output_dir)

    # Create trainer
    trainer = BatchTrainer(config, registry)

    # Filter adapters if specified
    adapters = None
    if args.adapters:
        adapters = args.adapters.split(",")

    # Train
    results = trainer.train_all(adapters=adapters, seed=args.seed)

    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    for name, result in results.items():
        status = result.status.value
        emoji = "[OK]" if status == "completed" else "[FAIL]"
        print(f"{emoji} {name}: {status} (loss: {result.final_loss:.4f})")

    return 0


def cmd_train_one(args: argparse.Namespace) -> int:
    """Train a single adapter."""
    from .config import (
        LoraLibraryConfig,
        AdapterConfig,
        DatasetConfig,
        TrainingHyperparameters,
    )
    from .registry import AdapterRegistry
    from .batch_trainer import BatchTrainer

    # Create minimal config
    config = LoraLibraryConfig(
        base_model_path=args.model,
        output_dir=args.output or "./lora_output",
    )

    # Create adapter config
    adapter = AdapterConfig(
        name=args.adapter,
        task=args.task or "general",
        r=args.rank,
        alpha=args.alpha,
        dataset=DatasetConfig(
            path=args.dataset,
            max_length=args.max_length,
        ),
        hyperparameters=TrainingHyperparameters(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
        ),
    )
    config.add_adapter(adapter)

    # Create registry and trainer
    registry = AdapterRegistry(config.output_dir)
    trainer = BatchTrainer(config, registry)

    # Train
    result = trainer.train_adapter(args.adapter, seed=args.seed)

    status = result.status.value
    print(f"\n[lora-library] Training {status}")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Steps: {result.training_steps}")
    print(f"  Time: {result.training_time_seconds:.1f}s")

    return 0 if status == "completed" else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate an adapter."""
    from .registry import AdapterRegistry
    from .evaluator import AdapterEvaluator
    from .datasets import TaskDatasetLoader

    print(f"[lora-library] Evaluating adapter: {args.adapter}")

    # Load registry
    registry = AdapterRegistry(args.registry)

    # Get adapter metadata
    meta = registry.get(args.adapter)
    if meta is None:
        print(f"Error: Adapter '{args.adapter}' not found")
        return 1

    # Load model (placeholder - would need actual implementation)
    print(f"[lora-library] Loading base model: {meta.base_model}")

    # Load evaluation dataset
    if args.eval_data:
        dataset = TaskDatasetLoader.load(meta.task, args.eval_data)
        examples = [{"text": ex.full_text} for ex in dataset.load()]
    else:
        print("Warning: No evaluation data specified, using dummy examples")
        examples = [{"text": "Test example"}]

    print(f"[lora-library] Evaluation dataset: {len(examples)} examples")

    # Note: Full evaluation requires model loading which isn't done here
    # This is a placeholder showing the CLI interface
    print(f"[lora-library] Task: {meta.task}")
    print(f"[lora-library] Best metrics: {meta.performance_metrics}")

    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Merge multiple adapters."""
    from .registry import AdapterRegistry
    from .merger import AdapterMerger, MergeStrategy

    print(f"[lora-library] Merging adapters: {args.adapters}")

    # Parse adapter names
    adapter_names = args.adapters.split(",")

    if len(adapter_names) < 2:
        print("Error: At least 2 adapters required for merging")
        return 1

    # Load registry
    registry = AdapterRegistry(args.registry)

    # Load adapter params
    adapters = []
    for name in adapter_names:
        try:
            params = registry.load_params(name)
            adapters.append(params)
            print(f"  Loaded: {name}")
        except Exception as e:
            print(f"  Error loading {name}: {e}")
            return 1

    # Parse weights if specified
    weights = None
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        if len(weights) != len(adapters):
            print("Error: Number of weights must match number of adapters")
            return 1

    # Get merge strategy
    strategy = MergeStrategy(args.strategy)

    # Merge
    merger = AdapterMerger()
    result = merger.merge(
        adapters=adapters,
        adapter_names=adapter_names,
        strategy=strategy,
        weights=weights,
    )

    # Save
    output_path = Path(args.output)
    result.save(output_path)

    print(f"\n[lora-library] Merged adapter saved to: {output_path}")
    print(f"  Strategy: {strategy.value}")
    if weights:
        print(f"  Weights: {weights}")

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List adapters in the registry."""
    from .registry import AdapterRegistry

    registry = AdapterRegistry(args.registry)

    adapters = list(registry)

    if not adapters:
        print("No adapters found in registry")
        return 0

    # Filter by task if specified
    if args.task:
        adapters = [a for a in adapters if a.task == args.task]

    if args.format == "json":
        data = [
            {
                "name": a.name,
                "task": a.task,
                "base_model": a.base_model,
                "versions": len(a.versions),
                "current_version": a.current_version,
                "metrics": a.performance_metrics,
            }
            for a in adapters
        ]
        print(json.dumps(data, indent=2))
    else:
        print(f"{'Name':<20} {'Task':<15} {'Versions':<10} {'Best Loss':<12}")
        print("-" * 60)

        for adapter in adapters:
            loss = adapter.performance_metrics.get("loss", "N/A")
            if isinstance(loss, float):
                loss = f"{loss:.4f}"
            print(
                f"{adapter.name:<20} {adapter.task:<15} "
                f"{len(adapter.versions):<10} {loss:<12}"
            )

    print(f"\nTotal: {len(adapters)} adapters")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the adapter server."""
    from .registry import AdapterRegistry

    print(f"[lora-library] Starting adapter server on port {args.port}")

    # Load registry
    registry = AdapterRegistry(args.registry)

    print(f"[lora-library] Registry: {args.registry}")
    print(f"[lora-library] Available adapters: {registry.list_names()}")

    # Note: Full server requires model loading
    # This is a placeholder showing the CLI interface

    if args.model:
        print(f"[lora-library] Loading base model: {args.model}")

        # Import server
        from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
        from .server import AdapterServer

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = FlaxAutoModelForCausalLM.from_pretrained(args.model)

        # Create server
        server = AdapterServer(
            model_fn=model.module.apply,
            base_params=model.params,
            tokenizer=tokenizer,
            adapters_dir=registry.adapters_dir,
            cache_size=args.cache_size,
        )

        # Load default adapter if specified
        if args.adapter:
            server.load_adapter(args.adapter)

        # Start REST API
        server.start_rest_api(host=args.host, port=args.port)
    else:
        print("Error: --model required for serve command")
        return 1

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export adapter catalog."""
    from .registry import AdapterRegistry

    registry = AdapterRegistry(args.registry)
    registry.export_catalog(args.output)

    print(f"[lora-library] Catalog exported to: {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="lora_library",
        description="LoRA Library - Train, manage, and serve specialized adapters",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train command
    train_parser = subparsers.add_parser("train", help="Train adapters from config")
    train_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to library configuration YAML",
    )
    train_parser.add_argument(
        "--model", "-m",
        help="Override base model path",
    )
    train_parser.add_argument(
        "--adapters", "-a",
        help="Comma-separated list of adapters to train (trains all if not specified)",
    )
    train_parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed",
    )

    # train-one command
    train_one_parser = subparsers.add_parser("train-one", help="Train a single adapter")
    train_one_parser.add_argument(
        "--adapter", "-a",
        required=True,
        help="Adapter name",
    )
    train_one_parser.add_argument(
        "--model", "-m",
        required=True,
        help="Base model path",
    )
    train_one_parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Training dataset path",
    )
    train_one_parser.add_argument(
        "--output", "-o",
        help="Output directory",
    )
    train_one_parser.add_argument(
        "--task", "-t",
        default="general",
        help="Task type (code, math, instruction, etc.)",
    )
    train_one_parser.add_argument(
        "--rank", "-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    train_one_parser.add_argument(
        "--alpha",
        type=float,
        default=32.0,
        help="LoRA alpha",
    )
    train_one_parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    train_one_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size",
    )
    train_one_parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="Number of epochs",
    )
    train_one_parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    train_one_parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed",
    )

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate an adapter")
    eval_parser.add_argument(
        "--adapter", "-a",
        required=True,
        help="Adapter name",
    )
    eval_parser.add_argument(
        "--registry", "-r",
        default="./lora_output",
        help="Registry path",
    )
    eval_parser.add_argument(
        "--eval-data", "-d",
        help="Evaluation dataset path",
    )
    eval_parser.add_argument(
        "--model", "-m",
        help="Base model path (overrides registry)",
    )

    # merge command
    merge_parser = subparsers.add_parser("merge", help="Merge multiple adapters")
    merge_parser.add_argument(
        "--adapters", "-a",
        required=True,
        help="Comma-separated adapter names",
    )
    merge_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for merged adapter",
    )
    merge_parser.add_argument(
        "--registry", "-r",
        default="./lora_output",
        help="Registry path",
    )
    merge_parser.add_argument(
        "--strategy", "-s",
        default="average",
        choices=["average", "weighted", "ties", "dare", "task_arithmetic"],
        help="Merge strategy",
    )
    merge_parser.add_argument(
        "--weights", "-w",
        help="Comma-separated weights for weighted merge",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List adapters")
    list_parser.add_argument(
        "--registry", "-r",
        default="./lora_output",
        help="Registry path",
    )
    list_parser.add_argument(
        "--task", "-t",
        help="Filter by task type",
    )
    list_parser.add_argument(
        "--format", "-f",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start adapter server")
    serve_parser.add_argument(
        "--registry", "-r",
        default="./lora_output",
        help="Registry path",
    )
    serve_parser.add_argument(
        "--model", "-m",
        help="Base model path",
    )
    serve_parser.add_argument(
        "--adapter", "-a",
        help="Default adapter to load",
    )
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to",
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port to listen on",
    )
    serve_parser.add_argument(
        "--cache-size",
        type=int,
        default=10,
        help="Maximum adapters to cache",
    )

    # export command
    export_parser = subparsers.add_parser("export", help="Export adapter catalog")
    export_parser.add_argument(
        "--registry", "-r",
        default="./lora_output",
        help="Registry path",
    )
    export_parser.add_argument(
        "--output", "-o",
        default="catalog.json",
        help="Output file path",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Dispatch to command handler
    commands = {
        "train": cmd_train,
        "train-one": cmd_train_one,
        "evaluate": cmd_evaluate,
        "merge": cmd_merge,
        "list": cmd_list,
        "serve": cmd_serve,
        "export": cmd_export,
    }

    handler = commands.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")
        return 1

    try:
        return handler(args)
    except Exception as e:
        print(f"Error: {e}")
        if os.environ.get("DEBUG"):
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
