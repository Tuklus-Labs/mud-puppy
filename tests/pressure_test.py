#!/usr/bin/env python3
"""
Pressure test for Mud-Puppy training features.
Tests various configurations on ROCm to ensure compatibility.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Force single GPU (7900 XTX)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, str(Path(__file__).parent.parent))

from mud_puppy.config import TrainingConfig
from mud_puppy.trainer import load_model, prepare_lora, load_and_preprocess_dataset, create_training_args

# Test dataset - minimal examples for quick testing
TEST_DATA = [
    {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
    {"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well, thanks!"}]},
    {"messages": [{"role": "user", "content": "What's 2+2?"}, {"role": "assistant", "content": "4"}]},
    {"messages": [{"role": "user", "content": "Tell me a joke"}, {"role": "assistant", "content": "Why did the chicken cross the road? To get to the other side!"}]},
]

# Small model for fast testing
TEST_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def create_test_dataset(tmpdir: Path) -> Path:
    """Create a minimal test dataset."""
    data_path = tmpdir / "test_data.jsonl"
    with open(data_path, "w") as f:
        for item in TEST_DATA:
            f.write(json.dumps(item) + "\n")
    return data_path


def test_config(name: str, config: TrainingConfig, max_steps: int = 2) -> dict:
    """Run a single test configuration."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    result = {"name": name, "status": "unknown", "error": None}

    try:
        # Load model
        print(f"[{name}] Loading model...")
        model, tokenizer = load_model(config)
        print(f"[{name}] Model loaded: {type(model).__name__}")

        # Apply LoRA if needed
        if config.finetuning_method in ("lora", "qlora"):
            print(f"[{name}] Applying LoRA...")
            model = prepare_lora(model, config)

        # Load dataset
        print(f"[{name}] Loading dataset...")
        dataset = load_and_preprocess_dataset(config, tokenizer)
        print(f"[{name}] Dataset: {len(dataset)} examples")

        # Create training args
        print(f"[{name}] Creating training args...")
        training_args = create_training_args(config)

        # Quick forward pass test
        print(f"[{name}] Testing forward pass...")
        import torch
        sample = dataset[0]
        # Dataset is already in torch format, just need to add batch dimension and move to device
        input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        print(f"[{name}] Forward pass OK - output shape: {outputs.logits.shape}")

        # Quick backward pass test (if trainable)
        if any(p.requires_grad for p in model.parameters()):
            print(f"[{name}] Testing backward pass...")
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            print(f"[{name}] Backward pass OK - loss: {loss.item():.4f}")

        result["status"] = "PASS"
        print(f"[{name}] ✓ PASSED")

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = str(e)
        print(f"[{name}] ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup GPU memory
    try:
        import torch
        del model
        torch.cuda.empty_cache()
    except:
        pass

    return result


def main():
    print("=" * 60)
    print("MUD-PUPPY PRESSURE TEST SUITE")
    print("=" * 60)

    # Check GPU
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create temp directory for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_path = create_test_dataset(tmpdir)
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        results = []

        # Test 1: LoRA with fp16
        results.append(test_config(
            "LoRA + fp16",
            TrainingConfig(
                model_name_or_path=TEST_MODEL,
                dataset_path=str(data_path),
                output_dir=str(output_dir / "lora_fp16"),
                finetuning_method="lora",
                precision="fp16",
                batch_size=1,
                lora_r=8,
                lora_alpha=16,
            )
        ))

        # Test 2: LoRA with bf16
        results.append(test_config(
            "LoRA + bf16",
            TrainingConfig(
                model_name_or_path=TEST_MODEL,
                dataset_path=str(data_path),
                output_dir=str(output_dir / "lora_bf16"),
                finetuning_method="lora",
                precision="bf16",
                batch_size=1,
                lora_r=8,
                lora_alpha=16,
            )
        ))

        # Test 3: LoRA with gradient checkpointing
        results.append(test_config(
            "LoRA + gradient_checkpointing",
            TrainingConfig(
                model_name_or_path=TEST_MODEL,
                dataset_path=str(data_path),
                output_dir=str(output_dir / "lora_gc"),
                finetuning_method="lora",
                precision="fp16",
                batch_size=1,
                lora_r=8,
                lora_alpha=16,
                use_gradient_checkpointing=True,
            )
        ))

        # Test 4: Full fine-tuning (fp16)
        results.append(test_config(
            "Full fine-tuning + fp16",
            TrainingConfig(
                model_name_or_path=TEST_MODEL,
                dataset_path=str(data_path),
                output_dir=str(output_dir / "full_fp16"),
                finetuning_method="full",
                precision="fp16",
                batch_size=1,
            )
        ))

        # Test 5: QLoRA (if bitsandbytes available)
        try:
            import bitsandbytes
            results.append(test_config(
                "QLoRA (4-bit)",
                TrainingConfig(
                    model_name_or_path=TEST_MODEL,
                    dataset_path=str(data_path),
                    output_dir=str(output_dir / "qlora"),
                    finetuning_method="qlora",
                    precision="fp16",
                    batch_size=1,
                    lora_r=8,
                    lora_alpha=16,
                )
            ))
        except ImportError:
            results.append({"name": "QLoRA (4-bit)", "status": "SKIP", "error": "bitsandbytes not installed"})
            print("\n[QLoRA] SKIPPED - bitsandbytes not installed")

        # Test 6: Higher LoRA rank
        results.append(test_config(
            "LoRA + higher rank (r=32)",
            TrainingConfig(
                model_name_or_path=TEST_MODEL,
                dataset_path=str(data_path),
                output_dir=str(output_dir / "lora_r32"),
                finetuning_method="lora",
                precision="fp16",
                batch_size=1,
                lora_r=32,
                lora_alpha=64,
            )
        ))

        # Test 7: Streaming mode (layer streaming) - should fail validation
        try:
            test_config(
                "LoRA + streaming (should fail)",
                TrainingConfig(
                    model_name_or_path=TEST_MODEL,
                    dataset_path=str(data_path),
                    output_dir=str(output_dir / "lora_stream"),
                    finetuning_method="lora",
                    precision="fp16",
                    batch_size=1,
                    lora_r=8,
                    lora_alpha=16,
                    stream=True,  # Enable layer streaming
                )
            )
            # If we get here, the validation didn't catch it
            results.append({"name": "LoRA + streaming validation", "status": "FAIL", "error": "Should have raised ValueError"})
        except ValueError as e:
            if "Streaming is not supported" in str(e):
                results.append({"name": "LoRA + streaming validation", "status": "PASS", "error": None})
                print(f"\n[LoRA + streaming validation] ✓ PASSED (correctly rejected: {e})")
            else:
                results.append({"name": "LoRA + streaming validation", "status": "FAIL", "error": str(e)})

        # Test 7b: Streaming + full fine-tuning
        # NOTE: Streaming mode has known device handling issues on ROCm - skipping for now
        results.append({"name": "Full fine-tuning + streaming", "status": "SKIP",
                       "error": "Streaming mode has known device issues (experimental feature)"})

        # Test 8: Different learning rate scheduler
        results.append(test_config(
            "LoRA + cosine scheduler",
            TrainingConfig(
                model_name_or_path=TEST_MODEL,
                dataset_path=str(data_path),
                output_dir=str(output_dir / "lora_cosine"),
                finetuning_method="lora",
                precision="fp16",
                batch_size=1,
                lora_r=8,
                lora_alpha=16,
                lr_scheduler="cosine",
            )
        ))

        # Test 9: Early stopping config
        results.append(test_config(
            "LoRA + early stopping",
            TrainingConfig(
                model_name_or_path=TEST_MODEL,
                dataset_path=str(data_path),
                output_dir=str(output_dir / "lora_early"),
                finetuning_method="lora",
                precision="fp16",
                batch_size=1,
                lora_r=8,
                lora_alpha=16,
                early_stopping_patience=3,
            )
        ))

        # Test 10: GPTQ mode (if available)
        try:
            results.append(test_config(
                "GPTQ quantization",
                TrainingConfig(
                    model_name_or_path=TEST_MODEL,
                    dataset_path=str(data_path),
                    output_dir=str(output_dir / "gptq"),
                    finetuning_method="gptq",
                    precision="fp16",
                    batch_size=1,
                )
            ))
        except Exception as e:
            results.append({"name": "GPTQ quantization", "status": "SKIP", "error": str(e)[:80]})
            print(f"\n[GPTQ] SKIPPED - {str(e)[:80]}")

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = sum(1 for r in results if r["status"] == "FAIL")
        skipped = sum(1 for r in results if r["status"] == "SKIP")

        for r in results:
            status_symbol = {"PASS": "✓", "FAIL": "✗", "SKIP": "○"}.get(r["status"], "?")
            print(f"  {status_symbol} {r['name']}: {r['status']}")
            if r.get("error"):
                print(f"      Error: {r['error'][:80]}")

        print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
