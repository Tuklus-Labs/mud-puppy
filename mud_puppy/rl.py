"""Reinforcement learning utilities for ROCm-first training.

This module provides GRPO-style training using TRL's GRPOTrainer when available,
falling back to a PPOTrainer-based approach for older TRL versions.

Expected dataset format (JSON/JSONL):

- ``prompt``: the input text to condition on
- ``target`` (optional): a reference response; if absent, the model will
  generate responses freely and rewards must be computed from scratch
- ``reward`` (optional): a scalar reward; if not provided, rewards can be
  computed by an external reward model or heuristic and injected into the
  loop.

If only ``prompt`` is present, this script will generate responses and assign
pseudo-rewards based on a simple length/penalty heuristic. For real GRPO
setups, you should provide an explicit ``reward`` column or integrate a
reward model.
"""

from __future__ import annotations

from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from .config import TrainingConfig

# Try to import GRPOTrainer (modern TRL)
GRPOTrainer = None
try:
    from trl import GRPOTrainer
except ImportError:
    pass

# Fallback to PPO-based approach
PPOTrainer = None
PPOConfig = None
AutoModelForCausalLMWithValueHead = None
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
except ImportError:
    pass


def _compute_heuristic_reward(responses: List[str], **kwargs) -> List[float]:
    """A very simple stand-in reward function.

    Penalizes extremely short or extremely long responses and rewards
    moderate-length outputs. Real use cases should plug in a learned reward
    model or task-specific metric.
    """
    rewards: List[float] = []
    for text in responses:
        length = len(text.split())
        if length < 4:
            rewards.append(-1.0)
        elif length > 512:
            rewards.append(-1.0)
        else:
            rewards.append(1.0)
    return rewards


def _load_rl_dataset(path: str):
    """Load an RL dataset with prompt column."""
    dataset = load_dataset("json", data_files=path)["train"]

    if "prompt" not in dataset.column_names:
        raise ValueError("RL dataset must contain a 'prompt' column")

    return dataset


def run_grpo_training(config: TrainingConfig):
    """Run GRPO (Guided Reinforcement Policy Optimization) training.

    This function uses TRL's native GRPOTrainer when available, which implements
    proper group-based RL for language models. For older TRL versions, it falls
    back to a PPOTrainer-based approximation.
    """
    print(f"[mud-puppy] Starting GRPO training")
    print(f"[mud-puppy] Model: {config.model_name_or_path}")
    print(f"[mud-puppy] Dataset: {config.dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = _load_rl_dataset(config.dataset_path)

    # Use native GRPOTrainer if available
    if GRPOTrainer is not None:
        print("[mud-puppy] Using native GRPOTrainer")
        _run_native_grpo(config, tokenizer, dataset)
    elif PPOTrainer is not None:
        print("[mud-puppy] Using PPO-based GRPO approximation")
        _run_ppo_grpo(config, tokenizer, dataset)
    else:
        raise RuntimeError(
            "Neither GRPOTrainer nor PPOTrainer available. "
            "Please install trl with: pip install trl"
        )


def _run_native_grpo(config: TrainingConfig, tokenizer, dataset):
    """Run training using TRL's native GRPOTrainer."""
    from trl import GRPOConfig

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path, trust_remote_code=config.trust_remote_code
    )

    # Define reward function
    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        """Compute rewards for generated completions."""
        return _compute_heuristic_reward(completions)

    # Configure GRPO training
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        fp16=config.precision == "fp16",
        bf16=config.precision == "bf16",
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        max_completion_length=128,
        num_generations=4,  # Number of completions per prompt for group comparison
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("[mud-puppy] GRPO training complete!")


def _run_ppo_grpo(config: TrainingConfig, tokenizer, dataset):
    """Run GRPO-style training using PPOTrainer as a fallback."""
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name_or_path, trust_remote_code=config.trust_remote_code
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name_or_path, trust_remote_code=config.trust_remote_code
    )

    ppo_config = PPOConfig(
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation,
        mini_batch_size=config.batch_size,
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        fp16=config.precision == "fp16",
        bf16=config.precision == "bf16",
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    gen_kwargs = {
        "max_new_tokens": 128,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Create a simple dataloader
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        return {"prompt": [item["prompt"] for item in batch]}

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )

    for epoch in range(config.num_epochs):
        for batch in dataloader:
            prompts: List[str] = batch["prompt"]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(
                model.pretrained_model.device
            )

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Compute heuristic rewards
            rewards = _compute_heuristic_reward(responses)

            # Convert to tensors for PPO step
            query_tensors = [inputs["input_ids"][i] for i in range(len(prompts))]
            response_tensors = [outputs[i] for i in range(len(prompts))]
            reward_tensors = [
                torch.tensor(r, dtype=torch.float32) for r in rewards
            ]

            # Run PPO step
            stats = trainer.step(query_tensors, response_tensors, reward_tensors)

        print(f"[mud-puppy] Completed epoch {epoch + 1}/{config.num_epochs}")

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("[mud-puppy] PPO-based GRPO training complete!")
