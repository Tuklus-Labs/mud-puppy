"""Embedding fine-tuning via Multiple Negatives Ranking Loss (MNRL).

This module implements contrastive embedding training for sentence-transformer
models. It uses in-batch negatives: each (anchor, positive) pair treats all
other positives in the batch as negatives, so effective negative count scales
with batch size.

Datasets are expected to be JSONL files with two fields:

- ``anchor``: the query / input text
- ``positive``: the semantically matching passage / text

Output is saved in a sentence-transformers-compatible directory layout so
the trained model can be loaded directly with
``SentenceTransformer(output_dir)``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .config import TrainingConfig


# ---------------------------------------------------------------------------
# Core components
# ---------------------------------------------------------------------------


def mean_pooling(
    model_output: Any,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token embeddings, respecting the attention mask.

    Args:
        model_output: Output from a HuggingFace model (needs ``.last_hidden_state``).
        attention_mask: Attention mask from the tokenizer, shape ``(B, T)``.

    Returns:
        Pooled embeddings of shape ``(B, D)``.
    """
    token_embeddings = model_output.last_hidden_state  # (B, T, D)
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def mnrl_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    scale: float = 20.0,
) -> torch.Tensor:
    """Multiple Negatives Ranking Loss.

    Each anchor should match the positive at the same index.  All other
    positives in the batch serve as in-batch negatives.

    Args:
        anchors: Normalized anchor embeddings, shape ``(B, D)``.
        positives: Normalized positive embeddings, shape ``(B, D)``.
        scale: Temperature scaling factor (higher = sharper distribution).

    Returns:
        Scalar cross-entropy loss.
    """
    scores = torch.mm(anchors, positives.t()) * scale
    labels = torch.arange(scores.size(0), device=scores.device)
    return F.cross_entropy(scores, labels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MNRLDataset(Dataset):
    """Reads a JSONL file of ``{anchor, positive}`` pairs."""

    def __init__(self, path: str) -> None:
        self.examples: List[Dict[str, str]] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    self.examples.append(obj)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.examples[idx]


def collate_mnrl(
    batch: List[Dict[str, str]],
    tokenizer: Any,
    max_length: int = 128,
) -> Dict[str, Any]:
    """Tokenize anchors and positives separately.

    Returns a dict with ``anchor_*`` and ``positive_*`` tokenizer outputs.
    """
    anchors = [ex["anchor"] for ex in batch]
    positives = [ex["positive"] for ex in batch]

    anchor_enc = tokenizer(
        anchors,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    positive_enc = tokenizer(
        positives,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "anchor_input_ids": anchor_enc["input_ids"],
        "anchor_attention_mask": anchor_enc["attention_mask"],
        "positive_input_ids": positive_enc["input_ids"],
        "positive_attention_mask": positive_enc["attention_mask"],
    }


# ---------------------------------------------------------------------------
# Save in sentence-transformers format
# ---------------------------------------------------------------------------


def _save_sentence_transformer(
    model: torch.nn.Module,
    tokenizer: Any,
    config: TrainingConfig,
) -> None:
    """Save model + tokenizer in a sentence-transformers-compatible layout.

    Directory structure::

        output_dir/
            config.json          (model config)
            model.safetensors    (weights)
            tokenizer*           (tokenizer files)
            modules.json         (ST module list)
            sentence_bert_config.json
            1_Pooling/
                config.json      (pooling config)
    """
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # If PEFT model, merge and unload first
    try:
        from peft import PeftModel

        if isinstance(model, PeftModel):
            model = model.merge_and_unload()
    except ImportError:
        pass

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Detect hidden size from model config
    model_config = getattr(model, "config", None)
    hidden_size = 384  # default for MiniLM
    if model_config is not None:
        hidden_size = getattr(model_config, "hidden_size", hidden_size)

    # modules.json -- tells SentenceTransformer how to compose the pipeline
    modules = [
        {
            "idx": 0,
            "name": "0",
            "path": "",
            "type": "sentence_transformers.models.Transformer",
        },
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling",
        },
    ]
    with open(os.path.join(output_dir, "modules.json"), "w") as f:
        json.dump(modules, f, indent=2)

    # Pooling config
    pooling_dir = os.path.join(output_dir, "1_Pooling")
    os.makedirs(pooling_dir, exist_ok=True)
    pooling_config = {
        "word_embedding_dimension": hidden_size,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }
    with open(os.path.join(pooling_dir, "config.json"), "w") as f:
        json.dump(pooling_config, f, indent=2)

    # sentence_bert_config.json
    sb_config = {
        "max_seq_length": 128,
        "do_lower_case": False,
    }
    with open(os.path.join(output_dir, "sentence_bert_config.json"), "w") as f:
        json.dump(sb_config, f, indent=2)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def run_embedding_training(config: TrainingConfig) -> None:
    """Run contrastive embedding fine-tuning with MNRL loss.

    This is the main entry point, matching the pattern of
    ``run_preference_training`` and ``train_reward_model``.

    Args:
        config: A :class:`TrainingConfig` with ``finetuning_method="embedding"``.
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"[mud-puppy] Starting embedding training (MNRL)...")
    print(f"[mud-puppy]   model: {config.model_name_or_path}")
    print(f"[mud-puppy]   dataset: {config.dataset_path}")

    # ------------------------------------------------------------------
    # Model + tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = config.precision == "bf16" and device.type != "cpu"

    if use_bf16:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)

    print(f"[mud-puppy]   device: {device}, precision: {config.precision}")

    # ------------------------------------------------------------------
    # Optional LoRA
    # ------------------------------------------------------------------
    if config.lora_r > 0 and config.finetuning_method == "embedding":
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["query", "key", "value"],
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            model = get_peft_model(model, lora_config)
            print(f"[mud-puppy]   LoRA enabled: r={config.lora_r}, alpha={config.lora_alpha}")
            model.print_trainable_parameters()
        except ImportError:
            print("[mud-puppy]   WARNING: peft not installed, skipping LoRA")

    # ------------------------------------------------------------------
    # ROCm cache flush
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Dataset + split
    # ------------------------------------------------------------------
    full_dataset = MNRLDataset(config.dataset_path)
    total = len(full_dataset)
    eval_size = max(1, int(total * 0.1))
    train_size = total - eval_size

    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, eval_size]
    )

    def collate_fn(batch):
        return collate_mnrl(batch, tokenizer, max_length=128)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.dataloader_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.dataloader_workers,
    )

    print(f"[mud-puppy]   train: {train_size} examples, eval: {eval_size} examples")

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        # Linear decay
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    global_step = 0

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Move to device
            anchor_ids = batch["anchor_input_ids"].to(device)
            anchor_mask = batch["anchor_attention_mask"].to(device)
            positive_ids = batch["positive_input_ids"].to(device)
            positive_mask = batch["positive_attention_mask"].to(device)

            # Forward
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                anchor_out = model(input_ids=anchor_ids, attention_mask=anchor_mask)
                positive_out = model(input_ids=positive_ids, attention_mask=positive_mask)

                anchor_emb = mean_pooling(anchor_out, anchor_mask)
                positive_emb = mean_pooling(positive_out, positive_mask)

                # L2 normalize
                anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
                positive_emb = F.normalize(positive_emb, p=2, dim=1)

                loss = mnrl_loss(anchor_emb, positive_emb, scale=20.0)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[mud-puppy]   epoch {epoch + 1}/{config.num_epochs} - avg loss: {avg_loss:.4f}")

        # Eval
        if len(eval_loader) > 0:
            model.eval()
            eval_loss = 0.0
            eval_batches = 0
            with torch.no_grad():
                for batch in eval_loader:
                    anchor_ids = batch["anchor_input_ids"].to(device)
                    anchor_mask = batch["anchor_attention_mask"].to(device)
                    positive_ids = batch["positive_input_ids"].to(device)
                    positive_mask = batch["positive_attention_mask"].to(device)

                    anchor_out = model(input_ids=anchor_ids, attention_mask=anchor_mask)
                    positive_out = model(input_ids=positive_ids, attention_mask=positive_mask)

                    anchor_emb = F.normalize(mean_pooling(anchor_out, anchor_mask), p=2, dim=1)
                    positive_emb = F.normalize(mean_pooling(positive_out, positive_mask), p=2, dim=1)

                    eval_loss += mnrl_loss(anchor_emb, positive_emb, scale=20.0).item()
                    eval_batches += 1

            avg_eval = eval_loss / max(eval_batches, 1)
            print(f"[mud-puppy]   eval loss: {avg_eval:.4f}")
            model.train()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    _save_sentence_transformer(model, tokenizer, config)
    print(f"[mud-puppy] Embedding training complete! Model saved to {config.output_dir}")
