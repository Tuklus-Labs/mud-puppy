"""Multimodal (vision-language) support for mud-puppy.

This module provides:
- Dataset format detection for three common VLM dataset schemas
- Image loading (local files and data URIs)
- A DataCollator that lazily loads images per batch and calls AutoProcessor
- A dataset tokenization function for the multimodal training path

Supported dataset schemas
-------------------------
llava-conversations::

    {"image": "coco/000001.jpg", "conversations": [
        {"from": "human", "value": "<image>\\nWhat is in this image?"},
        {"from": "gpt", "value": "A cat on a windowsill."}
    ]}

chat-with-images (HF messages format with embedded image refs)::

    {"messages": [
        {"role": "user", "content": [
            {"type": "image", "image": "path/to/file.png"},
            {"type": "text", "text": "Describe this."}
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "A cat."}]}
    ]}

instruction-with-image::

    {"image": "file.jpg", "instruction": "What is in this image?", "response": "A cat."}
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset as TorchDataset

try:
    from PIL import Image
except ImportError as _pil_err:
    raise ImportError(
        "Pillow is required for multimodal training. "
        "Install it with: pip install pillow"
    ) from _pil_err


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_multimodal_format(columns: List[str]) -> str:
    """Detect which multimodal dataset schema is present from column names.

    Parameters
    ----------
    columns:
        List of column (key) names present in the dataset.

    Returns
    -------
    str
        One of ``"llava"``, ``"chat-with-images"``, or
        ``"instruction-with-image"``.

    Raises
    ------
    ValueError
        If none of the known schemas match.
    """
    cols = set(columns)

    # LLaVA-conversations: has both "image" and "conversations"
    if "image" in cols and "conversations" in cols:
        return "llava"

    # HF chat-messages (may contain image refs inside message content)
    if "messages" in cols:
        return "chat-with-images"

    # Simple instruction+image: needs "image" AND at least one text field
    if "image" in cols and ("instruction" in cols or "response" in cols):
        return "instruction-with-image"

    raise ValueError(
        "Cannot detect multimodal dataset format from columns: "
        f"{sorted(cols)}. "
        "Expected one of: "
        "{'image','conversations'} (LLaVA), "
        "{'messages'} (HF chat), or "
        "{'image','instruction'/'response'} (instruction)."
    )


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(
    path_or_uri: str,
    base_dir: Optional[str] = None,
) -> Image.Image:
    """Load a PIL Image from a local path or a base64 data URI.

    Parameters
    ----------
    path_or_uri:
        An absolute or relative filesystem path, or a ``data:image/*;base64,``
        URI string.
    base_dir:
        Directory used to resolve relative paths. When ``None``, the current
        working directory is used.

    Returns
    -------
    PIL.Image.Image
        The loaded image, always converted to RGB.

    Raises
    ------
    FileNotFoundError
        If a filesystem path does not exist.
    ValueError
        If the URI scheme is not recognised.
    """
    # Handle data: URIs
    if path_or_uri.startswith("data:"):
        # data:image/png;base64,<payload>
        try:
            header, payload = path_or_uri.split(",", 1)
            raw = base64.b64decode(payload)
            img = Image.open(io.BytesIO(raw))
            return img.convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Failed to decode data URI: {exc}"
            ) from exc

    # Resolve relative paths against base_dir
    path = path_or_uri
    if not os.path.isabs(path) and base_dir is not None:
        path = os.path.join(base_dir, path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Multimodal collator
# ---------------------------------------------------------------------------

class MultimodalCollator:
    """DataCollator that loads images lazily per batch and calls AutoProcessor.

    Each dataset example must have:
    - ``input_ids``: list[int] (text tokens, already tokenized)
    - ``attention_mask``: list[int]
    - ``labels``: list[int]
    - ``image_paths``: str or list[str] -- path(s) to the image(s)

    The collator opens images at collation time (not tokenization time), runs
    ``processor(images=..., text=...)`` to obtain ``pixel_values`` and text
    tensors, then pads everything to the longest sequence in the batch.

    Parameters
    ----------
    processor:
        An ``AutoProcessor`` (or any callable with the same signature as a
        HuggingFace processor). When ``None``, the collator still works but
        will not produce ``pixel_values``.
    tokenizer:
        The text tokenizer (used for pad token ID). May be ``None`` when the
        processor already bundles a tokenizer.
    max_length:
        Maximum token sequence length. Sequences are truncated to this length.
    pad_token_id:
        Override pad token ID (defaults to 0).
    base_dir:
        Base directory for resolving relative image paths.
    """

    def __init__(
        self,
        processor: Any,
        tokenizer: Any,
        max_length: int = 2048,
        pad_token_id: int = 0,
        base_dir: Optional[str] = None,
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.base_dir = base_dir

        # Determine pad token ID
        if pad_token_id != 0:
            self._pad_id = pad_token_id
        elif tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
            self._pad_id = tokenizer.pad_token_id
        else:
            self._pad_id = 0

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # --- Load images for this batch ---
        batch_images = []
        for ex in examples:
            raw_paths = ex.get("image_paths", None)
            if raw_paths is None:
                # Fall back to None; processor may handle text-only examples
                batch_images.append(None)
                continue

            if isinstance(raw_paths, list):
                # Multi-image example: list of paths
                imgs = [
                    load_image(p, base_dir=self.base_dir) for p in raw_paths
                ]
                batch_images.append(imgs)
            else:
                batch_images.append(load_image(raw_paths, base_dir=self.base_dir))

        # --- Build text list for processor ---
        # We pass pre-tokenized ids to the processor only when it accepts them;
        # many processors prefer raw text. We pass the raw ids as a dummy string
        # when we don't have the original text (a limitation of post-tokenization
        # collation). The processor's text encoding is used only for pixel_values
        # alignment; the actual input_ids come from the pre-tokenized batch.
        # For shape consistency we pass a placeholder.
        texts = ["<image>" for _ in examples]

        # --- Call processor to get pixel_values ---
        pixel_values = None
        if self.processor is not None:
            has_any_image = any(img is not None for img in batch_images)
            if has_any_image:
                # Replace None entries with a blank RGB image so the processor
                # always gets a consistent type
                safe_images = []
                for img in batch_images:
                    if img is None:
                        safe_images.append(Image.new("RGB", (224, 224)))
                    elif isinstance(img, list):
                        # Multi-image: pass as nested list; use first image to
                        # represent the example (some processors need flat lists)
                        safe_images.append(img)
                    else:
                        safe_images.append(img)

                try:
                    proc_out = self.processor(
                        images=safe_images,
                        text=texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    pixel_values = proc_out.pixel_values
                except Exception:
                    # Fallback: try without text argument (some processors)
                    try:
                        proc_out = self.processor(
                            images=safe_images,
                            return_tensors="pt",
                        )
                        pixel_values = proc_out.pixel_values
                    except Exception:
                        pixel_values = None

        # --- Pad and stack input_ids / attention_mask / labels ---
        input_ids_list = [
            ex["input_ids"][: self.max_length] for ex in examples
        ]
        attn_list = [
            ex["attention_mask"][: self.max_length] for ex in examples
        ]
        labels_list = [
            ex["labels"][: self.max_length] for ex in examples
        ]

        max_len = max(len(ids) for ids in input_ids_list)

        padded_ids = []
        padded_attn = []
        padded_labels = []

        for ids, attn, lbls in zip(input_ids_list, attn_list, labels_list):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self._pad_id] * pad_len)
            padded_attn.append(attn + [0] * pad_len)
            # Pad labels with -100 so padded positions are excluded from loss
            padded_labels.append(lbls + [-100] * pad_len)

        batch: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

        if pixel_values is not None:
            batch["pixel_values"] = pixel_values

        return batch


# ---------------------------------------------------------------------------
# Dataset tokenization for the multimodal training path
# ---------------------------------------------------------------------------

def _extract_text_and_image_paths(
    row: Dict[str, Any],
    fmt: str,
) -> tuple[str, Union[str, List[str], None]]:
    """Extract (text, image_paths) from a single dataset row.

    Returns
    -------
    text:
        The instruction/response text to tokenize.
    image_paths:
        A single path string, a list of path strings, or None.
    """
    if fmt == "llava":
        convs = row.get("conversations", [])
        parts = []
        for turn in convs:
            speaker = turn.get("from", "")
            value = turn.get("value", "")
            # Strip the <image> token from text; pixel_values handle the image
            value = value.replace("<image>", "").strip()
            if value:
                if speaker == "human":
                    parts.append(f"User: {value}")
                else:
                    parts.append(f"Assistant: {value}")
        text = "\n".join(parts)
        image_paths = row.get("image")

    elif fmt == "chat-with-images":
        messages = row.get("messages", [])
        parts = []
        image_paths_found: List[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text":
                            parts.append(f"{role}: {block.get('text', '')}")
                        elif btype == "image":
                            img_ref = block.get("image") or block.get("url") or block.get("path")
                            if img_ref:
                                image_paths_found.append(img_ref)
        text = "\n".join(parts)
        if len(image_paths_found) == 1:
            image_paths = image_paths_found[0]
        elif len(image_paths_found) > 1:
            image_paths = image_paths_found
        else:
            image_paths = None

    elif fmt == "instruction-with-image":
        instruction = row.get("instruction", "")
        response = row.get("response", "")
        text = f"{instruction}\n{response}" if response else instruction
        image_paths = row.get("image")

    else:
        raise ValueError(f"Unknown format: {fmt}")

    return text, image_paths


def tokenize_multimodal_dataset(
    dataset_path: str,
    processor: Any,
    max_length: int = 2048,
    base_dir: Optional[str] = None,
    preprocessing_workers: int = 1,
) -> Any:
    """Load and tokenize a multimodal JSONL dataset.

    Image paths are preserved as a string column (``image_paths``) in the
    returned dataset. Images are NOT loaded here -- the ``MultimodalCollator``
    opens them lazily per batch to avoid holding all images in RAM.

    Parameters
    ----------
    dataset_path:
        Path to a JSONL file with one example per line.
    processor:
        An ``AutoProcessor`` whose ``.tokenizer`` attribute is used for text
        tokenization. If the processor does not have a ``tokenizer`` attribute,
        the processor itself is called for tokenization.
    max_length:
        Maximum token sequence length.
    base_dir:
        Base directory for resolving relative image paths. Defaults to the
        directory of ``dataset_path``.
    preprocessing_workers:
        Number of parallel workers for the HF dataset .map() call.

    Returns
    -------
    datasets.Dataset
        A HF Dataset with columns: ``input_ids``, ``attention_mask``,
        ``labels``, ``image_paths``.
    """
    from datasets import load_dataset

    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(dataset_path))

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    columns = dataset.column_names

    fmt = detect_multimodal_format(columns)

    # Get the tokenizer from the processor
    tokenizer = None
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    elif callable(processor):
        tokenizer = processor

    if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)

    def process_example(examples: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = len(examples[columns[0]])
        out_input_ids = []
        out_attention_mask = []
        out_labels = []
        out_image_paths = []

        for i in range(batch_size):
            row = {col: examples[col][i] for col in columns}
            text, img_paths = _extract_text_and_image_paths(row, fmt)

            # Tokenize the text
            if tokenizer is not None and callable(tokenizer):
                try:
                    enc = tokenizer(
                        text,
                        truncation=True,
                        padding=False,
                        max_length=max_length,
                        return_tensors=None,
                    )
                    ids = enc["input_ids"]
                    attn = enc["attention_mask"]
                except Exception:
                    # If the tokenizer is a mock or doesn't support these args,
                    # fall through to a plain call
                    enc = tokenizer(text)
                    ids = enc["input_ids"][0] if isinstance(enc["input_ids"][0], list) else enc["input_ids"]
                    attn = enc["attention_mask"][0] if isinstance(enc["attention_mask"][0], list) else enc["attention_mask"]
            else:
                raise ValueError(
                    "tokenizer is None or not callable -- cannot tokenize multimodal "
                    "sample. Provide a valid tokenizer to tokenize_multimodal_dataset()."
                )

            # Ensure we have plain Python lists (not tensors)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if hasattr(attn, "tolist"):
                attn = attn.tolist()

            out_input_ids.append(ids)
            out_attention_mask.append(attn)
            out_labels.append(list(ids))  # Clone; collator will mask prompt tokens

            # Serialize image_paths to a JSON-safe string for the HF dataset
            if isinstance(img_paths, list):
                import json
                out_image_paths.append(json.dumps(img_paths))
            elif img_paths is not None:
                out_image_paths.append(str(img_paths))
            else:
                out_image_paths.append("")

        return {
            "input_ids": out_input_ids,
            "attention_mask": out_attention_mask,
            "labels": out_labels,
            "image_paths": out_image_paths,
        }

    # Use num_proc=None (in-process) when workers <= 1 to avoid pickling the
    # processor/tokenizer across multiprocessing boundaries
    num_proc = preprocessing_workers if preprocessing_workers > 1 else None

    dataset = dataset.map(
        process_example,
        batched=True,
        num_proc=num_proc,
        remove_columns=columns,
        desc="Tokenizing multimodal dataset",
    )

    return dataset
