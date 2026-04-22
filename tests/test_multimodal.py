"""Tests for mud_puppy.multimodal -- vision-language training support."""

import io
import os
import struct
import zlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers: generate a minimal valid PNG in memory so we never need disk images
# ---------------------------------------------------------------------------

def _make_png_bytes(width: int = 4, height: int = 4) -> bytes:
    """Return raw bytes of a tiny valid RGB PNG."""

    def chunk(name: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        return length + name + data + crc

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = chunk(b"IHDR", ihdr_data)

    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00" + bytes([128, 64, 32] * width)
    compressed = zlib.compress(raw_rows, 9)
    idat = chunk(b"IDAT", compressed)
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


# ---------------------------------------------------------------------------
# 1. detect_multimodal_format
# ---------------------------------------------------------------------------

class TestDetectMultimodalFormat:
    def test_detect_format_llava(self):
        from mud_puppy.multimodal import detect_multimodal_format

        fmt = detect_multimodal_format(["image", "conversations"])
        assert fmt == "llava"

    def test_detect_format_chat_messages(self):
        from mud_puppy.multimodal import detect_multimodal_format

        fmt = detect_multimodal_format(["messages"])
        assert fmt == "chat-with-images"

    def test_detect_format_instruction_with_image(self):
        from mud_puppy.multimodal import detect_multimodal_format

        fmt = detect_multimodal_format(["image", "instruction", "response"])
        assert fmt == "instruction-with-image"

    def test_detect_format_unknown_raises(self):
        from mud_puppy.multimodal import detect_multimodal_format

        with pytest.raises(ValueError, match="Cannot detect"):
            detect_multimodal_format(["text"])

    def test_detect_format_instruction_without_image_raises(self):
        from mud_puppy.multimodal import detect_multimodal_format

        with pytest.raises(ValueError, match="Cannot detect"):
            detect_multimodal_format(["instruction", "response"])


# ---------------------------------------------------------------------------
# 2. load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_load_image_local_file(self, tmp_path):
        from mud_puppy.multimodal import load_image

        png_path = tmp_path / "test.png"
        png_path.write_bytes(_make_png_bytes())

        img = load_image(str(png_path))
        assert img.mode == "RGB"
        assert img.size == (4, 4)

    def test_load_image_respects_base_dir(self, tmp_path):
        from mud_puppy.multimodal import load_image

        sub = tmp_path / "images"
        sub.mkdir()
        png_path = sub / "cat.png"
        png_path.write_bytes(_make_png_bytes(8, 8))

        # Relative path + base_dir should resolve to the full path
        img = load_image("images/cat.png", base_dir=str(tmp_path))
        assert img.mode == "RGB"
        assert img.size == (8, 8)

    def test_load_image_absolute_ignores_base_dir(self, tmp_path):
        from mud_puppy.multimodal import load_image

        png_path = tmp_path / "abs.png"
        png_path.write_bytes(_make_png_bytes(2, 2))

        # Absolute path should work even if base_dir is something else
        img = load_image(str(png_path), base_dir="/some/other/dir")
        assert img.size == (2, 2)

    def test_load_image_data_uri(self):
        """load_image should handle data: URIs (base64-encoded PNG)."""
        import base64
        from mud_puppy.multimodal import load_image

        png_bytes = _make_png_bytes(4, 4)
        b64 = base64.b64encode(png_bytes).decode()
        uri = f"data:image/png;base64,{b64}"

        img = load_image(uri)
        assert img.mode == "RGB"
        assert img.size == (4, 4)

    def test_load_image_missing_file_raises(self):
        from mud_puppy.multimodal import load_image

        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/to/image.png")


# ---------------------------------------------------------------------------
# 3. MultimodalCollator (mock processor)
# ---------------------------------------------------------------------------

class TestMultimodalCollator:
    """Test the collator using a lightweight mock processor.

    We never download a real VLM -- the mock mimics AutoProcessor's call
    signature and return shape so we can test collation logic in isolation.
    """

    def _make_mock_processor(self, seq_len: int = 16, num_patches: int = 196):
        """Build a mock that behaves like AutoProcessor for a 224x224 ViT."""
        processor = MagicMock()

        def fake_call(images=None, text=None, padding=None, truncation=None,
                      max_length=None, return_tensors=None, **kwargs):
            batch_size = len(images) if images is not None else 1
            out = MagicMock()
            out.input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
            out.attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
            # 3-channel 224x224 normalized pixel values
            out.pixel_values = torch.zeros(batch_size, 3, 224, 224)
            # items() / keys() used internally; make the mock dict-like too
            out.__iter__ = lambda s: iter(["input_ids", "attention_mask", "pixel_values"])
            out.keys = lambda: ["input_ids", "attention_mask", "pixel_values"]
            out.__getitem__ = lambda s, k: getattr(s, k)
            return out

        processor.side_effect = fake_call
        return processor

    def _make_examples(self, tmp_path, n: int = 2):
        """Make n minimal examples with real image files."""
        examples = []
        for i in range(n):
            png = tmp_path / f"img_{i}.png"
            png.write_bytes(_make_png_bytes(4, 4))
            examples.append({
                "input_ids": [1, 2, 3, i + 10],
                "attention_mask": [1, 1, 1, 1],
                "labels": [1, 2, 3, i + 10],
                "image_paths": str(png),
            })
        return examples

    def test_collator_produces_expected_keys(self, tmp_path):
        from mud_puppy.multimodal import MultimodalCollator

        processor = self._make_mock_processor()
        collator = MultimodalCollator(
            processor=processor,
            tokenizer=None,
            max_length=32,
        )
        examples = self._make_examples(tmp_path, n=2)
        batch = collator(examples)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "pixel_values" in batch
        assert "labels" in batch

    def test_collator_batch_size_matches_input(self, tmp_path):
        from mud_puppy.multimodal import MultimodalCollator

        processor = self._make_mock_processor(seq_len=8)
        collator = MultimodalCollator(processor=processor, tokenizer=None, max_length=32)
        examples = self._make_examples(tmp_path, n=3)
        batch = collator(examples)

        # pixel_values should have batch dimension matching number of examples
        assert batch["pixel_values"].shape[0] == 3

    def test_collator_labels_are_tensor(self, tmp_path):
        from mud_puppy.multimodal import MultimodalCollator

        processor = self._make_mock_processor()
        collator = MultimodalCollator(processor=processor, tokenizer=None, max_length=32)
        examples = self._make_examples(tmp_path, n=2)
        batch = collator(examples)

        assert isinstance(batch["labels"], torch.Tensor)

    def test_collator_handles_single_example(self, tmp_path):
        from mud_puppy.multimodal import MultimodalCollator

        processor = self._make_mock_processor(seq_len=4)
        collator = MultimodalCollator(processor=processor, tokenizer=None, max_length=32)
        examples = self._make_examples(tmp_path, n=1)
        batch = collator(examples)

        assert batch["pixel_values"].shape[0] == 1

    def test_collator_multi_image_per_example(self, tmp_path):
        """Examples with a list of image paths (multi-image rows)."""
        from mud_puppy.multimodal import MultimodalCollator

        # Processor that handles list-of-lists of images
        processor = MagicMock()

        def fake_call(images=None, text=None, **kwargs):
            # images is a list-of-lists here; flatten for batch dim
            batch_size = len(images)
            out = MagicMock()
            out.input_ids = torch.ones(batch_size, 8, dtype=torch.long)
            out.attention_mask = torch.ones(batch_size, 8, dtype=torch.long)
            out.pixel_values = torch.zeros(batch_size, 3, 224, 224)
            out.keys = lambda: ["input_ids", "attention_mask", "pixel_values"]
            out.__getitem__ = lambda s, k: getattr(s, k)
            return out

        processor.side_effect = fake_call

        # Create two images per example
        examples = []
        for i in range(2):
            paths = []
            for j in range(2):
                p = tmp_path / f"img_{i}_{j}.png"
                p.write_bytes(_make_png_bytes(4, 4))
                paths.append(str(p))
            examples.append({
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [1, 2, 3],
                "image_paths": paths,
            })

        collator = MultimodalCollator(processor=processor, tokenizer=None, max_length=32)
        batch = collator(examples)
        assert batch["pixel_values"].shape[0] == 2


# ---------------------------------------------------------------------------
# 4. tokenize_multimodal_dataset (integration smoke test)
# ---------------------------------------------------------------------------

class TestTokenizeMultimodalDataset:
    """Smoke-test the dataset tokenization function directly."""

    def _write_llava_jsonl(self, tmp_path: Path, image_path: str) -> Path:
        import json
        data = {"image": image_path, "conversations": [
            {"from": "human", "value": "<image>\nWhat is in this image?"},
            {"from": "gpt", "value": "A cat on a windowsill."},
        ]}
        p = tmp_path / "llava.jsonl"
        p.write_text(json.dumps(data) + "\n")
        return p

    def _write_instruction_jsonl(self, tmp_path: Path, image_path: str) -> Path:
        import json
        data = {"image": image_path, "instruction": "Describe.", "response": "A cat."}
        p = tmp_path / "inst.jsonl"
        p.write_text(json.dumps(data) + "\n")
        return p

    def _write_chat_jsonl(self, tmp_path: Path, image_path: str) -> Path:
        import json
        data = {"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Describe this."}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "A cat."}]},
        ]}
        p = tmp_path / "chat.jsonl"
        p.write_text(json.dumps(data) + "\n")
        return p

    def _make_mock_processor(self):
        proc = MagicMock()
        proc.tokenizer = MagicMock()
        proc.tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 4]],
            "attention_mask": [[1, 1, 1, 1]],
        }
        proc.tokenizer.pad_token = None
        proc.tokenizer.eos_token = "<eos>"
        proc.tokenizer.pad_token_id = 0
        return proc

    def test_tokenize_llava_format(self, tmp_path):
        from mud_puppy.multimodal import tokenize_multimodal_dataset

        png = tmp_path / "cat.png"
        png.write_bytes(_make_png_bytes())
        ds_path = self._write_llava_jsonl(tmp_path, str(png))

        proc = self._make_mock_processor()
        proc.tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 4]],
            "attention_mask": [[1, 1, 1, 1]],
        }

        dataset = tokenize_multimodal_dataset(str(ds_path), proc, max_length=64,
                                              base_dir=str(tmp_path))
        assert len(dataset) == 1
        row = dataset[0]
        # Must have image_paths preserved as a string column
        assert "image_paths" in row or "image_paths" in dataset.column_names

    def test_tokenize_instruction_format(self, tmp_path):
        from mud_puppy.multimodal import tokenize_multimodal_dataset

        png = tmp_path / "dog.png"
        png.write_bytes(_make_png_bytes())
        ds_path = self._write_instruction_jsonl(tmp_path, str(png))

        proc = self._make_mock_processor()
        dataset = tokenize_multimodal_dataset(str(ds_path), proc, max_length=64,
                                              base_dir=str(tmp_path))
        assert len(dataset) == 1

    def test_tokenize_chat_format(self, tmp_path):
        from mud_puppy.multimodal import tokenize_multimodal_dataset

        png = tmp_path / "bird.png"
        png.write_bytes(_make_png_bytes())
        ds_path = self._write_chat_jsonl(tmp_path, str(png))

        proc = self._make_mock_processor()
        dataset = tokenize_multimodal_dataset(str(ds_path), proc, max_length=64,
                                              base_dir=str(tmp_path))
        assert len(dataset) == 1
