import torch
import pytest
from mud_puppy.packing import PackedCollator


def test_packed_collator_concatenates_short_sequences():
    examples = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5]},
        {"input_ids": [6, 7, 8, 9], "attention_mask": [1, 1, 1, 1], "labels": [6, 7, 8, 9]},
    ]
    coll = PackedCollator(max_seq_length=16, pad_to_multiple_of=8)
    batch = coll(examples)

    # All 3 examples should pack into one row (total tokens = 9, pad to 16)
    assert batch["input_ids"].shape == (1, 16)
    assert batch["input_ids"][0, :9].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_packed_collator_position_ids_reset_per_segment():
    examples = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5]},
    ]
    coll = PackedCollator(max_seq_length=8, pad_to_multiple_of=8)
    batch = coll(examples)
    # position_ids = [0,1,2, 0,1, pad,pad,pad]
    pos = batch["position_ids"][0]
    assert pos[:3].tolist() == [0, 1, 2]
    assert pos[3:5].tolist() == [0, 1]


def test_packed_collator_block_diagonal_mask_blocks_cross_segment_attention():
    examples = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5]},
    ]
    coll = PackedCollator(max_seq_length=8, pad_to_multiple_of=8)
    batch = coll(examples)
    mask = batch["attention_mask"]
    # Shape (1, seq, seq) for 2D block-diagonal
    assert mask.shape == (1, 8, 8)
    # Tokens 0,1,2 are segment A, tokens 3,4 are segment B
    assert mask[0, 0, 2].item() == 1  # same segment
    assert mask[0, 2, 0].item() == 1  # same segment
    assert mask[0, 0, 3].item() == 0  # cross-segment -> masked
    assert mask[0, 3, 2].item() == 0  # cross-segment -> masked
    assert mask[0, 3, 4].item() == 1  # segment B self


def test_packed_collator_labels_preserve_negative_hundred():
    examples = [
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        {"input_ids": [3, 4], "attention_mask": [1, 1], "labels": [3, 4]},
    ]
    coll = PackedCollator(max_seq_length=8, pad_to_multiple_of=8)
    batch = coll(examples)
    labels = batch["labels"][0].tolist()
    assert labels[0] == -100
    assert labels[1] == 2
    assert labels[2] == 3
    assert labels[3] == 4
    # Padding positions labelled -100
    assert labels[4:] == [-100, -100, -100, -100]
