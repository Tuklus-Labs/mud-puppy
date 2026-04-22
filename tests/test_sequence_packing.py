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


def test_packed_collator_truncates_single_example_exceeding_max_seq_length():
    """An example longer than max_seq_length must be truncated, not rejected.

    The PackedCollator docs say: "Truncate oversized examples to fit in one row
    alone." Verify that:
    1. The batch is produced without raising.
    2. The truncated example occupies exactly max_seq_length tokens.
    3. The input_ids are the FIRST max_seq_length tokens (head truncation).
    4. Labels and position_ids are also capped at max_seq_length.
    """
    max_seq = 8
    long_ids = list(range(1, 20))  # 19 tokens -- well beyond max_seq
    example = {
        "input_ids": long_ids,
        "attention_mask": [1] * len(long_ids),
        "labels": long_ids,
    }
    coll = PackedCollator(max_seq_length=max_seq, pad_to_multiple_of=1)
    batch = coll([example])

    # Exactly one row, sequence dimension == max_seq.
    assert batch["input_ids"].shape == (1, max_seq), (
        f"Expected (1, {max_seq}), got {batch['input_ids'].shape}"
    )
    # Head tokens preserved.
    assert batch["input_ids"][0].tolist() == long_ids[:max_seq], (
        "Truncation kept wrong tokens; expected first max_seq_length ids"
    )
    # Labels truncated correspondingly (no -100 padding for over-long example
    # that fills the whole row).
    assert batch["labels"][0].tolist() == long_ids[:max_seq], (
        "Labels not truncated to match truncated input_ids"
    )
    # position_ids must be [0, 1, ..., max_seq-1].
    assert batch["position_ids"][0].tolist() == list(range(max_seq)), (
        "position_ids incorrect for truncated example"
    )


def test_packed_collator_truncates_mixed_oversized_and_normal():
    """A batch with one oversized and one normal example must handle both correctly.

    The oversized example is placed alone in its row (truncated to max_seq_length).
    The normal example goes into a separate row.
    """
    max_seq = 6
    oversized = {
        "input_ids": list(range(100, 110)),  # 10 tokens
        "attention_mask": [1] * 10,
        "labels": list(range(100, 110)),
    }
    normal = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "labels": [1, 2, 3],
    }
    coll = PackedCollator(max_seq_length=max_seq, pad_to_multiple_of=1)
    batch = coll([oversized, normal])

    # Two rows: one for the truncated oversized example, one for the normal.
    assert batch["input_ids"].shape[0] == 2, (
        f"Expected 2 rows, got {batch['input_ids'].shape[0]}"
    )
    # Each row is exactly max_seq wide (normal example padded; oversized truncated).
    assert batch["input_ids"].shape[1] == max_seq, (
        f"Expected sequence length {max_seq}, got {batch['input_ids'].shape[1]}"
    )
    # First row: truncated oversized -- first max_seq tokens of range(100,110).
    assert batch["input_ids"][0, :max_seq].tolist() == list(range(100, 100 + max_seq)), (
        "Oversized example not truncated to first max_seq tokens"
    )


# ---------------------------------------------------------------------------
# E1: empty examples guard
# ---------------------------------------------------------------------------

def test_packed_collator_raises_on_empty_batch():
    """PackedCollator must raise ValueError (not crash with cryptic error) when
    given an empty list of examples."""
    coll = PackedCollator(max_seq_length=16, pad_to_multiple_of=8)
    with pytest.raises(ValueError, match="empty"):
        coll([])


# ---------------------------------------------------------------------------
# E2: attention mask dtype is int8 (memory-efficient)
# ---------------------------------------------------------------------------

def test_packed_collator_attention_mask_dtype_is_int8():
    """The block-diagonal attention mask must use int8 dtype, not int64.

    At large sequence lengths (e.g. seq=8192, B=32), int64 uses 16 GB while
    int8 uses 2 GB. The dtype change cuts mask memory by 8x.
    """
    examples = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5],   "attention_mask": [1, 1],   "labels": [4, 5]},
    ]
    coll = PackedCollator(max_seq_length=16, pad_to_multiple_of=8)
    batch = coll(examples)
    assert batch["attention_mask"].dtype == torch.int8, (
        f"Expected int8 attention_mask, got {batch['attention_mask'].dtype}"
    )


def test_packed_collator_all_oversized_mask_dtype_and_bounded():
    """All-oversized examples: each goes in its own row (truncated to max_seq).
    Verify dtype stays int8 and the batch doesn't balloon in size.
    """
    max_seq = 4
    batch_size = 8
    # 8 examples each of length 10 -- all exceed max_seq, each truncated to 4
    examples = [
        {
            "input_ids": list(range(i * 10, i * 10 + 10)),
            "attention_mask": [1] * 10,
            "labels": list(range(i * 10, i * 10 + 10)),
        }
        for i in range(batch_size)
    ]
    coll = PackedCollator(max_seq_length=max_seq, pad_to_multiple_of=1)
    batch = coll(examples)

    mask = batch["attention_mask"]
    assert mask.dtype == torch.int8, f"Expected int8, got {mask.dtype}"
    # Memory: 8 rows * 4 * 4 * 1 byte = 128 bytes; not the 4+ KB int64 would use
    assert mask.numel() == batch_size * max_seq * max_seq
    assert mask.element_size() == 1  # int8 is 1 byte
