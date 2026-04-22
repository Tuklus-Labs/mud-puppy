"""Sequence packing for causal LM training.

Concatenates multiple short examples into a single row up to `max_seq_length`,
emitting a block-diagonal attention mask and per-segment position IDs so the
model attends only within its own segment.

Key properties:
- Cross-segment attention is zero (block-diagonal mask).
- Position IDs reset to [0..n-1] for each segment.
- Labels preserve -100 masking from the original examples.
- Padding positions in labels are also set to -100.
- Rows are padded to a multiple of `pad_to_multiple_of` (default 8).

Models that do not accept 2-D attention masks should fall back to the normal
padded `CausalLMCollator`. Use `PackedCollator` only after verifying the
target model supports block-diagonal masks (pass `attention_mask` of shape
`(batch, seq, seq)` to transformers that accept it, or convert to a 4-D
`(batch, 1, seq, seq)` float mask with -inf outside the block).
"""

from typing import Any, Dict, List

import torch


class PackedCollator:
    """Greedy bin-packing collator for causal language model training.

    Packs multiple short examples into rows of length `max_seq_length`,
    emitting:

    - ``input_ids``: shape (batch, seq)
    - ``attention_mask``: shape (batch, seq, seq) -- block-diagonal, long
    - ``labels``: shape (batch, seq) -- preserves -100 from source
    - ``position_ids``: shape (batch, seq) -- resets per segment

    Padding tokens use ``pad_token_id`` for ``input_ids`` and -100 for
    ``labels``.
    """

    def __init__(
        self,
        max_seq_length: int,
        pad_to_multiple_of: int = 8,
        pad_token_id: int = 0,
    ) -> None:
        self.max_seq_length = max_seq_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a list of examples into a packed batch."""
        # --- Fail loudly if upstream tokenization forgot to produce labels ---
        for i, ex in enumerate(examples):
            if "labels" not in ex:
                raise KeyError(
                    f"example {i} missing 'labels' key; upstream tokenizer "
                    "must populate labels before packing. "
                    "See mud_puppy.trainer.tokenize_chat, which clones "
                    "input_ids into labels."
                )
            if "input_ids" not in ex:
                raise KeyError(
                    f"example {i} missing 'input_ids' key"
                )

        # --- Greedy first-fit bin-packing ---
        rows: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = []
        cur_len = 0

        for ex in examples:
            n = len(ex["input_ids"])
            if n > self.max_seq_length:
                # Truncate oversized examples to fit in one row alone.
                ex = {
                    k: (v[: self.max_seq_length] if isinstance(v, list) else v)
                    for k, v in ex.items()
                }
                n = self.max_seq_length

            if cur_len + n > self.max_seq_length and cur:
                rows.append(cur)
                cur, cur_len = [], 0

            cur.append(ex)
            cur_len += n

        if cur:
            rows.append(cur)

        # --- Determine padded sequence length ---
        max_row_len = max(
            sum(len(e["input_ids"]) for e in row) for row in rows
        )
        seq = self._round_up(max_row_len, self.pad_to_multiple_of)

        batch_size = len(rows)

        input_ids = torch.full((batch_size, seq), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, seq), -100, dtype=torch.long)
        position_ids = torch.zeros((batch_size, seq), dtype=torch.long)
        # Block-diagonal attention mask: long, shape (batch, seq, seq).
        attn = torch.zeros((batch_size, seq, seq), dtype=torch.long)

        for b, row in enumerate(rows):
            offset = 0
            for ex in row:
                n = len(ex["input_ids"])
                input_ids[b, offset : offset + n] = torch.tensor(
                    ex["input_ids"], dtype=torch.long
                )
                labels[b, offset : offset + n] = torch.tensor(
                    ex["labels"], dtype=torch.long
                )
                position_ids[b, offset : offset + n] = torch.arange(
                    n, dtype=torch.long
                )
                # Block on the diagonal: all positions within this segment
                # attend to all other positions in this segment.
                attn[b, offset : offset + n, offset : offset + n] = 1
                offset += n

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "position_ids": position_ids,
        }

    @staticmethod
    def _round_up(n: int, multiple: int) -> int:
        """Round n up to the nearest multiple of `multiple`."""
        return ((n + multiple - 1) // multiple) * multiple
