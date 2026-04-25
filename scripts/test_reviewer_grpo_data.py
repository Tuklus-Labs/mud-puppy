#!/usr/bin/env python3
"""Sanity-check the reviewer GRPO dataset end-to-end.

Loads the first 10 rows of reviewer-grpo.jsonl, asserts schema, and
passes synthetic completions through reviewer_verdict_reward to confirm
the reward function and dataset columns speak the same language.

Prints a one-table summary. CPU-only, no GPU needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    REPO_ROOT / "training_data_sets" / "reviewer" / "reviewer-grpo.jsonl"
)

# Pull mud-puppy from the repo root, not site-packages (works with
# `pip install -e .` too).
sys.path.insert(0, str(REPO_ROOT))


def _fmt_kws(kws):
    if not kws:
        return "<none>"
    out = ", ".join(k[:20] for k in kws[:3])
    if len(kws) > 3:
        out += f", +{len(kws)-3}"
    return out


def main() -> int:
    from mud_puppy.rl_verifier import reviewer_verdict_reward

    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    if not data_path.exists():
        print(f"ERROR: data file missing: {data_path}", file=sys.stderr)
        return 2

    rows = []
    with data_path.open() as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            rows.append(json.loads(line))

    print(f"loaded {len(rows)} rows from {data_path}")
    assert len(rows) == 10, "need 10 rows for test"

    # --- Schema -----------------------------------------------------
    required = {"prompt", "expected_verdict", "expected_reason_keywords"}
    for i, r in enumerate(rows):
        missing = required - set(r.keys())
        assert not missing, f"row {i} missing columns: {missing}"
        assert r["expected_verdict"] in ("pass", "fail"), (
            f"row {i} has bad verdict: {r['expected_verdict']!r}"
        )
        assert isinstance(r["expected_reason_keywords"], list), (
            f"row {i} keywords not list"
        )
        assert len(r["expected_reason_keywords"]) >= 1, (
            f"row {i} has zero keywords"
        )
        assert len(r["expected_reason_keywords"]) <= 5, (
            f"row {i} has > 5 keywords"
        )
    print("schema check passed")

    # --- Synthesize completions to exercise every reward branch -----
    # Branches covered:
    #   A. correct verdict + keyword match        -> +1.2
    #   B. correct verdict + no keyword match     -> +1.0
    #   C. wrong verdict                          -> -0.5
    #   D. unparseable                            -> -1.0
    #
    # Pattern: for each row, construct 4 synthetic completions that cover
    # the cases. We then call reward on the flattened batch, so the
    # index-alignment with the expected_* columns mirrors GRPO's internal
    # broadcast.

    completions = []
    expected_verdict = []
    expected_reason_keywords = []
    labels = []

    for r in rows:
        gold = r["expected_verdict"]
        opp = "fail" if gold == "pass" else "pass"
        kw0 = r["expected_reason_keywords"][0]

        # A. correct + keyword
        completions.append(
            json.dumps({"verdict": gold, "reason": f"contains {kw0}"})
        )
        labels.append("A:correct+kw")

        # B. correct + no keyword
        completions.append(
            json.dumps({
                "verdict": gold,
                "reason": "generic filler with nothing specific here",
            })
        )
        labels.append("B:correct+nokw")

        # C. wrong verdict
        completions.append(
            json.dumps({"verdict": opp, "reason": "wrong call"})
        )
        labels.append("C:wrong")

        # D. unparseable
        completions.append("I think the code is fine but no json here")
        labels.append("D:unparseable")

        # Each row's labels get replicated 4x so alignment with completions
        # matches what GRPO does on a 4-generation group.
        for _ in range(4):
            expected_verdict.append(gold)
            expected_reason_keywords.append(r["expected_reason_keywords"])

    rewards = reviewer_verdict_reward(
        completions=completions,
        expected_verdict=expected_verdict,
        expected_reason_keywords=expected_reason_keywords,
    )
    assert len(rewards) == len(completions)

    # --- Table ------------------------------------------------------
    print()
    print(f"{'row':>4} {'case':<14} {'verdict':<7} {'kws':<28} {'reward':>7}")
    print("-" * 64)
    expected_map = {
        "A:correct+kw":   1.2,
        "B:correct+nokw": 1.0,
        "C:wrong":       -0.5,
        "D:unparseable": -1.0,
    }
    mismatches = 0
    for idx, (c, label, r) in enumerate(
        zip(completions, labels, rewards)
    ):
        row_idx = idx // 4
        kws = expected_reason_keywords[idx]
        verdict = expected_verdict[idx]
        exp = expected_map[label]
        flag = " " if abs(r - exp) < 0.01 else " <-- mismatch"
        if flag.strip():
            mismatches += 1
        print(
            f"{row_idx:>4} {label:<14} {verdict:<7} "
            f"{_fmt_kws(kws):<28} {r:>+7.2f}{flag}"
        )

    print()
    print(f"total rewards: {len(rewards)}")
    print(f"expected branch hits (40 total): {40 - mismatches}/40 correct")
    if mismatches:
        print("FAIL: some rewards did not match expected branch values",
              file=sys.stderr)
        return 1
    print("OK: reward function produced all expected branch values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
