"""Unit tests for mud_puppy.rl_verifier.

Run: pytest tests/test_rl_verifier.py
"""

import pytest

from mud_puppy.rl_verifier import (
    parse_verdict_json,
    normalize_verdict,
    reviewer_verdict_reward,
    reward_factory,
)


# ---------------------------------------------------------------------------
# parse_verdict_json
# ---------------------------------------------------------------------------

class TestParseVerdictJson:
    def test_pure_json(self):
        out = parse_verdict_json('{"verdict": "pass", "reason": "ok"}')
        assert out == {"verdict": "pass", "reason": "ok"}

    def test_wrapped_in_prose(self):
        txt = 'Looking at this diff, I think {"verdict": "fail", "reason": "off-by-one"} is correct.'
        out = parse_verdict_json(txt)
        assert out == {"verdict": "fail", "reason": "off-by-one"}

    def test_no_json(self):
        assert parse_verdict_json("This looks good to me.") is None

    def test_empty(self):
        assert parse_verdict_json("") is None

    def test_malformed_json(self):
        assert parse_verdict_json('{"verdict": pass}') is None

    def test_json_without_verdict(self):
        # JSON present but no verdict key: reject
        out = parse_verdict_json('{"other": "stuff"}')
        assert out is None


# ---------------------------------------------------------------------------
# normalize_verdict
# ---------------------------------------------------------------------------

class TestNormalizeVerdict:
    @pytest.mark.parametrize("raw,expected", [
        ("pass", "pass"),
        ("PASS", "pass"),
        ("  pass  ", "pass"),
        ("approved", "pass"),
        ("lgtm", "pass"),
        (True, "pass"),
        ("fail", "fail"),
        ("Rejected", "fail"),
        ("nack", "fail"),
        (False, "fail"),
        ("maybe", None),
        ("", None),
        (None, None),
        (42, None),
    ])
    def test_cases(self, raw, expected):
        assert normalize_verdict(raw) == expected


# ---------------------------------------------------------------------------
# reviewer_verdict_reward
# ---------------------------------------------------------------------------

class TestReviewerVerdictReward:
    def test_correct_pass(self):
        r = reviewer_verdict_reward(
            ['{"verdict": "pass", "reason": "looks fine"}'],
            expected_verdict=["pass"],
        )
        assert r == [1.0]

    def test_correct_fail(self):
        r = reviewer_verdict_reward(
            ['{"verdict": "fail", "reason": "memory leak"}'],
            expected_verdict=["fail"],
        )
        assert r == [1.0]

    def test_wrong_verdict(self):
        r = reviewer_verdict_reward(
            ['{"verdict": "pass", "reason": "ok"}'],
            expected_verdict=["fail"],
        )
        assert r == [-0.5]

    def test_unparseable(self):
        r = reviewer_verdict_reward(
            ["this is nonsense, no JSON here"],
            expected_verdict=["fail"],
        )
        assert r == [-1.0]

    def test_reason_bonus_hit(self):
        r = reviewer_verdict_reward(
            ['{"verdict": "fail", "reason": "null pointer dereference at line 42"}'],
            expected_verdict=["fail"],
            expected_reason_keywords=[["null pointer"]],
        )
        assert r == [pytest.approx(1.2)]

    def test_reason_bonus_miss(self):
        r = reviewer_verdict_reward(
            ['{"verdict": "fail", "reason": "style issue"}'],
            expected_verdict=["fail"],
            expected_reason_keywords=[["null pointer"]],
        )
        assert r == [1.0]

    def test_reason_bonus_does_not_apply_on_wrong_verdict(self):
        # Even if reason is on-topic, wrong verdict still gets -0.5.
        r = reviewer_verdict_reward(
            ['{"verdict": "pass", "reason": "null pointer dereference present"}'],
            expected_verdict=["fail"],
            expected_reason_keywords=[["null pointer"]],
        )
        assert r == [-0.5]

    def test_group_of_completions(self):
        # Typical GRPO group: 4 completions per prompt, label broadcast.
        completions = [
            '{"verdict": "fail", "reason": "off-by-one"}',          # +1
            '{"verdict": "pass", "reason": "ok"}',                  # -0.5
            'garbage output',                                        # -1
            '{"verdict": "fail", "reason": "memory unsafety"}',     # +1
        ]
        r = reviewer_verdict_reward(
            completions,
            expected_verdict=["fail"] * 4,
        )
        assert r == [1.0, -0.5, -1.0, 1.0]

    def test_no_label_degrades_gracefully(self):
        r = reviewer_verdict_reward(
            ['{"verdict": "pass", "reason": "ok"}'],
            expected_verdict=None,
        )
        # Format-only path: parse OK but no label, so score 0.3 (not full reward)
        assert r == [pytest.approx(0.3)]

    def test_bool_label(self):
        r = reviewer_verdict_reward(
            ['{"verdict": "pass"}'],
            expected_verdict=[True],
        )
        assert r == [1.0]

    def test_missing_verdict_key(self):
        r = reviewer_verdict_reward(
            ['{"reason": "no verdict field"}'],
            expected_verdict=["pass"],
        )
        # parse_verdict_json returns None because no 'verdict' key.
        assert r == [-1.0]


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

class TestRewardFactory:
    def test_reviewer(self):
        fn = reward_factory("reviewer_verdict")
        r = fn(['{"verdict": "pass"}'], expected_verdict=["pass"])
        assert r == [1.0]

    def test_aliases(self):
        assert reward_factory("reviewer") is reward_factory("reviewer_verdict")
        assert reward_factory("coder") is reward_factory("coder_compile_test")

    def test_unknown_kind(self):
        with pytest.raises(ValueError):
            reward_factory("nonsense")
