"""Verifier-grounded reward functions for GRPO training.

Replaces the length-based heuristic in rl.py with real task-grounded signals:

- ``reviewer_verdict_reward``: reward a reviewer for producing a valid JSON
  verdict that matches a precomputed ground-truth label. Used for the
  Phase 5 reviewer training on agent-router.

- ``reviewer_reason_bonus``: optional additional signal that rewards
  reviews whose 'reason' field includes any of the ground-truth keywords.

- ``coder_compile_test_reward``: reward a coder for producing a patch
  that applies cleanly, compiles, and passes the supplied tests. Used
  for the Phase 6 coder training. Requires a working sandbox.

Dataset contract (reviewer path):
    Each row must include an ``expected_verdict`` string field, either
    "pass" or "fail". Optionally ``expected_reason_keywords`` (a list of
    strings) to drive the reason bonus.

Dataset contract (coder path):
    Each row must include ``repo_path``, ``base_commit``, and
    ``test_command``. Reward function spawns the sandbox, applies the
    completion as a patch, runs the tests, returns 1.0 / 0.0 / -0.5.
    This path is a stub in Phase 5 and fleshed out in Phase 6.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional, Sequence

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_verdict_json(completion: str) -> Optional[dict]:
    """Best-effort extraction of the verdict JSON from a model completion.

    Models sometimes wrap JSON in markdown fences, prefix it with
    'assistant:', or emit a whole think-then-answer block. We scan for the
    first balanced JSON object and try to parse it.

    Returns the parsed dict, or None if no valid JSON was found.
    """
    if not completion:
        return None
    # Fast path: the entire completion is JSON.
    stripped = completion.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "verdict" in obj:
            return obj
    # Slow path: find the first plausible JSON object in the string.
    for match in _JSON_OBJECT_RE.finditer(completion):
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            return obj
    return None


def normalize_verdict(raw: object) -> Optional[str]:
    """Collapse verdict spellings into the two canonical labels.

    Accepts things like True / False (bool), 'PASS', 'ok', 'approve', etc.
    Returns 'pass', 'fail', or None if unintelligible.
    """
    if raw is True:
        return "pass"
    if raw is False:
        return "fail"
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token in {"pass", "passed", "ok", "approve", "approved", "accept", "accepted", "lgtm"}:
        return "pass"
    if token in {"fail", "failed", "reject", "rejected", "nack", "deny", "denied", "bad"}:
        return "fail"
    return None


# ---------------------------------------------------------------------------
# Reviewer reward
# ---------------------------------------------------------------------------

def reviewer_verdict_reward(
    completions: Sequence[str],
    expected_verdict: Optional[Sequence[object]] = None,
    expected_reason_keywords: Optional[Sequence[Sequence[str]]] = None,
    **kwargs,
) -> List[float]:
    """Reward a reviewer completion against a known-correct verdict.

    Scoring (per completion):

    +1.0   parse OK AND verdict matches ground truth
    +0.2   additional, if expected_reason_keywords given AND completion's
           'reason' field mentions any of them (case-insensitive substring)
    -0.5   parse OK AND verdict is the opposite of ground truth
    -1.0   parse failed (no valid JSON with a 'verdict' field)

    The small penalty for a confident wrong answer is deliberately less
    severe than the parse failure penalty. We want the model to learn the
    output format first, then learn the discrimination.

    ``expected_verdict`` is passed through by TRL's GRPOTrainer when the
    dataset has that column and ``remove_unused_columns=False``. Each
    element aligns with the completion at the same index.
    """
    n = len(completions)
    rewards = [0.0] * n

    if expected_verdict is None:
        # No labels: degrade to format-only reward (0 for parse-pass,
        # -1 for parse-fail). Caller probably misconfigured the dataset.
        log.warning("reviewer_verdict_reward called without expected_verdict; falling back to format-only")
        expected_verdict = [None] * n

    # Alignment sanity
    if len(expected_verdict) != n:
        # GRPO generates multiple completions per prompt; each prompt's
        # label is repeated across the group. TRL handles this internally
        # by broadcasting, but just in case:
        log.warning(
            "expected_verdict length %d does not match completions %d; broadcasting",
            len(expected_verdict), n,
        )
        expected_verdict = list(expected_verdict) * (n // max(1, len(expected_verdict)))
        expected_verdict = expected_verdict[:n]

    keywords_per_row = expected_reason_keywords if expected_reason_keywords else [None] * n
    if len(keywords_per_row) != n:
        keywords_per_row = list(keywords_per_row) * (n // max(1, len(keywords_per_row)))
        keywords_per_row = keywords_per_row[:n]

    for i, completion in enumerate(completions):
        parsed = parse_verdict_json(completion)
        if parsed is None:
            rewards[i] = -1.0
            continue

        predicted = normalize_verdict(parsed.get("verdict"))
        if predicted is None:
            rewards[i] = -1.0
            continue

        gold = normalize_verdict(expected_verdict[i])
        if gold is None:
            # No ground-truth label on this row. Just reward parse success.
            rewards[i] = 0.3
            continue

        if predicted == gold:
            rewards[i] = 1.0
            # Optional reason bonus: does the completion's 'reason' touch on
            # any expected keyword? Only applies on correct verdicts.
            kws = keywords_per_row[i] or []
            if kws:
                reason = str(parsed.get("reason", "")).lower()
                if any(kw.lower() in reason for kw in kws if kw):
                    rewards[i] += 0.2
        else:
            rewards[i] = -0.5

    return rewards


# ---------------------------------------------------------------------------
# Coder reward (stub, filled in Phase 6)
# ---------------------------------------------------------------------------

def coder_compile_test_reward(
    completions: Sequence[str],
    repo: Optional[Sequence[str]] = None,
    repo_path: Optional[Sequence[str]] = None,
    base_commit: Optional[Sequence[str]] = None,
    test_command: Optional[Sequence[str]] = None,
    command_tier: Optional[Sequence[str]] = None,
    timeout_sec: int = 120,
    **kwargs,
) -> List[float]:
    """Reward a coder completion against a compile-and-test verifier.

    Scoring per completion:

    +1.0   patch applies AND compile succeeds AND tests pass
    +0.3   patch applies AND compile succeeds AND tests fail
    -0.5   patch applies BUT compile fails
    -1.0   patch fails to apply, is not a well-formed diff, or parse fails

    This implementation delegates the actual sandboxing to
    ``mud_puppy.coder_sandbox.run_sample``, which wraps bwrap and manages
    per-sample git worktrees. See that module for the hardening details
    and cache layout.

    Dataset columns expected (passed through by TRL when
    ``remove_unused_columns=False`` is set on the GRPOTrainer config):

    ``repo``
        Shorthand key, one of ``llamacpp``/``redis``/``sqlite``/``leveldb``.
        Optional if ``repo_path`` is provided.
    ``repo_path``
        Explicit path to a cached base clone. Either this or ``repo``
        must be present.
    ``base_commit``
        Full SHA the patch expects to apply against. Required.
    ``test_command``
        Optional per-sample override of the test command. If absent,
        the sandbox picks the default for the repo based on
        ``command_tier``.
    ``command_tier``
        Optional, one of ``quick``/``expensive``/``dryrun``. Defaults
        to ``quick`` if both this and ``test_command`` are absent.
    """
    n = len(completions)
    rewards = [-1.0] * n

    # Defer the import so that importing rl_verifier stays cheap for
    # callers (like the reviewer path) that never use this function.
    from .coder_sandbox import run_sample

    def _get(seq, i, default=None):
        if seq is None:
            return default
        if i < len(seq):
            return seq[i]
        return default

    if base_commit is None:
        log.warning("coder_compile_test_reward called without base_commit; "
                    "returning all -1.0")
        return rewards

    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Evaluate each completion in its own worktree. Cv1's sandbox already
    # guarantees worktree isolation via UUID, so N concurrent run_sample
    # calls will not race. The workload is bwrap + subprocess, so threads
    # are fine; the GIL is released during the subprocess wait.
    max_workers = int(os.environ.get("MUD_PUPPY_CODER_MAX_WORKERS", "0")) or n

    def _score(i: int) -> tuple[int, float]:
        patch = completions[i]
        row_repo = _get(repo, i)
        row_repo_path = _get(repo_path, i)
        row_commit = _get(base_commit, i)
        row_cmd = _get(test_command, i)
        row_tier = _get(command_tier, i, "quick")

        if row_commit is None:
            return i, -1.0
        if row_repo is None and row_repo_path is None:
            return i, -1.0

        try:
            result = run_sample(
                patch=patch or "",
                repo=row_repo or "",
                base_commit=row_commit,
                test_command=row_cmd,
                repo_path=row_repo_path,
                timeout_sec=timeout_sec,
                command_tier=row_tier or "quick",
            )
        except Exception as exc:
            log.warning("coder reward eval failed on row %d: %s", i, exc)
            return i, -1.0

        if not result.patch_applied:
            return i, -1.0
        if result.compiled and result.tests_passed:
            return i, 1.0
        if result.compiled and not result.tests_passed:
            return i, 0.3
        # patch applied, compile failed
        return i, -0.5

    # Short-circuit: for n == 1 avoid pool overhead entirely.
    if n == 1 or max_workers == 1:
        for i in range(n):
            _, score = _score(i)
            rewards[i] = score
        return rewards

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for future in as_completed(pool.submit(_score, i) for i in range(n)):
            idx, score = future.result()
            rewards[idx] = score

    return rewards


# ---------------------------------------------------------------------------
# Factory for GRPO wiring
# ---------------------------------------------------------------------------

def reward_factory(kind: str):
    """Return the reward callable matching the requested training kind.

    ``kind`` is one of: 'reviewer_verdict', 'coder_compile_test'.
    """
    kind = kind.lower().strip()
    if kind in {"reviewer_verdict", "reviewer", "verdict"}:
        return reviewer_verdict_reward
    if kind in {"coder_compile_test", "coder", "compile_test"}:
        return coder_compile_test_reward
    raise ValueError(f"Unknown reward kind: {kind}")
