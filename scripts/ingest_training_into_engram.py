#!/usr/bin/env python3
"""Push reviewer + coder training JSONLs into Engram as tagged atoms.

Cold-starts the Pensive graph so RO has real examples to retrieve during
the aegis-viz dogfood and subsequent runs. Each source JSONL line becomes
one atom with:

  kind      = "discovery" (good pattern) or "failure" (bug-injection negatives)
  project   = "training:<source>", e.g. "training:llamacpp-commits"
  shape     = short abstract of the row (prompt or task description)
  principle = the transferable lesson (target verdict, commit message, review)
  tags      = domain tags (reviewer/coder, language, repo, task kind)
  source    = provenance ("training_data:<filename>") so downstream can filter

We sample ~500-1500 rows per source file rather than ingest everything.
Retrieval cost scales with corpus size; a few thousand high-signal atoms
beats 100K noisy ones. Increase MAX_PER_SOURCE for more coverage once we
confirm retrieval quality is good.

Uses the `engram-emit` CLI (symlinked at ~/bin/engram-emit) as the write
interface. Best-effort: logs and continues on failure.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path("/home/aegis/Projects/mud-puppy/training_data_sets")

# Per-source row caps. llama.cpp gets the biggest share because the dogfood
# target (aegis-viz) is C++/Vulkan, which overlaps its corpus directly.
SOURCE_CAPS: Dict[str, int] = {
    "reviewer/swe-bench-train.jsonl":          600,
    "reviewer/bug-injection-multilang.jsonl":  400,
    "reviewer/pr-reviews-osint.jsonl":         500,
    "reviewer/codereviewer-sft-warmup.jsonl":  500,
    "coder/llamacpp-commits.jsonl":           1500,
    "coder/redis-commits.jsonl":               400,
    "coder/sqlite-commits.jsonl":              400,
    "coder/leveldb-commits.jsonl":              80,
}

# Per-source principle / tag / kind extractors. Each source file has a
# slightly different schema after Agent D's curation; we pull the right
# field from each into the atom layout.

def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _prompt_text(row: Dict) -> str:
    p = row.get("prompt", "")
    ctx = row.get("context")
    if isinstance(ctx, dict):
        # join a couple of the most useful keys
        for key in ("old_file", "instruction", "task", "problem_statement"):
            if key in ctx and ctx[key]:
                return f"{p}\n\n{key}: {str(ctx[key])}"
        return p + "\n\n" + json.dumps(ctx, ensure_ascii=False)[:1000]
    if isinstance(ctx, str):
        return f"{p}\n\n{ctx}"
    return p


def _target_text(row: Dict) -> str:
    t = row.get("target", "")
    if isinstance(t, (dict, list)):
        return json.dumps(t, ensure_ascii=False)
    return str(t)


def extract_reviewer_atom(row: Dict, source_file: str) -> Optional[Dict]:
    """Reviewer row -> atom shape. Bug injection emits 'failure' atoms;
    everything else emits 'discovery' atoms."""
    task = row.get("task", "")
    shape = _truncate(_prompt_text(row), 500)
    principle = _truncate(_target_text(row), 800)
    if not shape or not principle:
        return None

    if "bug-injection" in source_file:
        kind = "failure"
    else:
        kind = "discovery"

    tags = ["reviewer", task]
    meta = row.get("metadata") or {}
    if isinstance(meta, dict):
        for k in ("language", "repo", "source_file", "bug_class"):
            v = meta.get(k)
            if isinstance(v, str) and v:
                tags.append(f"{k}:{v}")

    return {
        "kind": kind,
        "project": f"training:{Path(source_file).stem}",
        "shape": shape,
        "principle": principle,
        "tags": tags,
    }


def extract_coder_atom(row: Dict, source_file: str) -> Optional[Dict]:
    """Coder row -> atom shape. Commit trajectories become 'discovery'
    atoms whose principle is the commit message + a summary of the diff
    shape."""
    target = row.get("target") or {}
    if isinstance(target, str):
        try:
            target = json.loads(target)
        except Exception:
            target = {"raw": target}

    prompt = row.get("prompt", "")
    commit_message = target.get("commit_message") if isinstance(target, dict) else None
    diff = target.get("diff") if isinstance(target, dict) else None

    shape = _truncate(prompt, 500)
    if not shape:
        return None

    # Principle is the commit message; diff snippet is appended to give
    # concrete code context when retrieval surfaces this atom.
    diff_snippet = _truncate(diff or "", 700)
    if commit_message:
        principle = commit_message
        if diff_snippet:
            principle = f"{commit_message}\n\n---\n{diff_snippet}"
    else:
        principle = diff_snippet or _target_text(row)
    principle = _truncate(principle, 1500)
    if not principle:
        return None

    repo = Path(source_file).stem.replace("-commits", "")
    tags = ["coder", "commit", f"repo:{repo}"]
    meta = row.get("metadata") or {}
    if isinstance(meta, dict):
        for k in ("language", "files_touched", "size_class"):
            v = meta.get(k)
            if isinstance(v, str) and v:
                tags.append(f"{k}:{v}")

    return {
        "kind": "discovery",
        "project": f"training:{Path(source_file).stem}",
        "shape": shape,
        "principle": principle,
        "tags": tags,
    }


EXTRACTORS: Dict[str, Callable[[Dict, str], Optional[Dict]]] = {
    "reviewer/swe-bench-train.jsonl":          extract_reviewer_atom,
    "reviewer/bug-injection-multilang.jsonl":  extract_reviewer_atom,
    "reviewer/pr-reviews-osint.jsonl":         extract_reviewer_atom,
    "reviewer/codereviewer-sft-warmup.jsonl":  extract_reviewer_atom,
    "coder/llamacpp-commits.jsonl":            extract_coder_atom,
    "coder/redis-commits.jsonl":               extract_coder_atom,
    "coder/sqlite-commits.jsonl":              extract_coder_atom,
    "coder/leveldb-commits.jsonl":             extract_coder_atom,
}


def emit_atom(atom: Dict) -> bool:
    """Invoke `engram-emit <kind>` with the atom fields. Returns True on
    success. Survives CLI failures by logging and continuing.

    engram-emit discovery/failure subcommands accept flags:
        --project, --principle, --context, --tags (comma-separated),
        --domain, --topic, --session-id, --narrative, --stakes
    We use:
        --project    -> training:<source-stem>
        --principle  -> the transferable rule (target verdict / commit msg)
        --context    -> the abstract shape (prompt / task description)
        --tags       -> domain tags joined by comma
        --domain     -> "reviewer" or "coder"
        --topic      -> a short topic label for retrieval keying
    """
    kind = atom["kind"]
    project = atom["project"]
    domain = "reviewer" if "reviewer" in atom.get("tags", []) else "coder"
    tags_csv = ",".join(t for t in atom.get("tags", []) if t)
    topic = tags_csv.split(",")[0] if tags_csv else domain

    cmd = [
        "engram-emit", kind,
        "--project", project,
        "--principle", atom["principle"],
        "--context", atom["shape"],
        "--tags", tags_csv + ",source:training_data",
        "--domain", domain,
        "--topic", topic,
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            timeout=10,
            text=True,
        )
        if r.returncode != 0:
            sys.stderr.write(f"[ingest] emit failed rc={r.returncode} stderr={r.stderr[:200]}\n")
            return False
        return True
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"[ingest] emit timeout for project={project}\n")
        return False
    except FileNotFoundError:
        sys.stderr.write("[ingest] engram-emit not in PATH; aborting\n")
        sys.exit(2)


def ingest_file(rel_path: str, cap: int, seed: int) -> int:
    abs_path = ROOT / rel_path
    if not abs_path.is_file():
        sys.stderr.write(f"[ingest] SKIP missing: {abs_path}\n")
        return 0

    # Read line count first for unbiased reservoir-ish sampling.
    with abs_path.open() as f:
        total = sum(1 for _ in f)
    if total == 0:
        return 0

    # If cap exceeds total, take all. Else uniform-random sample of cap.
    rng = random.Random(seed + hash(rel_path))
    if cap >= total:
        chosen = set(range(total))
    else:
        chosen = set(rng.sample(range(total), cap))

    extractor = EXTRACTORS[rel_path]
    emitted = 0
    skipped = 0
    start = time.time()

    with abs_path.open() as f:
        for idx, line in enumerate(f):
            if idx not in chosen:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            atom = extractor(row, rel_path)
            if atom is None:
                skipped += 1
                continue
            if emit_atom(atom):
                emitted += 1
            else:
                skipped += 1

    elapsed = time.time() - start
    rate = emitted / max(elapsed, 0.001)
    print(f"[ingest] {rel_path:55s} emitted={emitted:5d}  skipped={skipped:4d}  elapsed={elapsed:.1f}s  ({rate:.1f}/s)")
    return emitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260423)
    ap.add_argument("--only", help="comma-separated source filenames to limit to")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    only = set((args.only or "").split(",")) if args.only else None
    total = 0

    for rel, cap in SOURCE_CAPS.items():
        if only and Path(rel).name not in only:
            continue
        if args.dry_run:
            print(f"[dry-run] would ingest up to {cap} atoms from {rel}")
            continue
        total += ingest_file(rel, cap, args.seed)

    print(f"[ingest] DONE. total atoms emitted: {total}")


if __name__ == "__main__":
    main()
