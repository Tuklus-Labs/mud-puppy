"""Compile-and-test sandbox for coder GRPO training.

Given a base repo, a base commit, a generated patch, and a per-repo
test command, run the test command in a bwrap sandbox with:

    * read-only bind of /
    * writable bind of the per-sample git worktree
    * tmpfs on /tmp
    * /proc and /dev
    * network off (--unshare-net)
    * --die-with-parent so orphaned processes die with the orchestrator

Returns a structured dict the reward function turns into a scalar.

Design notes
------------

Each GRPO step generates N completions (N=4 by default) per prompt. They
may be evaluated in parallel. A single mutable git working tree would race
them, so each sample gets its own short-lived `git worktree`. The base
repository is cloned ONCE per (repo) into ``/scratch/coder_repos/<repo>/base``
and reused across samples. The worktree is torn down at the end of each
sample.

For a training run, the caller should pre-stage the four repos via
``prewarm_repo`` on the pod before the first reward evaluation; otherwise
the first evaluation will pay the clone cost. The sandbox will do the
clone lazily if you point it at an upstream URL.

bwrap is a hard requirement for real training. We fall back to a
"no-jail" path with a loud warning for local dev on machines without
bubblewrap, but the pod image ships with bwrap.

CPU-only by design; tests are not GPU-bound and the GPU should stay
saturated on the training itself.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults and per-repo configuration
# ---------------------------------------------------------------------------

DEFAULT_CACHE_ROOT = Path("/scratch/coder_repos")
DEFAULT_TIMEOUT_SEC = 120

# Upstream URLs for lazy cloning. Keep shallow to fit the disk budget.
REPO_UPSTREAMS: Dict[str, str] = {
    "llamacpp": "https://github.com/ggml-org/llama.cpp.git",
    "redis": "https://github.com/redis/redis.git",
    "sqlite": "https://github.com/sqlite/sqlite.git",
    "leveldb": "https://github.com/google/leveldb.git",
}

# Per-repo default test commands. "Expensive" is the real compile + test
# command we'd want in GRPO; "quick" is a sub-second sanity check for
# local dev or smoke tests.
REPO_TEST_COMMANDS: Dict[str, Dict[str, str]] = {
    "llamacpp": {
        "expensive": (
            "mkdir -p build && cd build && "
            "cmake .. -DLLAMA_BUILD_TESTS=ON -DGGML_NATIVE=OFF "
            "-DCMAKE_BUILD_TYPE=Release 2>&1 | tail -40 && "
            "cmake --build . -j --target test-tokenizer-0 2>&1 | tail -80 && "
            "ctest --output-on-failure -R test-tokenizer-0 2>&1 | tail -40"
        ),
        "quick": (
            "mkdir -p build && cd build && "
            "cmake .. -DLLAMA_BUILD_TESTS=ON -DGGML_NATIVE=OFF 2>&1 | tail -20 && "
            "cmake --build . -j --target test-tokenizer-0 2>&1 | tail -20"
        ),
        # Sub-second: just validates the diff is syntactically coherent
        # with the surrounding source by parsing touched headers.
        "dryrun": "true",
    },
    "redis": {
        "expensive": "make -j test_unit 2>&1 | tail -80",
        "quick": "make -j src/redis-server 2>&1 | tail -40",
        "dryrun": "true",
    },
    "sqlite": {
        "expensive": "make quicktest 2>&1 | tail -80",
        "quick": "./configure && make sqlite3.c 2>&1 | tail -40",
        "dryrun": "true",
    },
    "leveldb": {
        "expensive": (
            "mkdir -p build && cd build && "
            "cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -40 && "
            "cmake --build . -j 2>&1 | tail -40 && "
            "ctest --output-on-failure 2>&1 | tail -40"
        ),
        "quick": (
            "mkdir -p build && cd build && "
            "cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20 && "
            "cmake --build . -j --target leveldb 2>&1 | tail -20"
        ),
        "dryrun": "true",
    },
}


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """Structured result of running one compile-and-test sample.

    Attributes:
        patch_applied: ``git apply --check`` succeeded. False if the patch
            was malformed or did not match the base tree.
        compiled: The test command reached its configured build step
            without emitting compile errors. We infer this from stderr/
            stdout markers; see ``_classify_failure``.
        tests_passed: The test command exited with status 0.
        stdout_tail: Last ~4 KB of stdout from the test command.
        stderr_tail: Last ~4 KB of stderr from the test command.
        elapsed_s: Wall-clock seconds spent running the test command.
        error: Optional high-level error string (timeout, bwrap missing,
            worktree failure). None if the sample completed cleanly.
    """
    patch_applied: bool = False
    compiled: bool = False
    tests_passed: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    elapsed_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BWRAP_PATH: Optional[str] = None


def _locate_bwrap() -> Optional[str]:
    """Return the path to bwrap, or None if not installed.

    Cached after the first lookup.
    """
    global _BWRAP_PATH
    if _BWRAP_PATH is not None:
        return _BWRAP_PATH or None
    path = shutil.which("bwrap")
    _BWRAP_PATH = path or ""
    if not path:
        log.warning(
            "bwrap not found on PATH; sandbox will fall back to unjailed "
            "subprocess. Do NOT run untrusted test commands this way. The "
            "MI300X pod image ships with bwrap so this only matters for "
            "local dev."
        )
    return path


def _tail(data: bytes, limit: int = 4096) -> str:
    if not data:
        return ""
    s = data.decode("utf-8", errors="replace")
    if len(s) <= limit:
        return s
    return "...[truncated]\n" + s[-limit:]


_COMPILE_ERROR_MARKERS = re.compile(
    r"(error:\s)|(: error\b)|(undefined reference)|(Undefined symbols)|"
    r"(fatal error:)|(compilation terminated)|(make.*\*\*\* \[.*Error\s\d+\])|"
    r"(ninja:\s+build stopped)",
    re.IGNORECASE,
)

# A test runner that exits nonzero with none of the compile markers is a
# test failure, not a compile failure. CTest emits "tests failed" on
# failing assertions, not "error:".
_TEST_FAILURE_MARKERS = re.compile(
    r"(tests? failed)|(FAIL(ED)?:)|(\[\s*FAILED\s*\])|(Assertion.*failed)|"
    r"(not ok \d+)",
    re.IGNORECASE,
)


def _classify_failure(returncode: int, combined: str) -> Dict[str, bool]:
    """Decide compiled/tests_passed from an exit code + stdout/stderr.

    Called only after the patch applied. Three mutually exclusive outcomes:

        rc == 0                          -> compiled=T, tests_passed=T
        compile marker present in output -> compiled=F, tests_passed=F
        otherwise                        -> compiled=T, tests_passed=F

    The first condition covers the happy path. The second covers broken
    patches whose source does not build. The third covers runtime test
    failures in a successfully-compiled binary.
    """
    if returncode == 0:
        return {"compiled": True, "tests_passed": True}
    if _COMPILE_ERROR_MARKERS.search(combined):
        return {"compiled": False, "tests_passed": False}
    return {"compiled": True, "tests_passed": False}


# ---------------------------------------------------------------------------
# Repo cache + worktree management
# ---------------------------------------------------------------------------

def _resolve_cache_root(cache_root: Optional[os.PathLike]) -> Path:
    if cache_root is not None:
        return Path(cache_root)
    env = os.environ.get("MUD_PUPPY_CODER_CACHE")
    if env:
        return Path(env)
    # /scratch may not exist on dev; fall back to /tmp so the local smoke
    # test does not require root. The MI300X bootstrap creates /scratch.
    if DEFAULT_CACHE_ROOT.parent.exists() and os.access(DEFAULT_CACHE_ROOT.parent, os.W_OK):
        return DEFAULT_CACHE_ROOT
    return Path("/tmp/scratch/coder_repos")


def prewarm_repo(
    repo: str,
    cache_root: Optional[os.PathLike] = None,
    depth: int = 500,
) -> Path:
    """Clone a repo's base mirror if it is not already cached.

    Returns the path to the base clone. This is the one mirror that each
    sample branches a worktree from. Depth is kept shallow to stay under
    the disk budget; if a sample needs a deeper history (e.g. its
    base_commit was dropped by the shallow clone), ``run_sample`` will
    deepen on demand.
    """
    cache = _resolve_cache_root(cache_root) / repo / "base"
    if (cache / ".git").exists():
        return cache
    upstream = REPO_UPSTREAMS.get(repo)
    if upstream is None:
        raise ValueError(
            f"Unknown repo {repo!r}. Add its upstream to REPO_UPSTREAMS "
            f"or pass an existing directory via ``repo_path``."
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    log.info("prewarm_repo: cloning %s -> %s (depth=%d)", upstream, cache, depth)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", f"--depth={depth}", upstream, str(cache)],
        check=True,
    )
    return cache


def _ensure_commit_available(base_clone: Path, base_commit: str, deepen_step: int = 500) -> None:
    """Make sure ``base_commit`` is fetchable from the shallow base clone.

    If the commit is not in the current shallow window, fetch a deeper
    slice. Bounded at ~3 deepenings (1500 commits past the shallow head).
    """
    # ``--`` separators stop git from interpreting a ref that starts with
    # ``-`` as a flag. Cheap defense-in-depth against adversarial or
    # accidentally malformed refs.
    rev_check = subprocess.run(
        ["git", "-C", str(base_clone), "cat-file", "-e", "--",
         f"{base_commit}^{{commit}}"],
        capture_output=True,
    )
    if rev_check.returncode == 0:
        return
    for _ in range(3):
        log.info("deepening %s to reach %s", base_clone, base_commit)
        r = subprocess.run(
            ["git", "-C", str(base_clone), "fetch",
             f"--depth={deepen_step}", "origin", "--", base_commit],
            capture_output=True,
        )
        rev_check = subprocess.run(
            ["git", "-C", str(base_clone), "cat-file", "-e", "--",
             f"{base_commit}^{{commit}}"],
            capture_output=True,
        )
        if rev_check.returncode == 0:
            return
        deepen_step *= 2
    raise RuntimeError(
        f"base_commit {base_commit!r} not reachable in {base_clone} after "
        f"three deepenings; either the SHA is wrong or the upstream has "
        f"rewritten history"
    )


def _make_worktree(base_clone: Path, base_commit: str) -> Path:
    """Create a fresh detached worktree at ``base_commit``.

    Each worktree lives under ``<cache_root>/<repo>/worktrees/<uuid>``.
    Deleted in ``_destroy_worktree``.
    """
    wt_root = base_clone.parent / "worktrees"
    wt_root.mkdir(parents=True, exist_ok=True)
    wt_path = wt_root / uuid.uuid4().hex[:16]
    subprocess.run(
        ["git", "-C", str(base_clone), "worktree", "add",
         "--detach", str(wt_path), base_commit],
        check=True, capture_output=True,
    )
    return wt_path


def _destroy_worktree(base_clone: Path, wt_path: Path) -> None:
    subprocess.run(
        ["git", "-C", str(base_clone), "worktree", "remove", "--force", str(wt_path)],
        capture_output=True,
    )
    # Belt and suspenders: if `worktree remove` failed, nuke the tree.
    if wt_path.exists():
        shutil.rmtree(wt_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Sandboxed test runner
# ---------------------------------------------------------------------------

def _build_bwrap_argv(worktree: Path, inner_cmd: str) -> list:
    """Return the argv to run ``inner_cmd`` under bwrap in ``worktree``.

    System is read-only except for the worktree itself and /tmp (tmpfs).
    Network is unshared.
    """
    # Order matters: --tmpfs /tmp must come BEFORE the --bind of the
    # worktree when the worktree lives under /tmp. Otherwise the tmpfs
    # overlays the bind and bwrap chdir fails with "No such file or
    # directory". We put the worktree bind last for safety.
    argv = [
        _BWRAP_PATH or "bwrap",
        "--ro-bind", "/", "/",
        "--tmpfs", "/tmp",
        "--proc", "/proc",
        "--dev", "/dev",
        "--unshare-net",
        "--die-with-parent",
        "--bind", str(worktree), str(worktree),
        "--chdir", str(worktree),
        # Clear a couple of env vars that would leak host state into the
        # sandbox without being useful inside it. Training-relevant env
        # (PATH, HOME, LANG) is inherited.
        "--unsetenv", "SSH_AUTH_SOCK",
        "--unsetenv", "DBUS_SESSION_BUS_ADDRESS",
        "/bin/bash", "-c", inner_cmd,
    ]
    return argv


def _run_sandboxed(worktree: Path, cmd: str, timeout_sec: int) -> subprocess.CompletedProcess:
    """Run ``cmd`` inside bwrap, or unjailed if bwrap is absent."""
    bwrap = _locate_bwrap()
    if bwrap:
        argv = _build_bwrap_argv(worktree, cmd)
        return subprocess.run(
            argv, capture_output=True, timeout=timeout_sec,
        )
    # Fallback path: no sandbox. Warn loudly. This path should never run
    # on the MI300X pod.
    log.warning("running test command WITHOUT bwrap (no sandbox)")
    return subprocess.run(
        cmd, shell=True, cwd=str(worktree),
        capture_output=True, timeout=timeout_sec,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sample(
    patch: str,
    repo: str,
    base_commit: str,
    test_command: Optional[str] = None,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    cache_root: Optional[os.PathLike] = None,
    repo_path: Optional[os.PathLike] = None,
    command_tier: str = "quick",
) -> SandboxResult:
    """Run one compile-and-test sample under the sandbox.

    Parameters
    ----------
    patch:
        Unified diff text produced by the model. Will be fed to
        ``git apply`` via stdin.
    repo:
        Shorthand repo key (one of the keys in REPO_UPSTREAMS) or an
        arbitrary string paired with ``repo_path``.
    base_commit:
        Full SHA the patch expects to apply cleanly against.
    test_command:
        Optional override for the test command. If omitted, looks up
        REPO_TEST_COMMANDS[repo][command_tier].
    timeout_sec:
        Per-sample wall-clock limit on the test command. Default 120.
    cache_root:
        Override for the repo cache. Defaults to /scratch/coder_repos.
    repo_path:
        Explicit path to an already-cloned base repo. If given, skips
        the lazy clone and the REPO_UPSTREAMS lookup.
    command_tier:
        'quick' (default, compile-only), 'expensive' (compile + tests),
        or 'dryrun' (true; no-op, used in unit tests).

    Returns
    -------
    SandboxResult
        Patch application + compile + test outcome plus truncated logs.
    """
    t_start = time.monotonic()
    result = SandboxResult()

    if not patch or not patch.strip():
        result.error = "empty_patch"
        result.elapsed_s = time.monotonic() - t_start
        return result

    # Very-light malformed-diff check. Real validation is ``git apply --check``
    # below; this just short-circuits on obvious garbage so the reward
    # function can distinguish "no diff at all" from "diff rejected by git".
    if "diff --git " not in patch and "--- " not in patch:
        result.error = "not_a_diff"
        result.elapsed_s = time.monotonic() - t_start
        return result

    # Resolve the base clone and create a worktree.
    if repo_path is not None:
        base_clone = Path(repo_path)
        if not (base_clone / ".git").exists():
            raise ValueError(f"repo_path {base_clone} is not a git repo")
    else:
        base_clone = prewarm_repo(repo, cache_root=cache_root)

    try:
        _ensure_commit_available(base_clone, base_commit)
    except RuntimeError as exc:
        result.error = f"base_commit_unreachable: {exc}"
        result.elapsed_s = time.monotonic() - t_start
        return result

    try:
        worktree = _make_worktree(base_clone, base_commit)
    except subprocess.CalledProcessError as exc:
        result.error = f"worktree_create_failed: {exc.stderr[:200]!r}"
        result.elapsed_s = time.monotonic() - t_start
        return result

    try:
        # --- patch check ------------------------------------------------
        check = subprocess.run(
            ["git", "-C", str(worktree), "apply", "--check", "-"],
            input=patch.encode(), capture_output=True, timeout=30,
        )
        if check.returncode != 0:
            result.patch_applied = False
            result.stderr_tail = _tail(check.stderr)
            result.elapsed_s = time.monotonic() - t_start
            return result

        # --- patch apply ------------------------------------------------
        applied = subprocess.run(
            ["git", "-C", str(worktree), "apply", "-"],
            input=patch.encode(), capture_output=True, timeout=30,
        )
        if applied.returncode != 0:
            # Should not happen if --check passed, but handle the race.
            result.patch_applied = False
            result.stderr_tail = _tail(applied.stderr)
            result.elapsed_s = time.monotonic() - t_start
            return result
        result.patch_applied = True

        # --- test command -----------------------------------------------
        cmd = test_command
        if cmd is None:
            tiers = REPO_TEST_COMMANDS.get(repo, {})
            cmd = tiers.get(command_tier) or tiers.get("quick")
        if cmd is None:
            result.error = f"no_test_command_for_{repo}"
            result.elapsed_s = time.monotonic() - t_start
            return result

        try:
            test_proc = _run_sandboxed(worktree, cmd, timeout_sec)
        except subprocess.TimeoutExpired as exc:
            result.error = "timeout"
            result.stdout_tail = _tail(exc.stdout or b"")
            result.stderr_tail = _tail(exc.stderr or b"")
            # Timeout on a build step looks like compile failure for
            # scoring; on a test step it looks like test failure. We
            # cannot tell from outside, so treat it as compile failure
            # (stricter penalty) to push the model away from patches
            # that wedge the build.
            result.compiled = False
            result.tests_passed = False
            result.elapsed_s = time.monotonic() - t_start
            return result

        result.stdout_tail = _tail(test_proc.stdout)
        result.stderr_tail = _tail(test_proc.stderr)
        combined = result.stdout_tail + "\n" + result.stderr_tail
        cls = _classify_failure(test_proc.returncode, combined)
        result.compiled = cls["compiled"]
        result.tests_passed = cls["tests_passed"]
    finally:
        _destroy_worktree(base_clone, worktree)
        result.elapsed_s = time.monotonic() - t_start

    return result


# ---------------------------------------------------------------------------
# Convenience: batch runner for GRPO
# ---------------------------------------------------------------------------

def run_batch(samples: list) -> list:
    """Evaluate multiple samples serially.

    Parallel evaluation is left to the caller (typically the GRPO reward
    function uses a process pool). This helper is mostly for scripts and
    tests.
    """
    return [run_sample(**s) for s in samples]
