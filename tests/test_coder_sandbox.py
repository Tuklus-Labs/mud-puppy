"""Unit tests for mud_puppy.coder_sandbox and the coder reward path.

No real clones. Every test builds a throwaway git repo in a tempdir.
Total suite target: <30 seconds on a warm disk.

Run: pytest tests/test_coder_sandbox.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from mud_puppy.coder_sandbox import (
    SandboxResult,
    _classify_failure,
    _locate_bwrap,
    _tail,
    run_sample,
)
from mud_puppy.rl_verifier import coder_compile_test_reward


# ---------------------------------------------------------------------------
# Fixtures: build a fake git repo with a tiny C program and a Makefile
# ---------------------------------------------------------------------------

C_MAIN_V1 = """\
#include <stdio.h>

int add(int a, int b) { return a + b; }

int main(void) {
    if (add(2, 3) != 5) return 1;
    printf("ok\\n");
    return 0;
}
"""

MAKEFILE = """\
all: prog

prog: main.c
\tcc -o prog main.c

test: prog
\t./prog

clean:
\trm -f prog
"""


@pytest.fixture
def fake_repo(tmp_path):
    """Create a tiny C repo with one commit. Return its path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.c").write_text(C_MAIN_V1)
    (repo / "Makefile").write_text(MAKEFILE)

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial"],
        cwd=repo, check=True, env=env,
    )
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    return repo, sha


def _make_diff(repo: Path, file_rel: str, old: str, new: str) -> str:
    """Build a unified diff by writing a tmp file and diffing via git.

    We use ``git diff --no-index`` so the output format matches what
    ``git apply`` expects.
    """
    # Write the new content to a temp file next to the repo.
    scratch = repo.parent / "scratch"
    scratch.mkdir(exist_ok=True)
    (scratch / file_rel).parent.mkdir(parents=True, exist_ok=True)
    (scratch / file_rel).write_text(new)
    # Raw diff
    proc = subprocess.run(
        ["git", "diff", "--no-index", "--src-prefix=a/", "--dst-prefix=b/",
         str(repo / file_rel), str(scratch / file_rel)],
        capture_output=True, text=True,
    )
    raw = proc.stdout
    # The raw diff references absolute paths; rewrite them to repo-relative.
    raw = raw.replace(f"a{repo}/{file_rel}", f"a/{file_rel}")
    raw = raw.replace(f"b{scratch}/{file_rel}", f"b/{file_rel}")
    # Also fix the diff --git header
    raw = raw.replace(
        f"diff --git a{repo}/{file_rel} b{scratch}/{file_rel}",
        f"diff --git a/{file_rel} b/{file_rel}",
    )
    return raw


# ---------------------------------------------------------------------------
# Classifier / helper tests
# ---------------------------------------------------------------------------

class TestClassifyFailure:
    def test_exit_zero_is_pass(self):
        r = _classify_failure(0, "Build succeeded. All tests passed.")
        assert r == {"compiled": True, "tests_passed": True}

    def test_compile_error(self):
        r = _classify_failure(2, "foo.c:42: error: 'x' undeclared")
        assert r == {"compiled": False, "tests_passed": False}

    def test_undefined_reference(self):
        r = _classify_failure(1, "ld: undefined reference to `bar'")
        assert r == {"compiled": False, "tests_passed": False}

    def test_test_failure_but_compiled(self):
        r = _classify_failure(1, "all compiled. 3 tests failed.")
        assert r == {"compiled": True, "tests_passed": False}

    def test_runtime_nonzero_no_markers(self):
        # A bare nonzero exit with no recognizable markers is a test
        # failure (we compiled something, it ran, it returned nonzero).
        r = _classify_failure(1, "")
        assert r == {"compiled": True, "tests_passed": False}


def test_tail_truncates():
    out = _tail(b"x" * 10000, limit=100)
    assert out.startswith("...[truncated]")
    assert len(out) < 200


def test_tail_empty():
    assert _tail(b"") == ""


# ---------------------------------------------------------------------------
# run_sample: real git, real compile, no bwrap jail required
# ---------------------------------------------------------------------------

class TestRunSample:
    def test_valid_patch_compiles_and_passes(self, fake_repo):
        repo, sha = fake_repo
        # A trivial but valid edit: change the success message.
        new_main = C_MAIN_V1.replace('printf("ok\\n");', 'printf("yes\\n");')
        patch = _make_diff(repo, "main.c", C_MAIN_V1, new_main)
        assert "diff --git" in patch

        result = run_sample(
            patch=patch,
            repo="fake",
            base_commit=sha,
            test_command="make test",
            repo_path=repo,
            timeout_sec=30,
        )
        assert isinstance(result, SandboxResult)
        assert result.patch_applied is True
        assert result.compiled is True
        assert result.tests_passed is True
        assert result.error is None

    def test_malformed_patch(self, fake_repo):
        repo, sha = fake_repo
        patch = "not a diff at all just some garbage text\n"

        result = run_sample(
            patch=patch,
            repo="fake",
            base_commit=sha,
            test_command="true",
            repo_path=repo,
            timeout_sec=10,
        )
        assert result.patch_applied is False
        assert result.compiled is False
        assert result.tests_passed is False
        # Short-circuited with not_a_diff error; no runtime happened.
        assert result.error == "not_a_diff"

    def test_empty_patch(self, fake_repo):
        repo, sha = fake_repo
        result = run_sample(
            patch="",
            repo="fake",
            base_commit=sha,
            test_command="true",
            repo_path=repo,
        )
        assert result.patch_applied is False
        assert result.error == "empty_patch"

    def test_patch_that_breaks_compile(self, fake_repo):
        repo, sha = fake_repo
        # Replace `add(int a, int b)` with a deliberately broken signature.
        broken = C_MAIN_V1.replace(
            "int add(int a, int b) { return a + b; }",
            "int add(int a, int b) { return a + b + missing_var; }",
        )
        patch = _make_diff(repo, "main.c", C_MAIN_V1, broken)

        result = run_sample(
            patch=patch,
            repo="fake",
            base_commit=sha,
            test_command="make test",
            repo_path=repo,
            timeout_sec=30,
        )
        assert result.patch_applied is True
        assert result.compiled is False
        assert result.tests_passed is False

    def test_patch_that_compiles_but_tests_fail(self, fake_repo):
        repo, sha = fake_repo
        # Break the runtime assertion: add(2,3) will return 6, so main returns 1.
        broken = C_MAIN_V1.replace(
            "return a + b;", "return a + b + 1;",
        )
        patch = _make_diff(repo, "main.c", C_MAIN_V1, broken)

        result = run_sample(
            patch=patch,
            repo="fake",
            base_commit=sha,
            test_command="make test",
            repo_path=repo,
            timeout_sec=30,
        )
        assert result.patch_applied is True
        assert result.compiled is True
        assert result.tests_passed is False

    def test_patch_that_does_not_match(self, fake_repo):
        repo, sha = fake_repo
        # A diff against content that does not exist in the base.
        fake_diff = textwrap.dedent("""\
            diff --git a/main.c b/main.c
            --- a/main.c
            +++ b/main.c
            @@ -1,1 +1,1 @@
            -this line does not exist in the repo
            +replacement
        """)
        result = run_sample(
            patch=fake_diff,
            repo="fake",
            base_commit=sha,
            test_command="true",
            repo_path=repo,
            timeout_sec=10,
        )
        assert result.patch_applied is False

    def test_timeout(self, fake_repo):
        repo, sha = fake_repo
        # Valid no-op patch path via a tiny change, then a wedged test.
        new_main = C_MAIN_V1.replace('printf("ok\\n");', 'printf("maybe\\n");')
        patch = _make_diff(repo, "main.c", C_MAIN_V1, new_main)

        # Sleep longer than the timeout. bwrap does not help sleep run
        # faster, so this reliably trips the timeout path.
        result = run_sample(
            patch=patch,
            repo="fake",
            base_commit=sha,
            test_command="sleep 5",
            repo_path=repo,
            timeout_sec=2,
        )
        assert result.patch_applied is True
        assert result.error == "timeout"
        assert result.compiled is False
        assert result.tests_passed is False


# ---------------------------------------------------------------------------
# reward function plumbing
# ---------------------------------------------------------------------------

class TestCoderReward:
    def test_reward_scoring_on_valid_patch(self, fake_repo):
        repo, sha = fake_repo
        new_main = C_MAIN_V1.replace('printf("ok\\n");', 'printf("yes\\n");')
        patch = _make_diff(repo, "main.c", C_MAIN_V1, new_main)

        rewards = coder_compile_test_reward(
            completions=[patch],
            repo_path=[str(repo)],
            base_commit=[sha],
            test_command=["make test"],
            timeout_sec=30,
        )
        assert rewards == [1.0]

    def test_reward_on_broken_compile(self, fake_repo):
        repo, sha = fake_repo
        broken = C_MAIN_V1.replace(
            "int add(int a, int b) { return a + b; }",
            "int add(int a, int b) { return a + b + missing_var; }",
        )
        patch = _make_diff(repo, "main.c", C_MAIN_V1, broken)

        rewards = coder_compile_test_reward(
            completions=[patch],
            repo_path=[str(repo)],
            base_commit=[sha],
            test_command=["make test"],
            timeout_sec=30,
        )
        assert rewards == [-0.5]

    def test_reward_on_test_failure(self, fake_repo):
        repo, sha = fake_repo
        broken = C_MAIN_V1.replace("return a + b;", "return a + b + 1;")
        patch = _make_diff(repo, "main.c", C_MAIN_V1, broken)

        rewards = coder_compile_test_reward(
            completions=[patch],
            repo_path=[str(repo)],
            base_commit=[sha],
            test_command=["make test"],
            timeout_sec=30,
        )
        assert rewards == [0.3]

    def test_reward_on_malformed_diff(self, fake_repo):
        repo, sha = fake_repo
        rewards = coder_compile_test_reward(
            completions=["this is clearly not a diff"],
            repo_path=[str(repo)],
            base_commit=[sha],
            test_command=["true"],
        )
        assert rewards == [-1.0]

    def test_reward_requires_base_commit(self, fake_repo):
        repo, _sha = fake_repo
        rewards = coder_compile_test_reward(
            completions=["anything"],
            repo_path=[str(repo)],
            base_commit=None,
            test_command=["true"],
        )
        assert rewards == [-1.0]

    def test_reward_requires_repo_identity(self, fake_repo):
        repo, sha = fake_repo
        rewards = coder_compile_test_reward(
            completions=["anything"],
            repo_path=None,
            repo=None,
            base_commit=[sha],
            test_command=["true"],
        )
        assert rewards == [-1.0]


# ---------------------------------------------------------------------------
# bwrap probe (non-failing)
# ---------------------------------------------------------------------------

def test_bwrap_probe_non_failing():
    # Whether bwrap is present or not, the probe must not raise.
    path = _locate_bwrap()
    assert path is None or Path(path).exists()
