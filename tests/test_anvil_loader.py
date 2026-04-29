"""Tests for ``mud_puppy.anvil_loader``.

Validates the JSON load path, kernel-hash invalidation, malformed-JSON
fallback, bucket math, and the in-process cache. None of these need a
GPU or Triton -- the loader is pure Python plus a sha256 hash over a
couple of source files.

Fixture JSONs live under ``tests/fixtures/``. The canonical fixture
(``anvil_train_qwen3_mxfp4.json``) carries a placeholder
``kernel_hash`` value so it can travel through git without going stale
every time someone touches a kernel file. Tests that exercise the
"valid hash" branch read the fixture, swap the placeholder for the live
``compute_kernel_hash()`` value, and write the result to a tmp path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from mud_puppy import anvil_loader
from mud_puppy.anvil_loader import (
    AnvilTrainConfig,
    BUCKET_BOUNDARIES,
    NUM_BUCKETS,
    SCHEMA_V1,
    bucket_index,
    cache_path_for_model,
    compute_kernel_hash,
    load_for_model,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _clear_anvil_cache():
    """Ensure each test starts with a fresh in-process load cache."""
    anvil_loader.clear_cache()
    yield
    anvil_loader.clear_cache()


def _write_fixture_with_live_hash(tmp_path: Path) -> Path:
    """Copy the canonical fixture but substitute the placeholder hash.

    Returns the path to the rewritten fixture so tests can load it.
    """
    raw = (FIXTURE_DIR / "anvil_train_qwen3_mxfp4.json").read_text()
    payload = json.loads(raw)
    payload["kernel_hash"] = compute_kernel_hash()
    out = tmp_path / "anvil_train_qwen3_mxfp4_live.json"
    out.write_text(json.dumps(payload))
    return out


# ---------------------------------------------------------------------------
# Bucket math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    (1, 0),
    (1024, 0),
    (1025, 1),
    (4096, 1),
    (4097, 2),
    (8192, 2),
    (8193, 3),
    (16384, 3),
    (16385, 4),
    (1_000_000, 4),
    # Edge: zero / negative collapse to bucket 0.
    (0, 0),
    (-99, 0),
])
def test_bucket_index(value, expected):
    assert bucket_index(value) == expected


def test_num_buckets_matches_boundary_count():
    """5 buckets means 4 boundaries (n+1 = NUM_BUCKETS)."""
    assert len(BUCKET_BOUNDARIES) == NUM_BUCKETS - 1


# ---------------------------------------------------------------------------
# Kernel-hash computation
# ---------------------------------------------------------------------------


def test_compute_kernel_hash_is_stable():
    """Calling twice yields the same hash (no per-call randomness)."""
    h1 = compute_kernel_hash()
    h2 = compute_kernel_hash()
    assert h1 == h2
    assert h1.startswith("sha256:")
    # 64 hex chars after the prefix.
    assert len(h1) == len("sha256:") + 64


# ---------------------------------------------------------------------------
# AnvilTrainConfig.load
# ---------------------------------------------------------------------------


def test_load_missing_file_returns_none(tmp_path):
    p = tmp_path / "does-not-exist.json"
    assert AnvilTrainConfig.load(p) is None


def test_load_malformed_json_returns_none(caplog):
    """Garbage JSON triggers a warning but never raises."""
    p = FIXTURE_DIR / "anvil_train_malformed.json"
    with caplog.at_level(logging.WARNING):
        result = AnvilTrainConfig.load(p)
    assert result is None
    assert any("failed to parse" in rec.message for rec in caplog.records)


def test_load_wrong_schema_returns_none(caplog):
    """Schema mismatch is a hard reject -- caller falls through to autotune."""
    p = FIXTURE_DIR / "anvil_train_wrong_schema.json"
    with caplog.at_level(logging.WARNING):
        result = AnvilTrainConfig.load(p)
    assert result is None
    assert any("schema" in rec.message for rec in caplog.records)


def test_load_kernel_hash_mismatch_returns_none(tmp_path, caplog):
    """A kernel-source change invalidates the cached configs."""
    raw = (FIXTURE_DIR / "anvil_train_qwen3_mxfp4.json").read_text()
    payload = json.loads(raw)
    payload["kernel_hash"] = "sha256:" + "0" * 64  # any bogus hash
    p = tmp_path / "stale.json"
    p.write_text(json.dumps(payload))

    with caplog.at_level(logging.WARNING):
        result = AnvilTrainConfig.load(p)
    assert result is None
    assert any("kernel_hash" in rec.message for rec in caplog.records)


def test_load_valid_fixture(tmp_path):
    """Happy path: a fixture with the live kernel_hash loads successfully."""
    p = _write_fixture_with_live_hash(tmp_path)
    cfg = AnvilTrainConfig.load(p)
    assert cfg is not None
    assert cfg.schema == SCHEMA_V1
    assert cfg.gpu == "gfx1100"
    assert cfg.model == "Qwen3-8B"
    assert cfg.batch == 1
    assert cfg.seq == 4096


def test_load_caches_in_process(tmp_path, monkeypatch):
    """A second load() of the same path returns the same object."""
    p = _write_fixture_with_live_hash(tmp_path)
    cfg1 = AnvilTrainConfig.load(p)
    cfg2 = AnvilTrainConfig.load(p)
    assert cfg1 is cfg2

    # Mutating the file on disk after the first load must NOT change the
    # cached object -- in-process callers see the version they loaded.
    p.write_text("garbage that would otherwise fail to parse")
    cfg3 = AnvilTrainConfig.load(p)
    assert cfg3 is cfg1


def test_load_caches_negative_results(tmp_path):
    """Misses are also cached so repeat lookups don't re-stat the FS."""
    p = tmp_path / "nope.json"
    assert AnvilTrainConfig.load(p) is None
    # Create the file -- but the cache still says None.
    payload = {"schema": SCHEMA_V1, "kernel_hash": compute_kernel_hash(), "ops": {}}
    p.write_text(json.dumps(payload))
    assert AnvilTrainConfig.load(p) is None  # still negative


# ---------------------------------------------------------------------------
# get_kernel_config
# ---------------------------------------------------------------------------


def test_get_kernel_config_hits_known_cell(tmp_path):
    p = _write_fixture_with_live_hash(tmp_path)
    cfg = AnvilTrainConfig.load(p)
    assert cfg is not None

    # Bucket "1,1,1" = M in (1024, 4096], N in (1024, 4096], K in (1024, 4096].
    # Pick representative values right in the middle of each.
    out = cfg.get_kernel_config("mxfp4_fwd", M=4096, N=4096, K=4096)
    assert out is not None
    assert out["BLOCK_M"] == 128
    assert out["BLOCK_N"] == 64
    assert out["BLOCK_K"] == 32
    assert out["GROUP_M"] == 8
    assert out["num_warps"] == 8
    assert out["num_stages"] == 4


def test_get_kernel_config_buckets_correctly(tmp_path):
    """Different (M, N, K) values that hit the same cell return the same config."""
    p = _write_fixture_with_live_hash(tmp_path)
    cfg = AnvilTrainConfig.load(p)
    a = cfg.get_kernel_config("mxfp4_fwd", M=2048, N=2048, K=2048)
    b = cfg.get_kernel_config("mxfp4_fwd", M=4096, N=4096, K=4096)
    assert a is not None and b is not None
    assert a == b  # same cell ("1,1,1")


def test_get_kernel_config_miss_returns_none(tmp_path):
    """An unmapped cell returns None so the caller falls through to autotune."""
    p = _write_fixture_with_live_hash(tmp_path)
    cfg = AnvilTrainConfig.load(p)
    # Bucket "4,4,4" = M, N, K > 16384. Not present in the fixture.
    out = cfg.get_kernel_config("mxfp4_fwd", M=20000, N=20000, K=20000)
    assert out is None


def test_get_kernel_config_unknown_op_returns_none(tmp_path):
    p = _write_fixture_with_live_hash(tmp_path)
    cfg = AnvilTrainConfig.load(p)
    out = cfg.get_kernel_config("totally_made_up_op", M=4096, N=4096, K=4096)
    assert out is None


def test_get_kernel_config_returns_int4_cells(tmp_path):
    """The same fixture carries int4 ops too; both paths share the loader."""
    p = _write_fixture_with_live_hash(tmp_path)
    cfg = AnvilTrainConfig.load(p)
    assert cfg.get_kernel_config("int4_fwd", M=4096, N=4096, K=4096) is not None
    assert cfg.get_kernel_config("int4_grad_input", M=4096, N=4096, K=4096) is not None


def test_num_configs_counts_all_cells(tmp_path):
    p = _write_fixture_with_live_hash(tmp_path)
    cfg = AnvilTrainConfig.load(p)
    # Fixture: 3 mxfp4_fwd + 1 mxfp4_grad_input + 1 int4_fwd + 1 int4_grad_input = 6.
    assert cfg.num_configs() == 6


# ---------------------------------------------------------------------------
# cache_path_for_model + load_for_model
# ---------------------------------------------------------------------------


def test_cache_path_strips_hf_repo_prefix():
    p = cache_path_for_model("meta-llama/Llama-3-8B", batch=1, seq=4096, quant="mxfp4", gpu_gfx="gfx1100")
    assert p.name == "Llama-3-8B-mxfp4-b1s4096.json"
    assert "gfx1100" in str(p)


def test_cache_path_no_gpu():
    p = cache_path_for_model("modelname", batch=2, seq=2048, quant="int4", gpu_gfx="")
    assert "gfx" not in str(p)
    assert p.name == "modelname-int4-b2s2048.json"


def test_cache_path_handles_local_path():
    p = cache_path_for_model("/home/aegis/Models/Qwen3-8B/", batch=1, seq=4096, quant="mxfp4", gpu_gfx="gfx1100")
    assert p.name.startswith("Qwen3-8B-")


def test_load_for_model_missing_returns_none(tmp_path, monkeypatch):
    """When the cache file doesn't exist, load_for_model returns None."""
    # Redirect ~/.cache to tmp so the test doesn't touch the real cache.
    monkeypatch.setenv("HOME", str(tmp_path))
    out = load_for_model("totally-fake-model", batch=1, seq=4096, quant="mxfp4", gpu_gfx="gfx1100")
    assert out is None


def test_load_for_model_finds_fixture(tmp_path, monkeypatch):
    """Place a valid fixture at the canonical cache path; loader picks it up."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cache_dir = tmp_path / ".cache" / "anvil-train" / "gfx1100"
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / "Qwen3-8B-mxfp4-b1s4096.json"
    payload = json.loads((FIXTURE_DIR / "anvil_train_qwen3_mxfp4.json").read_text())
    payload["kernel_hash"] = compute_kernel_hash()
    cache_file.write_text(json.dumps(payload))

    out = load_for_model("Qwen3-8B", batch=1, seq=4096, quant="mxfp4", gpu_gfx="gfx1100")
    assert out is not None
    assert out.model == "Qwen3-8B"


# ---------------------------------------------------------------------------
# apply_to_model integration
# ---------------------------------------------------------------------------


def test_apply_to_model_returns_zero_when_no_cache(tmp_path, monkeypatch):
    """apply_to_model never raises when the JSON is missing."""
    monkeypatch.setenv("HOME", str(tmp_path))
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(8, 8))
    n = anvil_loader.apply_to_model(
        model, model_id="not-a-real-model", batch=1, seq=4096, quant="mxfp4",
        gpu_gfx="gfx1100",
    )
    assert n == 0
