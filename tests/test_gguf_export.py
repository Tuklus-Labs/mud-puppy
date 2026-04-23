"""Tests for the GGUF export pipeline.

These tests mock out subprocess.run so we don't actually invoke llama.cpp
or kernel-anvil. They cover:

* tool discovery (llama.cpp root detection via $LLAMA_CPP_ROOT)
* source classification (HF model vs peft adapter)
* pipeline ordering (convert -> quantize -> anvil)
* kernel-anvil output parsing (extract config JSON path)
* error paths (missing llama.cpp, failed subprocess)

End-to-end validation (actually running convert_hf_to_gguf.py on a real
model) is manual; hitting the llama.cpp toolchain from pytest would make
the suite brittle and slow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from mud_puppy import gguf_export
from mud_puppy.gguf_export import (
    ExportConfig,
    ExportResult,
    GgufExportError,
    _find_kernel_anvil,
    _find_llama_cpp_root,
    _find_quantize_binary,
    _is_peft_adapter,
    _parse_anvil_config_path,
    export_to_gguf,
)


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


def test_find_llama_cpp_root_respects_env(tmp_path, monkeypatch) -> None:
    fake_root = tmp_path / "llama.cpp"
    fake_root.mkdir()
    (fake_root / "convert_hf_to_gguf.py").write_text("# stub\n")
    monkeypatch.setenv("LLAMA_CPP_ROOT", str(fake_root))
    assert _find_llama_cpp_root() == fake_root


def test_find_llama_cpp_root_raises_when_missing(tmp_path, monkeypatch) -> None:
    # Point env at a dir without convert_hf_to_gguf.py
    monkeypatch.setenv("LLAMA_CPP_ROOT", str(tmp_path))
    # Also override home so fallback paths don't accidentally match
    monkeypatch.setenv("HOME", str(tmp_path / "no-home"))
    with pytest.raises(GgufExportError, match="llama.cpp checkout not found"):
        _find_llama_cpp_root()


def test_find_quantize_binary_locates_built_binary(tmp_path) -> None:
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    bin_path = root / "build" / "bin" / "llama-quantize"
    bin_path.write_text("#!/bin/sh\n")
    bin_path.chmod(0o755)
    assert _find_quantize_binary(root) == bin_path


def test_find_quantize_binary_raises_when_not_built(tmp_path) -> None:
    root = tmp_path / "llama.cpp"
    root.mkdir()
    with pytest.raises(GgufExportError, match="llama-quantize not found"):
        _find_quantize_binary(root)


def test_find_kernel_anvil_returns_none_when_missing(monkeypatch, tmp_path) -> None:
    # Override PATH and HOME so neither fallback path resolves.
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "no-home"))
    assert _find_kernel_anvil() is None


# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------


def test_is_peft_adapter_detects_adapter_config(tmp_path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}")
    assert _is_peft_adapter(tmp_path) is True


def test_is_peft_adapter_false_for_plain_hf(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}")
    assert _is_peft_adapter(tmp_path) is False


# ---------------------------------------------------------------------------
# anvil stdout parsing
# ---------------------------------------------------------------------------


def test_parse_anvil_config_path_finds_cache_json() -> None:
    stdout = (
        "kernel-anvil: profiling 8 shapes...\n"
        "kernel-anvil: wrote config to /home/aegis/.cache/smithy/model.json\n"
        "kernel-anvil: done\n"
    )
    assert _parse_anvil_config_path(stdout) == "/home/aegis/.cache/smithy/model.json"


def test_parse_anvil_config_path_none_when_absent() -> None:
    stdout = "kernel-anvil: nothing to do\n"
    assert _parse_anvil_config_path(stdout) is None


# ---------------------------------------------------------------------------
# End-to-end with subprocess mocked
# ---------------------------------------------------------------------------


def _scaffold_llama_cpp(tmp_path: Path) -> Path:
    """Create a fake llama.cpp checkout + build outputs."""
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("# fake converter\n")
    bin_path = root / "build" / "bin" / "llama-quantize"
    bin_path.write_text("#!/bin/sh\n")
    bin_path.chmod(0o755)
    return root


def _scaffold_hf_model(tmp_path: Path) -> Path:
    """Create a minimal HF-shaped source dir (no adapter)."""
    src = tmp_path / "trained-model"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "model.safetensors").write_text("stub")
    return src


def test_export_pipeline_hf_model_no_quant(tmp_path, monkeypatch) -> None:
    root = _scaffold_llama_cpp(tmp_path)
    src = _scaffold_hf_model(tmp_path)
    monkeypatch.setenv("LLAMA_CPP_ROOT", str(root))

    called = []

    def fake_run(cmd, **kwargs):
        called.append(cmd)
        # Simulate the GGUF write so the next step can find it.
        if "convert_hf_to_gguf.py" in " ".join(cmd):
            out_idx = cmd.index("--outfile") + 1
            Path(cmd[out_idx]).touch()
        elif Path(cmd[0]).name == "llama-quantize":
            Path(cmd[2]).touch()
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.object(gguf_export.subprocess, "run", side_effect=fake_run):
        with mock.patch.object(gguf_export, "_find_kernel_anvil", return_value=None):
            cfg = ExportConfig(
                source_dir=str(src),
                out_path="model.gguf",
                quant="",  # fp16, no quantize step
                optimize_with_kernel_anvil=False,
            )
            result = export_to_gguf(cfg)

    # Exactly one subprocess call: the converter. No quantize, no anvil.
    assert len(called) == 1
    assert "convert_hf_to_gguf.py" in " ".join(called[0])
    assert result.gguf_path.endswith("model.gguf")
    assert any("no merge needed" in s for s in result.steps)


def test_export_pipeline_hf_model_with_quant(tmp_path, monkeypatch) -> None:
    root = _scaffold_llama_cpp(tmp_path)
    src = _scaffold_hf_model(tmp_path)
    monkeypatch.setenv("LLAMA_CPP_ROOT", str(root))

    called = []

    def fake_run(cmd, **kwargs):
        called.append(cmd)
        if "convert_hf_to_gguf.py" in " ".join(cmd):
            out_idx = cmd.index("--outfile") + 1
            Path(cmd[out_idx]).touch()
        elif Path(cmd[0]).name == "llama-quantize":
            Path(cmd[2]).touch()
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.object(gguf_export.subprocess, "run", side_effect=fake_run):
        with mock.patch.object(gguf_export, "_find_kernel_anvil", return_value=None):
            cfg = ExportConfig(
                source_dir=str(src),
                out_path="model.gguf",
                quant="Q4_K_M",
                optimize_with_kernel_anvil=False,
            )
            result = export_to_gguf(cfg)

    # Converter + quantize (anvil disabled). Two calls.
    assert len(called) == 2
    assert "convert_hf_to_gguf.py" in " ".join(called[0])
    assert "llama-quantize" in called[1][0]
    assert called[1][-1] == "Q4_K_M"
    assert result.serve_command is not None
    assert "llama-server" in result.serve_command
    assert str(result.gguf_path) in result.serve_command


def test_export_pipeline_calls_kernel_anvil_when_available(tmp_path, monkeypatch) -> None:
    root = _scaffold_llama_cpp(tmp_path)
    src = _scaffold_hf_model(tmp_path)
    monkeypatch.setenv("LLAMA_CPP_ROOT", str(root))

    anvil_stub = tmp_path / "kernel-anvil"
    anvil_stub.write_text("#!/bin/sh\n")
    anvil_stub.chmod(0o755)

    called = []

    def fake_run(cmd, **kwargs):
        called.append(cmd)
        if "convert_hf_to_gguf.py" in " ".join(cmd):
            out_idx = cmd.index("--outfile") + 1
            Path(cmd[out_idx]).touch()
        elif Path(cmd[0]).name == "llama-quantize":
            Path(cmd[2]).touch()
            return mock.Mock(returncode=0, stdout="", stderr="")
        elif Path(cmd[0]).name == "kernel-anvil":
            return mock.Mock(
                returncode=0,
                stdout="kernel-anvil: wrote /home/aegis/.cache/smithy/model.json\n",
                stderr="",
            )
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.object(gguf_export.subprocess, "run", side_effect=fake_run):
        with mock.patch.object(
            gguf_export, "_find_kernel_anvil", return_value=str(anvil_stub)
        ):
            cfg = ExportConfig(
                source_dir=str(src),
                out_path="model.gguf",
                quant="Q4_K_M",
                optimize_with_kernel_anvil=True,
            )
            result = export_to_gguf(cfg)

    # Three calls: convert + quantize + anvil
    assert len(called) == 3
    assert Path(called[2][0]).name == "kernel-anvil"
    assert called[2][1] == "gguf-optimize"
    # Config path parsed from anvil stdout
    assert result.kernel_anvil_config == "/home/aegis/.cache/smithy/model.json"
    assert "SMITHY_CONFIG=" in result.serve_command


def test_export_pipeline_fails_on_converter_error(tmp_path, monkeypatch) -> None:
    root = _scaffold_llama_cpp(tmp_path)
    src = _scaffold_hf_model(tmp_path)
    monkeypatch.setenv("LLAMA_CPP_ROOT", str(root))

    import subprocess
    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd,
            output="stdout", stderr="convert failed: bad config",
        )

    with mock.patch.object(gguf_export.subprocess, "run", side_effect=fake_run):
        cfg = ExportConfig(source_dir=str(src), quant="")
        with pytest.raises(GgufExportError, match="convert_hf_to_gguf.py failed"):
            export_to_gguf(cfg)


def test_export_pipeline_missing_source_dir(tmp_path) -> None:
    cfg = ExportConfig(source_dir=str(tmp_path / "does-not-exist"))
    with pytest.raises(GgufExportError, match="does not exist"):
        export_to_gguf(cfg)
