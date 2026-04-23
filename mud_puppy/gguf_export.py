"""Export a trained model to GGUF and optionally optimize with kernel-anvil.

Pipeline: mud-puppy saves training output as HuggingFace safetensors.
To serve on llama.cpp's backend of choice (ROCm, Vulkan, Metal, CUDA)
the user typically wants a quantized GGUF. This module does:

    1. Optional LoRA merge (peft model -> full-precision safetensors)
    2. HF -> GGUF conversion via llama.cpp's convert_hf_to_gguf.py
    3. Optional quantization via llama.cpp's llama-quantize
    4. Optional kernel-anvil shape-specific tuning for the served
       model's GEMV shapes

Discovery: llama.cpp paths are resolved in order:
    $LLAMA_CPP_ROOT
    ~/Projects/llama.cpp
    ~/Projects/llama-cpp-mainline
    ~/Projects/llama-cpp-turboquant
    /usr/local/src/llama.cpp

The converter (`convert_hf_to_gguf.py`) lives at repo root; the
quantizer (`llama-quantize`) is in `build/bin/` after a cmake build.

kernel-anvil's gguf-optimize is discovered via `$PATH` -> `~/bin` ->
`~/Projects/kernel-anvil/.venv/bin`. Emitted command returns the final
config path so a caller (studio Library pane, CLI output) can print a
ready-to-use `llama-server` invocation.

All helpers run via subprocess. This module is pure orchestration; it
doesn't import llama.cpp or kernel-anvil Python symbols so we don't
pull their dependencies into the mud-puppy install for users who just
want to train.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExportConfig:
    """How to export a trained mud-puppy checkpoint to GGUF."""

    # Input: a mud-puppy output directory. Contains either a merged HF
    # safetensors model or a peft adapter (we merge if it's the latter).
    source_dir: str

    # Output GGUF destination. If relative, it's written inside source_dir.
    out_path: str = "model.gguf"

    # GGUF quantization. Empty string = fp16 GGUF (no quantization step).
    # Any llama-quantize type name works: Q4_K_M, Q5_K_M, Q8_0, etc.
    quant: str = "Q4_K_M"

    # If the source has a peft adapter, we need the base model path to
    # merge against. Auto-detected from `adapter_config.json` if present.
    base_model: Optional[str] = None

    # Run kernel-anvil gguf-optimize on the produced GGUF. Costs <1s;
    # produces ~2x decode speedup for AMD inference. Safe default.
    optimize_with_kernel_anvil: bool = True

    # Keep the intermediate fp16 GGUF on disk (before quantization). Useful
    # for later re-quantization at a different level without reconverting.
    keep_fp16_intermediate: bool = False


@dataclass
class ExportResult:
    """What happened during an export."""

    gguf_path: str
    steps: List[str] = field(default_factory=list)
    kernel_anvil_config: Optional[str] = None  # path to smithy JSON
    serve_command: Optional[str] = None  # ready-to-paste llama-server invocation


class GgufExportError(RuntimeError):
    """Export pipeline failed at some step; message names the step."""


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


def _find_llama_cpp_root() -> Path:
    """Locate a llama.cpp checkout containing converter + quantizer."""
    env = os.environ.get("LLAMA_CPP_ROOT")
    candidates: List[Path] = []
    if env:
        candidates.append(Path(env))
    home = Path.home()
    candidates += [
        home / "Projects" / "llama.cpp",
        home / "Projects" / "llama-cpp-mainline",
        home / "Projects" / "llama-cpp-turboquant",
        Path("/usr/local/src/llama.cpp"),
    ]
    for c in candidates:
        if (c / "convert_hf_to_gguf.py").is_file():
            return c
    raise GgufExportError(
        "llama.cpp checkout not found. Set $LLAMA_CPP_ROOT or place a "
        "checkout at ~/Projects/llama.cpp. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _find_quantize_binary(llama_cpp_root: Path) -> Path:
    """Locate the llama-quantize binary from a llama.cpp build."""
    candidates = [
        llama_cpp_root / "build" / "bin" / "llama-quantize",
        llama_cpp_root / "build-hip" / "bin" / "llama-quantize",
        llama_cpp_root / "build-rocm" / "bin" / "llama-quantize",
        llama_cpp_root / "llama-quantize",  # in-tree legacy
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    raise GgufExportError(
        f"llama-quantize not found under {llama_cpp_root}. Build llama.cpp "
        f"first: `cd {llama_cpp_root} && cmake -B build && cmake --build "
        f"build --target llama-quantize -j`"
    )


def _find_kernel_anvil() -> Optional[str]:
    """Locate the kernel-anvil CLI. Returns None if missing (optional)."""
    which = shutil.which("kernel-anvil")
    if which:
        return which
    home = Path.home()
    for c in [
        home / "bin" / "kernel-anvil",
        home / "Projects" / "kernel-anvil" / ".venv" / "bin" / "kernel-anvil",
    ]:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


# ---------------------------------------------------------------------------
# Merge LoRA adapter (if the source is an adapter directory)
# ---------------------------------------------------------------------------


def _is_peft_adapter(path: Path) -> bool:
    return (path / "adapter_config.json").is_file()


def _merge_peft_adapter(adapter_dir: Path, base_model: Optional[str]) -> Path:
    """Merge a peft adapter into its base model, return the merged directory.

    Returns a path to a new temp directory holding a standard HF
    safetensors model. Caller is responsible for cleanup if desired.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Auto-detect base_model from adapter_config.json if not given.
    if base_model is None:
        cfg_path = adapter_dir / "adapter_config.json"
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception as exc:
            raise GgufExportError(
                f"cannot read {cfg_path}: {exc}"
            ) from exc
        base_model = cfg.get("base_model_name_or_path")
        if not base_model:
            raise GgufExportError(
                "adapter_config.json has no base_model_name_or_path; "
                "pass base_model explicitly"
            )

    log.info("merging peft adapter from %s onto base %s", adapter_dir, base_model)

    # Load base + adapter, merge, save.
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="bfloat16", low_cpu_mem_usage=True
    )
    merged_model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged_model = merged_model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    out_dir = Path(tempfile.mkdtemp(prefix="mud-puppy-merged-"))
    merged_model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    log.info("merged model saved to %s", out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------


def export_to_gguf(config: ExportConfig) -> ExportResult:
    """Run the full export pipeline. Returns an ExportResult."""
    src = Path(config.source_dir).resolve()
    if not src.is_dir():
        raise GgufExportError(f"source_dir {src} does not exist")

    out_path = Path(config.out_path)
    if not out_path.is_absolute():
        out_path = (src / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = ExportResult(gguf_path=str(out_path))

    llama_cpp_root = _find_llama_cpp_root()
    converter = llama_cpp_root / "convert_hf_to_gguf.py"
    log.info("using llama.cpp at %s", llama_cpp_root)

    # --- Step 1: resolve / merge source ------------------------------------
    hf_dir = src
    tmp_merged_dir: Optional[Path] = None
    if _is_peft_adapter(src):
        tmp_merged_dir = _merge_peft_adapter(src, config.base_model)
        hf_dir = tmp_merged_dir
        result.steps.append(f"merged peft adapter -> {hf_dir}")
    else:
        result.steps.append(f"source is HF model (no merge needed)")

    # --- Step 2: HF -> GGUF ------------------------------------------------
    # Intermediate fp16 GGUF; we quantize it after if config.quant is set.
    if config.quant:
        fp16_gguf = out_path.with_suffix(".fp16.gguf")
    else:
        fp16_gguf = out_path

    log.info("converting HF -> GGUF (fp16): %s -> %s", hf_dir, fp16_gguf)
    convert_cmd = [
        "python3", str(converter),
        str(hf_dir),
        "--outfile", str(fp16_gguf),
        "--outtype", "f16",
    ]
    try:
        subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise GgufExportError(
            f"convert_hf_to_gguf.py failed (exit {exc.returncode}):\n"
            f"stdout: {exc.stdout[-600:]}\nstderr: {exc.stderr[-600:]}"
        ) from exc
    result.steps.append(f"converted to fp16 GGUF: {fp16_gguf}")

    # --- Step 3: quantize --------------------------------------------------
    if config.quant:
        quantizer = _find_quantize_binary(llama_cpp_root)
        log.info("quantizing %s to %s: %s", fp16_gguf, config.quant, out_path)
        quant_cmd = [
            str(quantizer), str(fp16_gguf), str(out_path), config.quant,
        ]
        try:
            subprocess.run(quant_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise GgufExportError(
                f"llama-quantize failed (exit {exc.returncode}):\n"
                f"stdout: {exc.stdout[-600:]}\nstderr: {exc.stderr[-600:]}"
            ) from exc
        result.steps.append(f"quantized to {config.quant}: {out_path}")

        if not config.keep_fp16_intermediate:
            try:
                fp16_gguf.unlink()
                result.steps.append(f"removed intermediate {fp16_gguf}")
            except OSError:
                pass  # best-effort

    # --- Step 4: kernel-anvil optimize ------------------------------------
    if config.optimize_with_kernel_anvil:
        anvil = _find_kernel_anvil()
        if anvil is None:
            result.steps.append(
                "kernel-anvil not found on PATH, skipping shape tuning"
            )
        else:
            log.info("running kernel-anvil gguf-optimize on %s", out_path)
            anvil_cmd = [anvil, "gguf-optimize", str(out_path)]
            try:
                proc = subprocess.run(
                    anvil_cmd, check=True, capture_output=True, text=True,
                )
                # kernel-anvil writes to ~/.cache/smithy/<model>.json by default.
                # Parse the stdout to extract the config path it wrote.
                cfg_path = _parse_anvil_config_path(proc.stdout)
                result.kernel_anvil_config = cfg_path
                result.steps.append(
                    f"kernel-anvil tuned shapes -> {cfg_path or '(path not parsed)'}"
                )
            except subprocess.CalledProcessError as exc:
                result.steps.append(
                    f"kernel-anvil failed (exit {exc.returncode}), "
                    f"continuing anyway: {exc.stderr[-200:]}"
                )

    # --- Step 5: emit the serve command ------------------------------------
    serve_parts = []
    if result.kernel_anvil_config:
        serve_parts.append(f'SMITHY_CONFIG={result.kernel_anvil_config}')
    serve_parts.append("llama-server")
    serve_parts.append(f'-m {out_path}')
    serve_parts.append("-ngl 999")
    result.serve_command = " \\\n    ".join(serve_parts)

    # --- Cleanup -----------------------------------------------------------
    if tmp_merged_dir is not None and tmp_merged_dir.exists():
        shutil.rmtree(tmp_merged_dir, ignore_errors=True)
        result.steps.append(f"cleaned up merged temp dir")

    return result


def _parse_anvil_config_path(stdout: str) -> Optional[str]:
    """Find ~/.cache/smithy/... in kernel-anvil's stdout."""
    for line in stdout.splitlines():
        if ".cache/smithy" in line and ".json" in line:
            # Greedy extract: find the last whitespace-free token ending .json
            for tok in line.split():
                if tok.endswith(".json") and ".cache/smithy" in tok:
                    return tok
    return None
