"""Post-training integration with Heretic for refusal-direction removal.

Pipeline slot: train -> (LoRA merge) -> HERETIC -> GGUF export / quantize.

Heretic runs under mud-puppy's OWN ROCm-torch environment. This module
spawns heretic through the ``scripts/heretic_auto.py`` driver which:

- stubs ``bitsandbytes`` in sys.modules so heretic's imports succeed
- disables PEFT's bnb dispatcher so LoRA injection uses the default path
  (our ``bnb_rocm.Linear4bit`` is isinstance-compatible with nn.Linear)
- monkey-patches heretic's interactive menus with scripted responders
- optionally routes ``--quantization BNB_4BIT`` through
  ``mud_puppy.bnb_rocm.quantize_model_4bit`` after base model load

The subprocess is a child of the current mud-puppy python, sharing the
same site-packages, torch-ROCm stack, and model caches. It's still a
subprocess (not an in-process import) so that:

- heretic's monkey-patches do not leak into mud-puppy's main process
- a heretic crash does not take down the training run's parent
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

DRIVER_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "heretic_auto.py"


@dataclass
class HereticConfig:
    """Knobs that the mud-puppy CLI exposes."""

    model_dir: str
    out_dir: str

    # Optuna trial count. 30 is heretic's default; more = better direction
    # estimate, diminishing returns past ~50.
    n_trials: int = 30

    # "NONE" (full precision) or "BNB_4BIT" (routed through bnb_rocm).
    # Required for 14B+ models on 24GB cards.
    quantization: str = "NONE"

    # "merge" gives a standalone HF checkpoint (what GGUF export wants).
    # "adapter" gives only the LoRA delta.
    merge_strategy: str = "merge"

    # Heretic's good/bad prompt pools (dataset ids)
    good_prompts_dataset: Optional[str] = None
    good_prompts_split: Optional[str] = None
    good_prompts_column: Optional[str] = None
    bad_prompts_dataset: Optional[str] = None
    bad_prompts_split: Optional[str] = None
    bad_prompts_column: Optional[str] = None

    # Chat system prompt during analysis
    system_prompt: Optional[str] = None

    # Free-form passthrough to heretic for anything not surfaced above
    extra_args: List[str] = field(default_factory=list)

    # Error policy
    fail_on_no_trials: bool = True

    # Override the python interpreter (default: current one, which is
    # mud-puppy's ROCm env)
    python_bin: str = sys.executable


class HereticError(RuntimeError):
    """Raised when the heretic subprocess errors out."""


def _build_heretic_argv(cfg: HereticConfig) -> List[str]:
    argv: List[str] = [
        "--model", cfg.model_dir,
        "--n-trials", str(cfg.n_trials),
        # We ALWAYS pass quantization=NONE to heretic's own CLI; if the
        # caller requested BNB_4BIT, our driver handles that via bnb_rocm
        # post-load. This keeps heretic's from_pretrained on the fp16/bf16
        # path which is what our driver's monkey-patch expects.
        "--quantization", "NONE",
    ]

    if cfg.good_prompts_dataset:
        argv += ["--good-prompts.dataset", cfg.good_prompts_dataset]
    if cfg.good_prompts_split:
        argv += ["--good-prompts.split", cfg.good_prompts_split]
    if cfg.good_prompts_column:
        argv += ["--good-prompts.column", cfg.good_prompts_column]

    if cfg.bad_prompts_dataset:
        argv += ["--bad-prompts.dataset", cfg.bad_prompts_dataset]
    if cfg.bad_prompts_split:
        argv += ["--bad-prompts.split", cfg.bad_prompts_split]
    if cfg.bad_prompts_column:
        argv += ["--bad-prompts.column", cfg.bad_prompts_column]

    if cfg.system_prompt:
        argv += ["--system-prompt", cfg.system_prompt]

    argv += list(cfg.extra_args)
    return argv


def run_heretic(cfg: HereticConfig) -> str:
    """Invoke heretic on an HF model directory, return the output path.

    Raises HereticError on subprocess failure or (if fail_on_no_trials)
    on an exit that produced no saved model.
    """
    if not os.path.isdir(cfg.model_dir):
        raise HereticError(f"Model dir does not exist: {cfg.model_dir}")
    if not os.path.isfile(os.path.join(cfg.model_dir, "config.json")):
        raise HereticError(
            f"No config.json at {cfg.model_dir}; heretic cannot load this"
        )

    if not DRIVER_SCRIPT.is_file():
        raise HereticError(f"heretic_auto.py driver missing at {DRIVER_SCRIPT}")

    os.makedirs(cfg.out_dir, exist_ok=True)

    heretic_argv = _build_heretic_argv(cfg)
    cmd = [
        cfg.python_bin, str(DRIVER_SCRIPT),
        "--save-dir", cfg.out_dir,
        "--merge-strategy", cfg.merge_strategy,
        "--quantization", cfg.quantization,
    ]
    if cfg.fail_on_no_trials:
        cmd.append("--fail-on-no-trials")
    cmd.append("--")
    cmd += heretic_argv

    log.info("[heretic_hook] running: %s", " ".join(cmd))
    print(f"[mud-puppy] Running heretic on {cfg.model_dir} -> {cfg.out_dir}")
    print(f"[mud-puppy] heretic trials={cfg.n_trials} quant={cfg.quantization}")

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise HereticError(
            f"heretic subprocess exited with code {proc.returncode}"
        )

    saved_config = os.path.join(cfg.out_dir, "config.json")
    if not os.path.isfile(saved_config):
        if cfg.fail_on_no_trials:
            raise HereticError(
                f"heretic exited 0 but no config.json at {cfg.out_dir}; "
                "run likely ended without selecting a trial. Check "
                "--heretic-n-trials and the good/bad prompt pools."
            )
        log.warning("heretic produced no output; returning original model_dir")
        return cfg.model_dir

    log.info("heretic produced abliterated model at %s", cfg.out_dir)
    return cfg.out_dir


def heretic_output_dir(base_output_dir: str) -> str:
    """Convention: heretic output lives under <training-run>/heretic/."""
    return os.path.join(base_output_dir, "heretic")


def is_heretic_available() -> bool:
    """Cheap probe for CLI warnings when --heretic is requested but the
    package is not installed in mud-puppy's env.
    """
    try:
        import heretic  # noqa: F401
        return True
    except ImportError:
        return False
