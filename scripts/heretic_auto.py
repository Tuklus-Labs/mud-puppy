#!/usr/bin/env python3
"""Non-interactive heretic driver for mud-puppy.

Runs the upstream p-e-w/heretic-llm algorithm on a model directory, but:
- Under mud-puppy's own ROCm-torch environment (no CUDA venv).
- With a ``bitsandbytes`` stub injected into sys.modules so heretic's
  imports succeed; quantization actually routes through
  ``mud_puppy.bnb_rocm``.
- With heretic's interactive menus monkey-patched so the trial selection,
  save path, and merge-strategy prompts answer automatically from our
  CLI flags.

Usage:
    python scripts/heretic_auto.py \\
        --save-dir /path/to/out \\
        --quantization NONE \\
        -- \\
        --model /path/to/model \\
        --n-trials 30

Requires heretic-llm installed into mud-puppy's env (pip install heretic-llm).
Must be invoked with mud-puppy's python, NOT heretic's old CUDA venv.
"""

from __future__ import annotations

import argparse
import os
import sys
import types


# ---------------------------------------------------------------------------
# bitsandbytes stub: make heretic's `import bitsandbytes as bnb` work
# without installing a CUDA-built bitsandbytes. heretic only reaches
# bnb.functional.dequantize_4bit inside a code path guarded by
# `quant_state is not None`, and our bnb_rocm.Linear4bit weights never
# set quant_state, so the stub's dequantize_4bit is a safety net.
# ---------------------------------------------------------------------------

def _install_bitsandbytes_stub() -> None:
    """Register a minimal bitsandbytes module in sys.modules.

    Must run BEFORE importing heretic.* because heretic.model does
    ``import bitsandbytes as bnb`` at module-load time.

    We also force PEFT's ``is_bnb_available`` to return False so PEFT's
    LoRA injector skips its bnb-specific dispatchers (which would try to
    import real bitsandbytes from the .bnb subpackage of PEFT). Our
    bnb_rocm.Linear4bit is isinstance-compatible with nn.Linear, so PEFT
    treats it through the default dispatcher, which is what we want.
    """
    if "bitsandbytes" in sys.modules:
        return  # real or prior stub already present; don't clobber

    import importlib.machinery

    bnb = types.ModuleType("bitsandbytes")
    functional = types.ModuleType("bitsandbytes.functional")

    def _dequantize_4bit_unreachable(*args, **kwargs):
        raise RuntimeError(
            "bitsandbytes.functional.dequantize_4bit was called from the "
            "heretic stub. This means heretic reached a bnb-specific code "
            "path that the mud_puppy.bnb_rocm shim does not handle. "
            "Expected: bnb_rocm.Linear4bit exposes a dequantized .weight "
            "property so heretic never enters this branch. Investigate."
        )

    functional.dequantize_4bit = _dequantize_4bit_unreachable
    bnb.functional = functional
    nn_mod = types.ModuleType("bitsandbytes.nn")
    nn_mod.Linear4bit = object  # sentinel; isinstance checks return False
    bnb.nn = nn_mod

    # Give every stub a real ModuleSpec so importlib.util.find_spec does
    # not crash on spec==None (PEFT's is_bnb_available calls find_spec).
    bnb.__spec__ = importlib.machinery.ModuleSpec(name="bitsandbytes", loader=None)
    functional.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.functional", loader=None
    )
    nn_mod.__spec__ = importlib.machinery.ModuleSpec(
        name="bitsandbytes.nn", loader=None
    )

    sys.modules["bitsandbytes"] = bnb
    sys.modules["bitsandbytes.functional"] = functional
    sys.modules["bitsandbytes.nn"] = nn_mod


def _disable_peft_bnb_dispatchers() -> None:
    """Force PEFT to treat bitsandbytes as unavailable so it skips its
    bnb-specific LoRA dispatchers.

    PEFT modules do ``from ..import_utils import is_bnb_available`` at their
    own load time, binding the original function into their own namespace.
    Patching ``peft.import_utils.is_bnb_available`` alone does NOT reach
    those already-bound references. We walk ``sys.modules`` and replace
    every occurrence in modules whose name starts with 'peft'.

    Called both before and after heretic imports PEFT to cover both cases.
    """
    def _always_false(*args, **kwargs):
        return False

    try:
        import peft.import_utils as _piu
    except ImportError:
        return

    for name in ("is_bnb_available", "is_bnb_4bit_available"):
        if hasattr(_piu, name):
            setattr(_piu, name, _always_false)

    # Now replace every 'is_bnb_*' attribute on every currently-loaded
    # peft submodule. Works because PEFT modules imported the function by
    # name; we re-bind their local name to our sentinel.
    for modname, mod in list(sys.modules.items()):
        if not modname.startswith("peft"):
            continue
        if mod is None:
            continue
        for name in ("is_bnb_available", "is_bnb_4bit_available"):
            if hasattr(mod, name):
                setattr(mod, name, _always_false)


# ---------------------------------------------------------------------------
# Our CLI args (everything before --); everything after -- is heretic's
# ---------------------------------------------------------------------------

def parse_driver_args() -> tuple[argparse.Namespace, list[str]]:
    if "--" in sys.argv[1:]:
        idx = sys.argv.index("--")
        driver_argv = sys.argv[1:idx]
        heretic_argv = sys.argv[idx + 1 :]
    else:
        driver_argv = sys.argv[1:]
        heretic_argv = []

    ap = argparse.ArgumentParser(description="Non-interactive heretic driver")
    ap.add_argument("--save-dir", required=True,
                    help="Absolute path to write the abliterated model")
    ap.add_argument("--merge-strategy", default="merge",
                    choices=["merge", "adapter"],
                    help='"merge" saves the merged model, "adapter" saves only '
                         "the LoRA delta. Default is merge.")
    ap.add_argument("--quantization", default="NONE",
                    choices=["NONE", "BNB_4BIT"],
                    help="If BNB_4BIT, route through mud_puppy.bnb_rocm. "
                         "Required for 14B+ models on 24GB cards.")
    ap.add_argument("--fail-on-no-trials", action="store_true")
    return ap.parse_args(driver_argv), heretic_argv


# ---------------------------------------------------------------------------
# Menu auto-responder
# ---------------------------------------------------------------------------

def install_scripted_prompts(save_dir: str, merge_strategy: str) -> None:
    """Replace heretic.main.prompt_{select,text,path} with scripted responders.

    Stateful: after the FIRST successful save (detected by the path prompt
    firing and then the action menu re-entering), subsequent prompts steer
    heretic cleanly out of its menu loops so the process exits once we have
    an artifact on disk.
    """
    state: dict[str, bool] = {"saved_once": False}

    def _log(msg: str) -> None:
        print(f"[heretic_auto] {msg}", flush=True)

    def scripted_select(message, choices):
        msg = str(message).lower()

        def _val(c):
            return c.value if hasattr(c, "value") else c

        def _title(c):
            return c.title if hasattr(c, "title") else str(c)

        # Two similar prompts share "proceed":
        #   - "How do you want to proceed?"   (merge-strategy: merge/cancel)
        #   - "How would you like to proceed?" (resume-study: continue/restart/"")
        # Disambiguate by inspecting choice values.
        if "proceed" in msg:
            values = [str(_val(c)).lower() for c in choices]
            if any(v in ("continue", "restart") for v in values):
                # Resume-study prompt: always restart so we get a clean trial set.
                for c in choices:
                    if str(_val(c)).lower() == "restart":
                        _log("resume-study prompt -> 'restart'")
                        return _val(c)
                # Fallback: if no restart choice, exit
                _log("resume-study prompt -> '' (no restart option)")
                return ""
            # Otherwise this is the merge-strategy prompt.
            for c in choices:
                v = _val(c)
                if isinstance(v, str) and v.lower() == "merge":
                    _log("merge-strategy prompt -> 'merge'")
                    return v
            return None

        # Trial selection: "Which trial do you want to use?"
        # After we've saved once, exit by returning "" so heretic's outer
        # loop hits `return`.
        if "which trial" in msg or ("trial" in msg and "use" in msg):
            if state["saved_once"]:
                _log("trial prompt (post-save) -> '' to exit")
                return ""
            for c in choices:
                v = _val(c)
                if v not in ("continue", "", None) and not isinstance(v, str):
                    _log(f"trial prompt -> {_title(c)}")
                    return v
            _log("no usable trial; returning '' to exit")
            return ""

        # Post-trial action menu.
        # First pass: pick Save. Second pass (after save completes):
        # pick "Return to the trial selection menu" so heretic goes back
        # to the outer loop, where we then return "" on the next trial prompt.
        if "decensored" in msg or "what do you want to do" in msg:
            if state["saved_once"]:
                for c in choices:
                    v = _val(c)
                    if isinstance(v, str) and "return" in v.lower():
                        _log(f"action prompt (post-save) -> {v}")
                        return v
                _log("action prompt (post-save) -> None to break")
                return None
            for c in choices:
                v = _val(c)
                if isinstance(v, str) and "save" in v.lower() and "folder" in v.lower():
                    _log(f"action prompt -> {v}")
                    return v
            return None

        _log(f"UNHANDLED prompt_select: {message!r}; returning None")
        return None

    def scripted_text(message, **kwargs):
        msg = str(message).lower()
        if "additional trial" in msg or "how many" in msg:
            return "0"
        return ""

    def scripted_path(message):
        print(f"[heretic_auto] path prompt -> {save_dir}", flush=True)
        state["saved_once"] = True
        return save_dir

    import heretic.main as _hm
    _hm.prompt_select = scripted_select
    _hm.prompt_text = scripted_text
    _hm.prompt_path = scripted_path
    import heretic.utils as _hu
    _hu.prompt_select = scripted_select
    _hu.prompt_text = scripted_text
    _hu.prompt_path = scripted_path


# ---------------------------------------------------------------------------
# Optional: route heretic's BNB_4BIT load through bnb_rocm after load
# ---------------------------------------------------------------------------

def install_bnb_rocm_hook(quantization_kind: str) -> None:
    """Monkey-patch heretic.model.Model._get_quantization_config to return
    None (so heretic's from_pretrained does NOT attempt a BNB 4-bit load via
    transformers' bitsandbytes integration), and wrap _load_model so that
    the loaded bf16 model is 4-bit-quantized in place via mud_puppy.bnb_rocm.

    When --quantization NONE, this is a no-op.
    """
    if quantization_kind != "BNB_4BIT":
        return

    import heretic.model as _mdl
    from mud_puppy.bnb_rocm import quantize_model_4bit
    import torch

    original_init = _mdl.Model.__init__

    def _patched_get_quantization_config(self, dtype):
        # Never return a BnB config; we quantize post-load.
        return None

    _mdl.Model._get_quantization_config = _patched_get_quantization_config

    def _patched_init(self, *args, **kwargs):
        print("[heretic_auto] loading base model (bf16) for bnb_rocm quant", flush=True)
        original_init(self, *args, **kwargs)
        print("[heretic_auto] running bnb_rocm.quantize_model_4bit in place", flush=True)
        inner = getattr(self, "model", None) or getattr(self, "_model", None) or self
        if hasattr(inner, "model"):
            inner = inner.model  # unwrap one more level
        dtype = torch.bfloat16
        quantize_model_4bit(inner, dtype=dtype)
        print("[heretic_auto] bnb_rocm quant done", flush=True)

    _mdl.Model.__init__ = _patched_init


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    driver_args, heretic_argv = parse_driver_args()

    save_dir = os.path.abspath(driver_args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[heretic_auto] save_dir={save_dir}", flush=True)
    print(f"[heretic_auto] merge_strategy={driver_args.merge_strategy}", flush=True)
    print(f"[heretic_auto] quantization={driver_args.quantization}", flush=True)
    print(f"[heretic_auto] heretic args: {heretic_argv}", flush=True)

    # 1) stub bitsandbytes FIRST, before any heretic import
    _install_bitsandbytes_stub()
    _disable_peft_bnb_dispatchers()

    # 2) install the prompt auto-responder (patches heretic.main names)
    install_scripted_prompts(save_dir, driver_args.merge_strategy)

    # 3) hook bnb_rocm if BNB quantization was requested
    install_bnb_rocm_hook(driver_args.quantization)

    # 4) rewrite argv so heretic's argparse sees only its own args
    sys.argv = ["heretic"] + heretic_argv

    # 5) drive. Re-run the PEFT-bnb disable in case heretic's imports
    # triggered loading of additional peft submodules that re-bound
    # is_bnb_available from the unpatched source.
    from heretic.main import main as heretic_main
    _disable_peft_bnb_dispatchers()
    heretic_main()

    # 6) verify output landed
    expected = os.path.join(save_dir, "config.json")
    if not os.path.exists(expected):
        print(
            f"[heretic_auto] WARNING: no config.json at {save_dir}; "
            "heretic likely exited without selecting a trial",
            flush=True,
        )
        if driver_args.fail_on_no_trials:
            return 2

    print(f"[heretic_auto] done. output at {save_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
