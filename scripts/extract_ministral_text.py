"""
Extract the text-only causal-LM backbone from Ministral-3-14B-Reasoning
(Mistral3ForConditionalGeneration, a VLM) and save as a standalone
CausalLM checkpoint for text fine-tuning.

Result goes to /home/aegis/Models/Ministral-3-14B-Text/ (HF format,
loadable by AutoModelForCausalLM).

This is a lossless extraction of the language_model submodule. The vision
tower and multimodal projector are dropped. The resulting model can be
fine-tuned as a pure text LLM and later merged back with the VLM if
vision is ever needed again.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
)


SRC = Path("/home/aegis/Models/Ministral-3-14B-Reasoning")
DST = Path("/home/aegis/Models/Ministral-3-14B-Text")


def main():
    DST.mkdir(parents=True, exist_ok=True)

    print(f"Loading VLM config from {SRC}")
    vlm_cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=True)
    text_cfg = vlm_cfg.text_config
    text_cfg.architectures = ["MistralForCausalLM"]
    print(f"text layers={text_cfg.num_hidden_layers}, hidden={text_cfg.hidden_size}, vocab={text_cfg.vocab_size}")

    print("Loading full VLM into CPU memory (bf16)")
    vlm = Mistral3ForConditionalGeneration.from_pretrained(
        SRC,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # The causal-LM backbone is vlm.language_model. It is already a
    # MistralForCausalLM instance; we just need to save it cleanly.
    lm = getattr(vlm, "language_model", None)
    if lm is None:
        for name in ("model", "text_model"):
            lm = getattr(vlm, name, None)
            if lm is not None:
                break
    if lm is None:
        print("Could not locate text backbone; aborting.")
        sys.exit(2)
    print(f"Backbone type: {type(lm).__name__}")

    print(f"Saving text-only model to {DST}")
    lm.save_pretrained(DST, safe_serialization=True, max_shard_size="5GB")

    # Copy tokenizer assets verbatim.
    print("Copying tokenizer files")
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tekken.json",
    ):
        src = SRC / name
        if src.exists():
            shutil.copy2(src, DST / name)
    # Re-serialize the config as a text-only config so AutoModelForCausalLM
    # picks it up without trust_remote_code.
    text_cfg_dict = text_cfg.to_dict()
    text_cfg_dict["architectures"] = ["MistralForCausalLM"]
    text_cfg_dict["model_type"] = text_cfg_dict.get("model_type", "mistral")
    with (DST / "config.json").open("w") as f:
        json.dump(text_cfg_dict, f, indent=2)

    # Copy generation config if present and strip vision-specific hooks.
    gc = SRC / "generation_config.json"
    if gc.exists():
        shutil.copy2(gc, DST / "generation_config.json")

    # Verify round-trip load.
    print("Round-trip verify: AutoModelForCausalLM on new dir")
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        DST,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    nparams = sum(p.numel() for p in m.parameters())
    print(f"Loaded OK. Params: {nparams/1e9:.2f} B")

    print("Done.")


if __name__ == "__main__":
    main()
