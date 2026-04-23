"""Load Mistral's native (``consolidated.safetensors``) checkpoints.

Mistral ships text-only and VL checkpoints from its own training stack
with a layout that predates HuggingFace's naming: ``layers.N.attention.wq``
rather than ``model.layers.N.self_attn.q_proj``, ``feed_forward.w1/w2/w3``
rather than ``mlp.{gate,down,up}_proj``, and a single
``consolidated.safetensors`` blob instead of sharded ``model-*.safetensors``
with an index file.

HuggingFace's ``from_pretrained`` refuses these checkpoints outright
because it looks for canonical filenames + canonical tensor names.
This module detects the native format, constructs an HF model instance
from the config, then remaps and loads the tensors directly.

Scope: text-only extraction. If the checkpoint contains a vision tower
(Mistral3/Pixtral), those tensors are skipped. The returned module is
ready for LoRA/QLoRA fine-tuning on the language backbone.

Supported families tested: Mistral3 (Ministral-3), Mistral-7B-v0.1
native release, Pixtral. Future Mistral releases that keep the
``layers.N.attention.w{q,k,v,o}`` shape will work unchanged.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def is_mistral_native_checkpoint(path: str) -> bool:
    """True if ``path`` looks like Mistral's native checkpoint format.

    The tell is ``consolidated.safetensors`` with no
    ``model.safetensors.index.json`` or ``model.safetensors`` sibling.
    A single ``consolidated.safetensors`` with ``model-*.safetensors``
    alongside is NOT native (it's an HF-sharded variant named that way
    by accident).
    """
    if not os.path.isdir(path):
        return False
    has_consolidated = os.path.isfile(os.path.join(path, "consolidated.safetensors"))
    has_hf_single = os.path.isfile(os.path.join(path, "model.safetensors"))
    has_hf_index = os.path.isfile(os.path.join(path, "model.safetensors.index.json"))
    return has_consolidated and not has_hf_single and not has_hf_index


# ---------------------------------------------------------------------------
# Tensor-name remapping
# ---------------------------------------------------------------------------


# Static (non-layer) tensor remaps. Any key in the native checkpoint that
# matches a left side is renamed to the right side. Skipped keys (prefixes
# that don't belong to the language backbone) are handled separately.
# HF's MistralForCausalLM wraps its backbone under a ``model.`` prefix;
# only ``lm_head`` lives at the root.
_STATIC_RENAMES: Dict[str, str] = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight":           "model.norm.weight",
    "output.weight":         "lm_head.weight",
}

# Per-layer remaps. ``{L}`` is substituted with the layer index.
_LAYER_RENAMES: List[Tuple[str, str]] = [
    ("layers.{L}.attention.wq.weight",     "model.layers.{L}.self_attn.q_proj.weight"),
    ("layers.{L}.attention.wk.weight",     "model.layers.{L}.self_attn.k_proj.weight"),
    ("layers.{L}.attention.wv.weight",     "model.layers.{L}.self_attn.v_proj.weight"),
    ("layers.{L}.attention.wo.weight",     "model.layers.{L}.self_attn.o_proj.weight"),
    ("layers.{L}.feed_forward.w1.weight",  "model.layers.{L}.mlp.gate_proj.weight"),
    ("layers.{L}.feed_forward.w3.weight",  "model.layers.{L}.mlp.up_proj.weight"),
    ("layers.{L}.feed_forward.w2.weight",  "model.layers.{L}.mlp.down_proj.weight"),
    ("layers.{L}.attention_norm.weight",   "model.layers.{L}.input_layernorm.weight"),
    ("layers.{L}.ffn_norm.weight",         "model.layers.{L}.post_attention_layernorm.weight"),
]

# Prefixes to skip entirely when building a text-only model. Mistral3 /
# Pixtral include a vision tower and a projector; those have their own
# sub-modules in the HF model that we simply don't populate for
# text-only fine-tuning.
_SKIP_PREFIXES: Tuple[str, ...] = (
    "vision_encoder.",
    "vision_language_adapter.",
    "patch_merger.",
    "pre_mm_projector_norm.",
)


def _remap_key(native_key: str) -> Optional[str]:
    """Native tensor name -> HF tensor name, or None if this key should be skipped."""
    if any(native_key.startswith(p) for p in _SKIP_PREFIXES):
        return None

    if native_key in _STATIC_RENAMES:
        return _STATIC_RENAMES[native_key]

    # Layer tensors: extract the layer number via regex and re-template.
    m = re.match(r"^layers\.(\d+)\.(.*)$", native_key)
    if m:
        layer_idx, tail = m.group(1), m.group(2)
        template = f"layers.{{L}}.{tail}"
        for tmpl_native, tmpl_hf in _LAYER_RENAMES:
            if tmpl_native == template:
                return tmpl_hf.replace("{L}", layer_idx)
        # Unknown per-layer tensor -- skip noisily.
        log.warning(
            "mistral_native: unknown per-layer tensor '%s', skipping", native_key
        )
        return None

    log.warning("mistral_native: unknown tensor '%s', skipping", native_key)
    return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_mistral_native_as_causal_lm(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load a Mistral-native checkpoint into an HF causal-LM module.

    For a pure text Mistral checkpoint (no vision_encoder) this returns
    a fully-populated ``MistralForCausalLM``.

    For a Mistral3/Pixtral checkpoint we instantiate the language
    submodule only and wrap it so the downstream trainer sees a plain
    causal-LM. The vision tower is NOT loaded -- text-only LoRA/QLoRA
    leaves it frozen by definition, and the memory saving from skipping
    the vision weights is significant (a few GB on Pixtral/Mistral3).

    Raises ``FileNotFoundError`` if the checkpoint shape doesn't match.
    """
    if not is_mistral_native_checkpoint(path):
        raise FileNotFoundError(
            f"not a Mistral-native checkpoint: {path} "
            f"(need consolidated.safetensors with no model.safetensors sibling)"
        )

    from safetensors import safe_open
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(path)

    # Mistral3 and the like carry a ``text_config`` that defines the
    # language backbone. For pure-text Mistral the outer config IS the
    # text config.
    text_cfg = getattr(cfg, "text_config", None) or cfg

    # Build the HF language-backbone model from the text config. We use
    # MistralForCausalLM which accepts Ministral3Config too (same
    # architecture family; just a newer rope scaling).
    model = _instantiate_language_model(text_cfg, dtype=dtype)

    # Open the native checkpoint and build a remapped state dict.
    native_state = {}
    ckpt_path = os.path.join(path, "consolidated.safetensors")
    with safe_open(ckpt_path, framework="pt") as f:
        native_keys = list(f.keys())
        remapped_count = 0
        skipped_count = 0
        for nk in native_keys:
            hf_key = _remap_key(nk)
            if hf_key is None:
                skipped_count += 1
                continue
            native_state[hf_key] = f.get_tensor(nk).to(dtype)
            remapped_count += 1

    log.info(
        "mistral_native: remapped %d tensors, skipped %d (vision/unknown)",
        remapped_count, skipped_count,
    )

    # Load with strict=False because the HF model has extra buffers
    # (rotary embedding caches, attention bias masks) that the native
    # checkpoint doesn't carry. Those are initialized by
    # _instantiate_language_model and don't need loading.
    missing, unexpected = model.load_state_dict(native_state, strict=False)
    if unexpected:
        log.warning(
            "mistral_native: %d unexpected keys (dropped): %s",
            len(unexpected), unexpected[:5],
        )
    # Only "real" missing keys are a correctness problem. Rotary / bias
    # buffers showing up as missing is normal.
    real_missing = [
        k for k in missing
        if not any(token in k for token in (
            "rotary_emb", "inv_freq", "bias_mask", "attn_mask"
        ))
    ]
    if real_missing:
        log.warning(
            "mistral_native: %d real missing keys after load: %s",
            len(real_missing), real_missing[:5],
        )

    model.tie_weights()
    model.eval()
    return model


def _instantiate_language_model(text_cfg, dtype: torch.dtype) -> nn.Module:
    """Build an empty HF ``MistralForCausalLM`` from a text-config object.

    Handles the subtle difference between a pure-text ``MistralConfig``
    and the ``text_config`` subsection of a ``Mistral3Config``: both have
    the same fields, but the latter's ``model_type`` is ``"ministral3"``
    which transformers' AutoModelForCausalLM registry doesn't recognize.
    We cast via ``MistralConfig`` to sidestep that.
    """
    from transformers import MistralConfig, MistralForCausalLM

    # Copy the field values into a MistralConfig the HF registry accepts.
    # This is safe because Ministral3 is architecturally MistralForCausalLM
    # plus rope-scaling kwargs; the newer rope parameters are consumed by
    # MistralConfig transparently if present.
    fields = {}
    for attr in (
        "vocab_size", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "hidden_act",
        "max_position_embeddings", "initializer_range",
        "rms_norm_eps", "use_cache", "pad_token_id",
        "bos_token_id", "eos_token_id", "tie_word_embeddings",
        "rope_theta", "sliding_window", "attention_dropout",
        "head_dim",
    ):
        if hasattr(text_cfg, attr):
            val = getattr(text_cfg, attr)
            if val is not None:
                fields[attr] = val

    mistral_cfg = MistralConfig(**fields)
    # Rope scaling fields that Ministral3 carries can be attached
    # post-hoc; MistralForCausalLM's attention will honor them if the
    # attribute exists.
    rope_params = getattr(text_cfg, "rope_parameters", None)
    if rope_params is not None:
        mistral_cfg.rope_parameters = rope_params

    model = MistralForCausalLM(mistral_cfg)
    model = model.to(dtype)
    return model
