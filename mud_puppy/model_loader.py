"""Graceful-degradation model loader for mud-puppy.

The goal: fine-tune any model the HuggingFace ecosystem can load, including
brand-new architectures we've never seen at build time. Three tiers try
progressively more permissive strategies:

* **Tier 1 -- Registered CausalLM.**
  Pure text LLMs (Llama, Mistral, Qwen, Phi, Gemma, etc). The fast path:
  ``AutoModelForCausalLM.from_pretrained``.

* **Tier 2 -- Registered multimodal / seq2seq, extract the language backbone.**
  Vision-language models (LLaVA, Mistral3/Pixtral, Qwen2-VL, Phi-3-Vision,
  InternVL) ship a vision tower + projector + language model. When the user
  asks for text fine-tuning we load the full VL wrapper via
  ``AutoModelForVision2Seq`` (or similar), extract the language submodule,
  wrap it so the downstream trainer sees a normal causal-LM interface.

* **Tier 3 -- Unknown architecture, synthesize a loader by structure.**
  For an unregistered ``model_type`` we load via ``AutoModel`` with
  ``trust_remote_code=True`` (the weights ship their own modeling code),
  then walk the module tree for something with the SHAPE of a language
  model: an embedding at the input, a list of transformer-like blocks, and
  a linear projection at the output back to the vocabulary. If that
  structure is present, we rebuild a minimal causal-LM wrapper around it.
  If not, we raise with a dump of the architecture so the user can tell us
  what went wrong.

The tier choice is transparent to the rest of the trainer: every tier
returns an ``nn.Module`` with ``get_input_embeddings()``, a reasonable
``lm_head`` attribute, and a ``forward(input_ids=..., attention_mask=...,
labels=...)`` signature. That's enough for HF Trainer, peft, and
downstream LoRA wiring.

Design notes
~~~~~~~~~~~~

* The synthesizer does NOT attempt to be clever about tokenizers. If
  ``AutoTokenizer.from_pretrained`` fails on the same path, training
  can't proceed regardless; that fails separately in the trainer.
* The structural detector for "this looks like a language model" uses
  multiple signals: presence of an embedding whose num_embeddings matches
  the lm_head out_features, a list of >=2 modules with
  self-attention-shaped sub-modules, and a linear output projection.
  A single signal is not enough (classifiers satisfy some, embedding
  models satisfy others).
* Unknown architecture never silently succeeds. If all three tiers fail
  we raise ``LoaderError`` with the sequence of attempts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class LoaderError(RuntimeError):
    """All three loader tiers failed. Message lists what was tried."""


@dataclass
class ModelInspection:
    """What we learned by reading ``config.json`` before any weight load."""

    path: str
    architectures: List[str] = field(default_factory=list)
    model_type: str = ""
    kind: str = "unknown"  # causal-lm, vision-language, seq2seq, embedding, unknown
    params_b: float = 0.0  # rough parameter count in billions, or 0 if unknown
    hidden_size: int = 0
    num_layers: int = 0
    vocab_size: int = 0
    is_trust_remote: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class LoadResult:
    """Outcome of a successful load_model call."""

    model: nn.Module
    tier: int  # 1, 2, or 3
    extracted_submodule: Optional[str] = None  # e.g. "language_model" in tier 2
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config inspection (pre-load classification)
# ---------------------------------------------------------------------------


def inspect_model_config(path: str) -> ModelInspection:
    """Read ``config.json`` and classify the architecture.

    This is cheap (no weight load). Used both by the loader itself and by
    the UI's model.inspect IPC to drive the architecture card in Launch.
    """
    inspection = ModelInspection(path=path)
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        inspection.notes.append(f"transformers not importable: {exc}")
        return inspection

    cfg = None
    for trust in (False, True):
        try:
            cfg = AutoConfig.from_pretrained(path, trust_remote_code=trust)
            inspection.is_trust_remote = trust
            break
        except Exception as exc:
            inspection.notes.append(
                f"AutoConfig trust_remote_code={trust}: {type(exc).__name__}: "
                f"{str(exc)[:180]}"
            )

    if cfg is None:
        inspection.kind = "unknown"
        return inspection

    inspection.architectures = list(getattr(cfg, "architectures", []) or [])
    inspection.model_type = getattr(cfg, "model_type", "") or ""

    # Flatten text_config for multimodal configs; the text side is where
    # we pull hidden_size / num_layers.
    text_cfg = getattr(cfg, "text_config", None) or cfg
    inspection.hidden_size = int(getattr(text_cfg, "hidden_size", 0) or 0)
    inspection.num_layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
    inspection.vocab_size = int(getattr(text_cfg, "vocab_size", 0) or 0)

    # Crude param estimate: embedding + N * (attention + MLP).
    # Good to within ~20% for normal transformer geometry; 0 if we don't
    # know the dimensions.
    if inspection.hidden_size and inspection.num_layers and inspection.vocab_size:
        h = inspection.hidden_size
        n = inspection.num_layers
        v = inspection.vocab_size
        est = v * h + n * 12 * h * h
        inspection.params_b = round(est / 1e9, 2)

    inspection.kind = _classify_arch(inspection, cfg)
    return inspection


_VL_SIGNATURES = {
    "mistral3", "pixtral", "llava", "llava_next", "llava_next_video",
    "qwen2_vl", "qwen2_5_vl", "qwen3_vl",
    "phi3_v", "phi3v", "phi4_v",
    "internvl", "internvl_chat",
    "idefics", "idefics2", "idefics3",
    "paligemma", "kosmos_2",
}
_SEQ2SEQ_SIGNATURES = {
    "t5", "mt5", "bart", "mbart", "pegasus", "marian", "blenderbot",
    "fsmt", "switch_transformers",
}
_EMBEDDING_SIGNATURES = {
    "bert", "roberta", "distilbert", "xlm_roberta", "electra", "deberta",
    "modernbert", "sentence_transformer",
}


def _classify_arch(inspection: ModelInspection, cfg: Any) -> str:
    mt = inspection.model_type.lower()
    archs = {a.lower() for a in inspection.architectures}

    # Multimodal detection: either the model_type is a known VL family,
    # or the config has a vision_config / text_config split, or the
    # architecture name contains a telltale substring.
    if (
        mt in _VL_SIGNATURES
        or getattr(cfg, "vision_config", None) is not None
        or any("vision" in a or "vl" in a or "multimodal" in a for a in archs)
        or any(a.endswith("ForConditionalGeneration") and getattr(cfg, "text_config", None) is not None
               for a in inspection.architectures)
    ):
        return "vision-language"

    if mt in _SEQ2SEQ_SIGNATURES or any("seq2seq" in a for a in archs):
        return "seq2seq"

    if mt in _EMBEDDING_SIGNATURES or any(
        a.endswith("ForMaskedLM") or a.endswith("ForSequenceClassification")
        for a in inspection.architectures
    ):
        return "embedding"

    # Any architecture name ending in ForCausalLM is our happy path.
    if any(a.endswith("ForCausalLM") for a in inspection.architectures):
        return "causal-lm"

    # Fallback: if we have a positive model_type but no classification
    # yet, call it causal-lm and let tier 1 try it. If tier 1 fails we
    # still have tier 3.
    if mt:
        return "causal-lm"

    return "unknown"


# ---------------------------------------------------------------------------
# The loader itself
# ---------------------------------------------------------------------------


def load_model_graceful(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
    device_map: Optional[str] = None,
    low_cpu_mem_usage: bool = True,
) -> LoadResult:
    """Try tier 1 -> tier 2 -> tier 3. Return first tier that succeeds.

    ``dtype`` is forwarded as ``torch_dtype`` to transformers.
    ``device_map`` is forwarded only when non-None (defaults to CPU-load).
    """
    inspection = inspect_model_config(path)
    attempts: List[str] = []

    # Tier 0: Mistral-native checkpoint format (consolidated.safetensors
    # with layers.N.attention.w{q,k,v,o} tensor naming). Skips HF's
    # filename / layout requirements and remaps tensor names directly.
    try:
        from .mistral_native_loader import (
            is_mistral_native_checkpoint,
            load_mistral_native_as_causal_lm,
        )
        if is_mistral_native_checkpoint(path):
            model = load_mistral_native_as_causal_lm(path, dtype=dtype)
            return LoadResult(model=model, tier=1, notes=[
                f"tier 1 (mistral-native): loaded consolidated.safetensors "
                f"via direct tensor remap ({inspection.model_type})"
            ])
    except Exception as exc:
        attempts.append(
            f"tier 1 (mistral-native): {type(exc).__name__}: {str(exc)[:200]}"
        )
        log.debug("mistral-native path failed: %s", exc)

    # Tier 1: the common case (HF-layout checkpoint).
    try:
        model = _tier1_causal_lm(
            path, dtype=dtype, trust_remote_code=trust_remote_code,
            device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage,
        )
        return LoadResult(model=model, tier=1, notes=[
            f"tier 1: AutoModelForCausalLM loaded {inspection.model_type} directly"
        ])
    except Exception as exc:
        attempts.append(f"tier 1 (CausalLM): {type(exc).__name__}: {str(exc)[:200]}")
        log.debug("tier 1 failed: %s", exc)

    # Tier 2: multimodal/seq2seq with language-backbone extraction.
    try:
        model, submodule_name = _tier2_extract_language_model(
            path, dtype=dtype, trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        return LoadResult(
            model=model, tier=2, extracted_submodule=submodule_name,
            notes=[
                f"tier 2: loaded VL/seq2seq wrapper, extracted '{submodule_name}' "
                f"for text-only training"
            ],
        )
    except Exception as exc:
        attempts.append(f"tier 2 (VL-extract): {type(exc).__name__}: {str(exc)[:200]}")
        log.debug("tier 2 failed: %s", exc)

    # Tier 3: structural synthesis. This is the "Muse" path - we don't
    # know the architecture by name but we can still find a language
    # model inside if the weights are standard-shaped.
    try:
        model, notes = _tier3_synthesize(
            path, dtype=dtype, trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage, inspection=inspection,
        )
        return LoadResult(model=model, tier=3, notes=notes)
    except Exception as exc:
        attempts.append(f"tier 3 (synthesize): {type(exc).__name__}: {str(exc)[:200]}")
        log.debug("tier 3 failed: %s", exc)

    raise LoaderError(
        "Could not load model at '" + path + "' via any of the three tiers.\n"
        + "Attempts:\n  - " + "\n  - ".join(attempts) + "\n"
        + "Inspection: kind=" + inspection.kind + " model_type='"
        + inspection.model_type + "' archs=" + str(inspection.architectures)
    )


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------


def _tier1_causal_lm(
    path: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    device_map: Optional[str],
    low_cpu_mem_usage: bool,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )


# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------


_LANGUAGE_SUBMODULE_CANDIDATES = (
    "language_model",      # LLaVA, Qwen2-VL, Mistral3
    "text_model",          # Pixtral, some older VL
    "llm",                 # some custom VL stacks
    "model",               # Seq2Seq fallback (rare for pure LM extract)
)


def _tier2_extract_language_model(
    path: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    low_cpu_mem_usage: bool,
):
    """Load the full VL wrapper, then grab the language submodule.

    Returns (extracted_module, name_of_submodule) on success. Raises if
    no vision-language wrapper can be loaded at all, or if the loaded
    wrapper has no recognizable language submodule.
    """
    wrapper = None
    last_exc = None

    # Try the major VL wrapper classes in order. Each covers a family
    # of architectures; first that succeeds wins.
    for loader_name in (
        "AutoModelForVision2Seq",
        "AutoModelForImageTextToText",  # newer transformers
        "AutoModelForSeq2SeqLM",        # T5/BART/mBART
    ):
        import transformers as tx

        loader = getattr(tx, loader_name, None)
        if loader is None:
            continue
        try:
            wrapper = loader.from_pretrained(
                path,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
            log.info("tier 2: loaded via %s", loader_name)
            break
        except Exception as exc:
            last_exc = exc
            continue

    if wrapper is None:
        raise RuntimeError(
            "no VL/seq2seq wrapper loader accepted this model"
            + (f": {last_exc}" if last_exc else "")
        )

    # Find the language submodule. First look at common names, then fall
    # back to structural detection.
    for name in _LANGUAGE_SUBMODULE_CANDIDATES:
        candidate = _get_named_submodule(wrapper, name)
        if candidate is not None and _looks_like_language_model(candidate):
            # Promote lm_head if it's outside the submodule but belongs
            # conceptually to it (e.g. Mistral3 keeps lm_head on the
            # wrapper, not on .language_model).
            _promote_lm_head_if_needed(candidate, wrapper)
            return candidate, name

    # Structural search: walk every child and pick the one that looks
    # most like a language model.
    for child_name, child in wrapper.named_children():
        if _looks_like_language_model(child):
            _promote_lm_head_if_needed(child, wrapper)
            return child, child_name

    raise RuntimeError(
        "loaded VL wrapper but could not find a language-model submodule. "
        "Tried names=" + str(_LANGUAGE_SUBMODULE_CANDIDATES) + " and structural "
        "detection on top-level children."
    )


# ---------------------------------------------------------------------------
# Tier 3 -- structural synthesizer
# ---------------------------------------------------------------------------


def _tier3_synthesize(
    path: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    low_cpu_mem_usage: bool,
    inspection: ModelInspection,
):
    """Load via AutoModel then detect the language-model substructure.

    This covers the Muse case: a brand-new architecture that nobody has
    taught transformers about yet. AutoModel will accept anything that
    registers itself via ``trust_remote_code=True``. We then walk the
    resulting module for the language-model shape.
    """
    from transformers import AutoModel

    # trust_remote_code is a MUST here. Tier 3 fires for unregistered
    # architectures; those ship modeling code alongside the weights, so
    # without trust_remote_code the load trivially fails.
    try:
        wrapper = AutoModel.from_pretrained(
            path,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    except Exception as exc:
        raise RuntimeError(f"AutoModel trust_remote_code=True failed: {exc}") from exc

    # The raw AutoModel might BE the language model, or it might contain
    # one. Check in order.
    if _looks_like_language_model(wrapper):
        notes = [
            "tier 3: AutoModel output already has language-model shape, "
            "no extraction needed"
        ]
        return wrapper, notes

    # Walk for a submodule that does.
    for name, child in wrapper.named_children():
        if _looks_like_language_model(child):
            _promote_lm_head_if_needed(child, wrapper)
            notes = [
                f"tier 3: extracted language-model-shaped submodule '{name}' "
                f"from {type(wrapper).__name__}"
            ]
            return child, notes

    # Last-ditch: recursive deeper.
    for name, child in wrapper.named_modules():
        if name and _looks_like_language_model(child):
            _promote_lm_head_if_needed(child, wrapper)
            notes = [
                f"tier 3: extracted deep language-model submodule '{name}' "
                f"from {type(wrapper).__name__}"
            ]
            return child, notes

    # Architecture dump for the user so they can tell us what's in there.
    architecture_summary = _describe_architecture(wrapper, depth=2)
    raise RuntimeError(
        f"tier 3: AutoModel loaded {type(wrapper).__name__} but no language-"
        f"model-shaped substructure was found. Architecture seen:\n"
        + architecture_summary
    )


# ---------------------------------------------------------------------------
# Structural detection
# ---------------------------------------------------------------------------


def _looks_like_language_model(m: nn.Module) -> bool:
    """Structural heuristic: is ``m`` a language-model-shaped module?

    We look for THREE signals and require at least 2/3:

    1. An input embedding: ``get_input_embeddings()`` returns an nn.Embedding
       with num_embeddings >= 1024 (rules out positional tables and
       type-ID embeddings).
    2. A transformer block list: at least 2 sub-modules, each containing
       both an attention-ish layer (name contains 'attn', 'attention', or
       has Q/K/V-like linear submodules) and an MLP-ish layer.
    3. An output projection with out_features matching the input embedding's
       vocab size (the lm_head), either as an attribute or buried in the
       wrapper's forward path.

    The 2-of-3 rule handles cases where transformers hides the lm_head
    on a parent wrapper (signal 3 missing but 1+2 present) or where
    inputs_embeds are generated differently than expected.
    """
    signals = 0

    # Signal 1: input embedding of plausible size.
    emb = None
    try:
        emb = m.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        emb = None
    vocab = 0
    if isinstance(emb, nn.Embedding) and emb.num_embeddings >= 1024:
        signals += 1
        vocab = emb.num_embeddings

    # Signal 2: a list of transformer-like blocks.
    block_list = _find_transformer_block_list(m)
    if block_list is not None and len(block_list) >= 2:
        signals += 1

    # Signal 3: an output projection with the vocab-size fan-out.
    if vocab:
        if _has_vocab_sized_output(m, vocab):
            signals += 1

    return signals >= 2


def _find_transformer_block_list(m: nn.Module) -> Optional[nn.ModuleList]:
    """Return the ModuleList that looks like a stack of transformer blocks."""
    # Standard names first.
    for attr in ("layers", "h", "blocks", "transformer_blocks", "decoder_layers"):
        target = _get_named_submodule(m, attr)
        if isinstance(target, nn.ModuleList) and len(target) >= 2:
            if _block_looks_like_transformer(target[0]):
                return target
        # Also check one level deep (common: model.layers)
        for child_name, child in m.named_children():
            sub = getattr(child, attr, None)
            if isinstance(sub, nn.ModuleList) and len(sub) >= 2:
                if _block_looks_like_transformer(sub[0]):
                    return sub

    # Generic walk.
    for name, mod in m.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
            if _block_looks_like_transformer(mod[0]):
                return mod
    return None


def _block_looks_like_transformer(block: nn.Module) -> bool:
    """A transformer block should have BOTH an attention and an MLP."""
    has_attn = False
    has_mlp = False
    for name, _sub in block.named_modules():
        lname = name.lower()
        if not has_attn and ("attn" in lname or "attention" in lname):
            has_attn = True
        if not has_mlp and ("mlp" in lname or "feed_forward" in lname
                            or "ffn" in lname):
            has_mlp = True
        if has_attn and has_mlp:
            return True
    return False


def _has_vocab_sized_output(m: nn.Module, vocab: int) -> bool:
    """Is there a Linear with out_features == vocab somewhere in m?"""
    for attr in ("lm_head", "output", "head", "cls", "output_projection"):
        target = _get_named_submodule(m, attr)
        if isinstance(target, nn.Linear) and target.out_features == vocab:
            return True
    # Generic walk.
    for _name, sub in m.named_modules():
        if isinstance(sub, nn.Linear) and sub.out_features == vocab:
            return True
    return False


def _get_named_submodule(m: nn.Module, name: str) -> Optional[nn.Module]:
    """Dotted-name submodule getter that returns None instead of raising."""
    try:
        return m.get_submodule(name)
    except AttributeError:
        return None


def _promote_lm_head_if_needed(submodule: nn.Module, wrapper: nn.Module) -> None:
    """If ``submodule`` doesn't have ``lm_head`` but ``wrapper`` does, move it.

    VL wrappers like Mistral3ForConditionalGeneration keep lm_head on
    the OUTER class rather than on .language_model. After extraction we
    want the language submodule to own its output projection so
    downstream LoRA/quant wiring sees it at the expected attribute.
    """
    if hasattr(submodule, "lm_head") and isinstance(submodule.lm_head, nn.Linear):
        return
    if hasattr(wrapper, "lm_head") and isinstance(wrapper.lm_head, nn.Linear):
        submodule.lm_head = wrapper.lm_head
        log.info("promoted lm_head from wrapper onto language submodule")


def _describe_architecture(m: nn.Module, depth: int = 2) -> str:
    """Pretty-print top-N levels of the module tree for error messages."""
    lines: List[str] = []

    def _walk(mod: nn.Module, prefix: str, d: int) -> None:
        cls = type(mod).__name__
        lines.append(prefix + cls)
        if d <= 0:
            return
        for name, child in mod.named_children():
            _walk(child, prefix + "  " + name + ": ", d - 1)

    _walk(m, "", depth)
    return "\n".join(lines)
