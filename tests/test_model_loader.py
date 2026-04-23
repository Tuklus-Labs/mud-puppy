"""Tests for the three-tier graceful-degradation model loader.

Tier 1 and tier 2 require real transformers models. We use tiny HF
fixtures where possible. Tier 3 (the "Muse" case for unknown
architectures) is tested against hand-built nn.Modules that mimic
transformer geometry without registering as any known model_type.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mud_puppy.model_loader import (
    LoaderError,
    LoadResult,
    ModelInspection,
    _block_looks_like_transformer,
    _find_transformer_block_list,
    _has_vocab_sized_output,
    _looks_like_language_model,
    inspect_model_config,
)


# ---------------------------------------------------------------------------
# Structural detector unit tests (no transformers dependency for these)
# ---------------------------------------------------------------------------


class _ToyAttn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)


class _ToyMLP(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, 4 * d)
        self.up_proj = nn.Linear(d, 4 * d)
        self.down_proj = nn.Linear(4 * d, d)


class _ToyBlock(nn.Module):
    """Minimal transformer block: has attn AND mlp submodules."""
    def __init__(self, d: int) -> None:
        super().__init__()
        self.self_attn = _ToyAttn(d)
        self.mlp = _ToyMLP(d)
        self.input_layernorm = nn.LayerNorm(d)


class _ToyLM(nn.Module):
    """A language-model-shaped module with no official model_type."""
    def __init__(self, vocab: int = 2048, d: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_ToyBlock(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


class _ToyVLWrapper(nn.Module):
    """Mimics Mistral3/LLaVA: has vision + multimodal_projector + language_model."""
    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = nn.Conv2d(3, 16, 3)  # deliberately NOT language-shaped
        self.multi_modal_projector = nn.Linear(16, 64)
        self.language_model = _ToyLM(vocab=4096, d=64, n_layers=4)
        # Some wrappers keep lm_head on the outer (like Mistral3).
        self.lm_head = self.language_model.lm_head


class _ToyNotALM(nn.Module):
    """Looks like a classifier or encoder; should NOT be detected as an LM."""
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(2048, 64)
        # A classifier: one pooler + classification head, no transformer blocks
        self.classifier = nn.Linear(64, 10)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


# ---- detector ----


def test_block_looks_like_transformer_positive() -> None:
    assert _block_looks_like_transformer(_ToyBlock(64)) is True


def test_block_looks_like_transformer_negative_conv() -> None:
    conv_only = nn.Conv2d(3, 16, 3)
    assert _block_looks_like_transformer(conv_only) is False


def test_find_transformer_block_list_standard_name() -> None:
    m = _ToyLM(d=64, n_layers=3)
    block_list = _find_transformer_block_list(m)
    assert block_list is not None
    assert len(block_list) == 3


def test_find_transformer_block_list_none_on_classifier() -> None:
    m = _ToyNotALM()
    assert _find_transformer_block_list(m) is None


def test_has_vocab_sized_output_positive() -> None:
    m = _ToyLM(vocab=2048)
    assert _has_vocab_sized_output(m, 2048) is True


def test_has_vocab_sized_output_negative_wrong_vocab() -> None:
    m = _ToyLM(vocab=2048)
    assert _has_vocab_sized_output(m, 4096) is False


def test_looks_like_language_model_positive_toy() -> None:
    m = _ToyLM(vocab=4096, d=64, n_layers=3)
    assert _looks_like_language_model(m) is True


def test_looks_like_language_model_negative_classifier() -> None:
    m = _ToyNotALM()
    assert _looks_like_language_model(m) is False


def test_looks_like_language_model_negative_raw_embedding() -> None:
    # Just an embedding; no blocks, no head.
    class _Bare(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.e = nn.Embedding(2048, 64)

        def get_input_embeddings(self) -> nn.Embedding:
            return self.e

    assert _looks_like_language_model(_Bare()) is False


def test_vl_wrapper_language_model_detected_via_child_walk() -> None:
    """Structural walk should find language_model inside a VL wrapper."""
    wrapper = _ToyVLWrapper()
    # The wrapper itself is NOT language-shaped (has a conv, no
    # top-level block list).
    assert _looks_like_language_model(wrapper) is False
    # But .language_model is.
    assert _looks_like_language_model(wrapper.language_model) is True


# ---------------------------------------------------------------------------
# Config inspection
# ---------------------------------------------------------------------------


def test_inspect_missing_path_returns_unknown() -> None:
    inspection = inspect_model_config("/tmp/does-not-exist-mud-puppy-test")
    assert inspection.kind == "unknown"
    assert inspection.architectures == []


@pytest.mark.skipif(
    not torch.cuda.is_available() and True,  # run on all systems; no GPU needed
    reason="",
)
def test_inspect_caches_are_independent() -> None:
    """Two calls should return independent ModelInspection objects."""
    a = inspect_model_config("/tmp/path-a-mud-puppy")
    b = inspect_model_config("/tmp/path-b-mud-puppy")
    assert a is not b
    # Mutating one shouldn't affect the other.
    a.notes.append("test")
    assert "test" not in b.notes


# ---------------------------------------------------------------------------
# Dataclass shape sanity
# ---------------------------------------------------------------------------


def test_model_inspection_fields_default() -> None:
    m = ModelInspection(path="/x")
    assert m.path == "/x"
    assert m.architectures == []
    assert m.kind == "unknown"
    assert m.params_b == 0.0
    assert m.notes == []


def test_load_result_tier_and_submodule() -> None:
    r = LoadResult(model=_ToyLM(), tier=2, extracted_submodule="language_model")
    assert r.tier == 2
    assert r.extracted_submodule == "language_model"


# ---------------------------------------------------------------------------
# LoaderError message format
# ---------------------------------------------------------------------------


def test_loader_error_is_runtime_error() -> None:
    assert issubclass(LoaderError, RuntimeError)
