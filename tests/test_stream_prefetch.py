"""Tests for LayerStreamer prefetch ring."""
import pytest
import torch
import torch.nn as nn
from mud_puppy.stream import LayerStreamer


class ToyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin(x)


class ToyTransformer(nn.Module):
    def __init__(self, n_layers=4, dim=64):
        super().__init__()
        self.embed = nn.Embedding(100, dim)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyBlock(dim) for _ in range(n_layers)])
        self.model.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, 100)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.model.layers:
            h = layer(h)
        h = self.model.norm(h)
        return self.lm_head(h)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_streamer_limits_resident_layers():
    model = ToyTransformer(n_layers=4, dim=64)
    streamer = LayerStreamer(model, prefetch_layers=2)

    assert streamer.device.type == "cuda"
    for layer in model.model.layers:
        assert next(layer.parameters()).device.type == "cpu"

    assert next(model.embed.parameters()).device.type == "cuda"
    assert next(model.lm_head.parameters()).device.type == "cuda"

    x = torch.randint(0, 100, (2, 8), device="cuda")
    out = model(x)
    assert out.shape == (2, 8, 100)

    resident = sum(
        1 for layer in model.model.layers
        if next(layer.parameters()).device.type == "cuda"
    )
    # After forward, at most K layers should be resident in ring slots.
    assert resident <= streamer.prefetch_layers + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_streamer_backward_works():
    model = ToyTransformer(n_layers=4, dim=64)
    LayerStreamer(model, prefetch_layers=2)
    x = torch.randint(0, 100, (2, 8), device="cuda")
    out = model(x)
    loss = out.sum()
    loss.backward()
    # Grads exist on embed/lm_head
    assert model.embed.weight.grad is not None
    assert model.lm_head.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_streamer_respects_prefetch_depth():
    model = ToyTransformer(n_layers=6, dim=64)
    streamer = LayerStreamer(model, prefetch_layers=3)
    assert streamer.prefetch_layers == 3
    assert len(streamer._ring_slots) == 3


def test_streamer_rejects_zero_prefetch_layers():
    """prefetch_layers=0 is invalid; LayerStreamer should refuse to build.

    CUDA availability is irrelevant -- the ValueError fires before the
    CUDA check, so this can run on any host.
    """
    model = ToyTransformer(n_layers=2, dim=16)
    with pytest.raises(ValueError, match="prefetch_layers"):
        LayerStreamer(model, prefetch_layers=0)
    with pytest.raises(ValueError, match="prefetch_layers"):
        LayerStreamer(model, prefetch_layers=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_streamer_pins_lora_adapters():
    """LoRA adapters must stay GPU-resident through eviction cycles."""
    peft = pytest.importorskip("peft")

    # Build the toy model and wrap with a peft LoRA config targeting the
    # Linear inside each ToyBlock.
    model = ToyTransformer(n_layers=4, dim=64)
    lora_config = peft.LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["lin"],
        lora_dropout=0.0,
        bias="none",
    )
    model = peft.get_peft_model(model, lora_config)

    # Peft wraps the model; _find_layers still needs model.model.layers.
    # peft's PeftModel exposes the base model at .base_model.model so the
    # layers live at model.base_model.model.model.layers. For the toy test,
    # reach in and use the underlying ToyTransformer.
    inner = model.base_model.model
    streamer = LayerStreamer(inner, prefetch_layers=2)

    def _assert_lora_on_gpu(stage: str) -> None:
        found_any = False
        for pname, param in inner.named_parameters():
            if ("lora_A" in pname) or ("lora_B" in pname):
                found_any = True
                assert param.device.type == "cuda", (
                    f"{stage}: LoRA param {pname} left GPU (device={param.device})"
                )
        assert found_any, "test bug: no LoRA params found on the model"

    # After wrap, LoRA must be on GPU.
    _assert_lora_on_gpu("post-wrap")

    # Forward pass cycles layers through ring slots, triggering evictions.
    x = torch.randint(0, 100, (2, 8), device="cuda")
    out = inner(x)
    _assert_lora_on_gpu("post-forward")

    # Backward must also keep LoRA on GPU.
    loss = out.sum()
    loss.backward()
    _assert_lora_on_gpu("post-backward")

    # Second forward pass: full eviction cycle repeats.
    _ = inner(x)
    _assert_lora_on_gpu("post-second-forward")
