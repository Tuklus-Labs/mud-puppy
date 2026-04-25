"""Convert a BF16 gpt-oss GGUF into HF safetensors.

The input must be an all-BF16 GGUF (no MXFP4 left). Produce one with:

    llama-quantize --allow-requantize <input.gguf> <bf16.gguf> BF16

This script then:
    1. Reads every tensor from the BF16 GGUF.
    2. Maps GGUF tensor names to HF gpt-oss names, including the MoE
       expert reshape (GGUF stores gate and up separately + per-expert
       as [E, intermediate, hidden]; HF wants them fused + transposed
       as gate_up_proj of shape [E, hidden, 2*intermediate]).
    3. Writes sharded safetensors + model.safetensors.index.json.

Usage:
    python scripts/gguf_to_hf_gpt_oss.py \\
        --gguf /home/aegis/Models/gpt-oss-20b-bf16.gguf \\
        --hf-dir /home/aegis/Models/gpt-oss-20b-hf \\
        --shard-size-gb 5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import gguf
import numpy as np
import torch
from safetensors.torch import save_file


# --------------------------------------------------------------------------
# Tensor name maps
# --------------------------------------------------------------------------


# Top-level tensors (not layer-specific).
TOP_LEVEL_NAME_MAP: Dict[str, str] = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


# Per-layer suffixes. GGUF tensor name = "blk.{N}.{gguf_suffix}",
# HF tensor name = "model.layers.{N}.{hf_suffix}".
LAYER_NAME_MAP: Dict[str, str] = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_output.bias": "self_attn.o_proj.bias",
    "attn_sinks.weight": "self_attn.sinks",
    "ffn_gate_inp.weight": "mlp.router.weight",
    "ffn_gate_inp.bias": "mlp.router.bias",
    # MoE experts handled specially (fused gate+up, transposed layout):
    #   ffn_gate_exps.weight + ffn_up_exps.weight -> mlp.experts.gate_up_proj
    #   ffn_gate_exps.bias  + ffn_up_exps.bias  -> mlp.experts.gate_up_proj_bias
    #   ffn_down_exps.weight                     -> mlp.experts.down_proj (transposed)
    #   ffn_down_exps.bias                       -> mlp.experts.down_proj_bias
}


LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


# --------------------------------------------------------------------------
# Tensor decoding
# --------------------------------------------------------------------------


def _read_tensor(t: "gguf.ReaderTensor") -> torch.Tensor:
    """Return tensor as an fp32/fp16/bf16 torch tensor with correct shape.

    GGUF ``t.data`` returns a numpy view; dtypes 0/1 are fp32/fp16 which
    numpy handles natively. For bf16 (dtype 30 in the gguf enum) numpy
    returns raw uint8 bytes and we must reinterpret.
    """
    name = t.name
    # ggml shape is reversed relative to numpy/torch; the gguf lib returns
    # data already in numpy order so t.data.shape is the one to trust.
    data = t.data  # numpy array

    if data.dtype == np.float32:
        return torch.from_numpy(data.copy())
    if data.dtype == np.float16:
        return torch.from_numpy(data.copy())

    # Anything else should be bf16 after --allow-requantize ... BF16.
    # Raw bytes: reinterpret as bf16 and reshape.
    if data.dtype == np.uint8:
        # The *true* shape is ggml-reported [cols, rows, ...] reversed.
        ggml_shape = [int(s) for s in t.shape]
        torch_shape = list(reversed(ggml_shape))
        # Number of bf16 elements we expect.
        n_elems = 1
        for d in torch_shape:
            n_elems *= d
        if data.size != n_elems * 2:
            raise RuntimeError(
                f"tensor {name!r}: expected {n_elems*2} bytes for bf16 shape {torch_shape}, "
                f"got {data.size}"
            )
        # Reinterpret raw bytes as bf16.
        bf16 = (
            torch.from_numpy(data.reshape(-1).copy())
            .view(torch.bfloat16)
        )
        return bf16.reshape(torch_shape)

    raise RuntimeError(f"tensor {name!r}: unsupported numpy dtype {data.dtype}")


# --------------------------------------------------------------------------
# Per-layer MoE remap
# --------------------------------------------------------------------------


def _fuse_moe_layer(
    layer_tensors: Dict[str, torch.Tensor],
    layer_idx: int,
) -> Dict[str, torch.Tensor]:
    """Build the 4 MoE tensors HF expects from the 6 GGUF inputs.

    Input (per layer, all [E, X, Y] torch tensors):
        ffn_gate_exps.weight  [E, intermediate, hidden]
        ffn_up_exps.weight    [E, intermediate, hidden]
        ffn_down_exps.weight  [E, hidden, intermediate]
        ffn_gate_exps.bias    [E, intermediate]
        ffn_up_exps.bias      [E, intermediate]
        ffn_down_exps.bias    [E, hidden]

    Output (HF gpt_oss naming):
        mlp.experts.gate_up_proj       [E, hidden, 2*intermediate]
        mlp.experts.gate_up_proj_bias  [E, 2*intermediate]
        mlp.experts.down_proj          [E, intermediate, hidden]
        mlp.experts.down_proj_bias     [E, hidden]
    """
    prefix = f"model.layers.{layer_idx}.mlp.experts."

    def require(key: str) -> torch.Tensor:
        if key not in layer_tensors:
            raise KeyError(f"layer {layer_idx}: missing GGUF tensor {key!r}")
        return layer_tensors.pop(key)

    gate_w = require("ffn_gate_exps.weight")      # [E, I, H]
    up_w = require("ffn_up_exps.weight")          # [E, I, H]
    down_w = require("ffn_down_exps.weight")      # [E, H, I]
    gate_b = require("ffn_gate_exps.bias")        # [E, I]
    up_b = require("ffn_up_exps.bias")            # [E, I]
    down_b = require("ffn_down_exps.bias")        # [E, H]

    # HF convention: gate_up_proj [E, H, 2I] where the last axis is
    # INTERLEAVED [gate_0, up_0, gate_1, up_1, ...]. HF unpacks with
    # gate = gate_up[..., ::2], up = gate_up[..., 1::2]. Concatenation
    # would put gate then up and silently break the SwiGLU.
    gate_w_t = gate_w.transpose(-1, -2).contiguous()  # [E, H, I]
    up_w_t = up_w.transpose(-1, -2).contiguous()      # [E, H, I]
    E, H, I = gate_w_t.shape
    # Stack on a new last axis -> [E, H, I, 2], then reshape to [E, H, 2I].
    # Row-major reshape keeps pairs adjacent, producing the ::2 / 1::2 layout.
    gate_up = torch.stack([gate_w_t, up_w_t], dim=-1).reshape(E, H, 2 * I).contiguous()
    gate_up_bias = torch.stack([gate_b, up_b], dim=-1).reshape(E, 2 * I).contiguous()

    # down_proj: HF wants [E, I, H]
    down_w_hf = down_w.transpose(-1, -2).contiguous()  # [E, I, H]

    return {
        prefix + "gate_up_proj": gate_up,
        prefix + "gate_up_proj_bias": gate_up_bias,
        prefix + "down_proj": down_w_hf,
        prefix + "down_proj_bias": down_b,
    }


# --------------------------------------------------------------------------
# Shard writer
# --------------------------------------------------------------------------


def _write_shards(
    state_dict: Dict[str, torch.Tensor],
    out_dir: Path,
    shard_size_bytes: int,
) -> Dict[str, str]:
    """Write tensors across multiple safetensors shards.

    Returns a weight-map dict keyed by tensor name whose values are the
    shard filename that tensor lives in. Suitable for the HF
    ``model.safetensors.index.json`` file.
    """
    # Stable ordering so reruns produce identical sharding.
    sorted_names = sorted(state_dict.keys())
    shards: List[List[str]] = [[]]
    shard_sizes: List[int] = [0]

    for name in sorted_names:
        t = state_dict[name]
        tsize = t.numel() * t.element_size()
        if shard_sizes[-1] + tsize > shard_size_bytes and shards[-1]:
            shards.append([])
            shard_sizes.append(0)
        shards[-1].append(name)
        shard_sizes[-1] += tsize

    total = len(shards)
    weight_map: Dict[str, str] = {}
    total_bytes = sum(shard_sizes)

    for i, names in enumerate(shards):
        fname = f"model-{i+1:05d}-of-{total:05d}.safetensors"
        path = out_dir / fname
        tensors_this_shard = {n: state_dict[n].contiguous() for n in names}
        save_file(tensors_this_shard, str(path))
        for n in names:
            weight_map[n] = fname
        print(f"  wrote {fname}  ({shard_sizes[i]/1e9:.2f} GB, {len(names)} tensors)")

    print(f"total: {total_bytes/1e9:.2f} GB across {total} shards")
    return weight_map


# --------------------------------------------------------------------------
# Main conversion
# --------------------------------------------------------------------------


def convert(gguf_path: Path, out_dir: Path, shard_size_gb: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"reading {gguf_path} ...")
    reader = gguf.GGUFReader(str(gguf_path))
    print(f"  {len(reader.tensors)} tensors in GGUF")

    # Collect per-layer GGUF tensors; the MoE suffix needs 6 tensors per layer
    # to be fused. Everything else gets direct-mapped.
    per_layer: Dict[int, Dict[str, torch.Tensor]] = {}
    state_dict: Dict[str, torch.Tensor] = {}

    for t in reader.tensors:
        name = t.name
        tensor = _read_tensor(t)

        # Top-level?
        if name in TOP_LEVEL_NAME_MAP:
            state_dict[TOP_LEVEL_NAME_MAP[name]] = tensor
            continue

        # Per-layer?
        m = LAYER_RE.match(name)
        if not m:
            print(f"  skipping unknown tensor: {name}")
            continue
        layer_idx = int(m.group(1))
        suffix = m.group(2)

        if suffix in LAYER_NAME_MAP:
            hf_name = f"model.layers.{layer_idx}.{LAYER_NAME_MAP[suffix]}"
            state_dict[hf_name] = tensor
        elif suffix in (
            "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
            "ffn_gate_exps.bias",   "ffn_up_exps.bias",   "ffn_down_exps.bias",
        ):
            per_layer.setdefault(layer_idx, {})[suffix] = tensor
        else:
            print(f"  skipping unknown layer tensor: {name}")

    # Fuse expert tensors per layer.
    print(f"fusing MoE experts for {len(per_layer)} layers ...")
    for layer_idx, layer_t in per_layer.items():
        state_dict.update(_fuse_moe_layer(layer_t, layer_idx))
        if layer_t:
            # Any keys not consumed by _fuse_moe_layer would be a bug.
            raise RuntimeError(f"layer {layer_idx}: unused expert tensors {list(layer_t)}")

    print(f"final HF state dict: {len(state_dict)} tensors")

    # Sanity check: sizes should be in GB range for 20B model.
    total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    print(f"total weight bytes: {total_bytes/1e9:.2f} GB")

    # Write sharded safetensors.
    shard_size_bytes = int(shard_size_gb * 1024**3)
    print(f"writing shards (size {shard_size_gb} GB each) ...")
    weight_map = _write_shards(state_dict, out_dir, shard_size_bytes)

    # Write the index file.
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    index_path = out_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"wrote {index_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--hf-dir", required=True, type=Path)
    p.add_argument("--shard-size-gb", type=float, default=5.0)
    args = p.parse_args()
    convert(args.gguf, args.hf_dir, args.shard_size_gb)


if __name__ == "__main__":
    main()
