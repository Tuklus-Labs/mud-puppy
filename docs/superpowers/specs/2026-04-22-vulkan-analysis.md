# Vulkan for mud-puppy: analysis

Date: 2026-04-22
Status: Research / decision document (no implementation yet)
Question: Could Vulkan be a viable backend for mud-puppy, for either
training or inference, and if so, where does it fit?

## TL;DR

| Use case | Vulkan viability | Our recommendation |
|---|---|---|
| Training (forward + backward + optimizer) | Essentially none | Do not pursue |
| Inference against mud-puppy-trained weights | Strong, via llama.cpp | Add GGUF export path, then llama.cpp/Vulkan does the rest |
| Training kernel autonomy from ROCm | Possible via Kompute but prohibitive | Not this year |

## Why training on Vulkan does not make sense today

1. **PyTorch has no Vulkan compute backend for training.** The Vulkan
   backend that exists in PyTorch is the `vulkan` dispatch key, which is
   mobile-focused (Android) and is inference-only by design. It does
   not implement autograd, doesn't support most optimizers, and its
   kernel coverage is a small subset of even `cpu` dispatch. It has
   been in this state for years with no signal of change.
2. **Writing our own training graph on top of a Vulkan compute library
   is a multi-year effort.** Libraries exist (Kompute, ncnn, mace,
   MNN), but they are inference-oriented. Building autograd, gradient
   scaling, AdamW, LoRA, QLoRA, activation checkpointing, FSDP, etc.
   from scratch on top of raw SPIR-V compute shaders is the
   "replace PyTorch" project, not a "replace ROCm" project.
3. **The ROCm lock-in we would actually benefit from escaping is at
   inference time, not training time.** Our training targets are AMD
   (7900 XTX now, RDNA4 soon, possibly MI300/MI400 later). ROCm is
   required for serious training throughput on AMD regardless of
   Vulkan existing.

## Why Vulkan for inference IS attractive

1. **llama.cpp's Vulkan backend is production-grade.** It runs on every
   vendor (AMD, NVIDIA, Intel, Apple via MoltenVK), it does not require
   a CUDA or ROCm driver, and it delivers 70-95% of the native backend
   throughput in practice. For deploying mud-puppy-trained models to
   heterogeneous hardware (someone else's laptop, a Linux desktop
   without ROCm installed, a Steam Deck, etc.), Vulkan is the path of
   least friction.
2. **GGUF is the pivot format.** llama.cpp ingests GGUF. If mud-puppy
   knows how to export a trained model to GGUF, every
   llama.cpp-compatible runtime — including the Vulkan backend —
   immediately picks it up. We already have `--method gptq` that
   outputs a custom format; a `--export-gguf` post-training step would
   be additive, not a backend swap.
3. **Quantization formats align.** GGUF supports Q4_K_M, Q5_K_M, Q8_0,
   and related quant types that map cleanly from our INT4 (QLoRA) and
   MXFP4 outputs. Real GPTQ output (once the parity subagent lands)
   also round-trips to GGUF Q4_K.
4. **No driver dependency.** A user who trains on a ROCm workstation
   and ships to a Vulkan-only laptop (common scenario: train on Gary's
   7900 XTX, infer on a thin client or the AegisOS Lite drone stack)
   never has to install ROCm on the deployment machine. That's a real
   unlock for AegisOS multi-domain deployments.

## What a Vulkan export path looks like

Scope for a hypothetical `v0.5` feature:

1. **New CLI flag**: `--export-gguf` runs after training completes.
2. **New module**: `mud_puppy/gguf_export.py`
   - Convert the trained model to llama.cpp's expected tensor layout
     (column-major for some ops, specific naming conventions for
     LLaMA-style and GPT-2-style architectures).
   - Emit metadata: tokenizer, chat template, special tokens, rope
     scaling config, model arch.
   - Quantize on the fly: `--gguf-quant {f16, q8_0, q5_k_m, q4_k_m}`.
     Reuse our existing INT4/MXFP4 quantization routines where the
     GGUF format permits (Q4_K_M is close but not identical to our
     packed INT4 format; some repacking required).
3. **Test**: round-trip a tiny model through mud-puppy → GGUF →
   llama.cpp and verify token-level output matches baseline within
   tolerance.
4. **Docs**: a "Deploying trained models" section in README covering
   when to use mud-puppy native checkpoint vs GGUF vs HuggingFace
   format, with a sizing/latency table.

Estimated scope: 2-3 days focused engineering + a day of end-to-end
validation across 2-3 model families (LLaMA, Mistral, Qwen).

## What about just using llama.cpp as our inference backend in Phos?

Separate question but worth noting. mud-puppy-studio today only cares
about training; inference is handled elsewhere (AEGIS inference stack,
llama.cpp, ollama, etc.). If we wanted a "test this checkpoint
immediately after training" button in the studio:

- Add a `inference.run` IPC method in the Phos shell that shells out
  to `llama-server --gpu-layers -1 -m <gguf_path>` and opens a small
  chat pane.
- Requires the GGUF export path to exist first.
- Separately worthwhile; UX win for iterate-train-evaluate loops.

## Recommendation

1. **No Vulkan training backend.** The effort-to-benefit ratio is bad,
   and the ROCm stack on the 7900 XTX / RDNA4 is the right training
   target.
2. **Yes to GGUF export (future)**. This is the practical way mud-puppy
   users get Vulkan inference: the export path, not a rewrite. Plan
   for v0.5 or later.
3. **Watch `torch.distributed.pipelining` and PyTorch's native FP8
   evolution on ROCm.** If AMD's ROCm team lands hardware FP8 properly
   (already there on MI300+, RDNA4), we have a stronger story than
   wrapping Vulkan.
4. **If someone wants Vulkan-accelerated inference in the studio
   today**, the right answer is: export trained adapter to GGUF
   manually, run llama.cpp's Vulkan backend separately. Not a mud-puppy
   feature yet.

## One specific scenario to revisit

**AegisOS Lite drone stack, Intel Arc / integrated GPU deployment,
Jetson-style ARM targets**: if any of those becomes a real mud-puppy
deployment target, the GGUF export path moves from "nice to have" to
"required." That's the trigger to prioritize the work.
