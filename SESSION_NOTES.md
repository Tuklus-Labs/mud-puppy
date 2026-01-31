# Mud-Puppy Session Notes

## 2026-01-31: ROCm Compatibility + QLoRA + Streaming

### What Was Done

**Original Request:** Fix Mud-Puppy training issues for voice intent classifier, then pressure test all features. Subsequently: implement ROCm-native QLoRA and layer streaming.

### Fixes Applied

1. **Transformers 5.0 Compatibility**
   - `tokenizer=` parameter renamed to `processing_class=` in SFTTrainer
   - `report_to=None` no longer valid, changed to `[]`
   - `AutoModelForVision2Seq` import made optional (not in all transformers builds)

2. **ROCm Device Handling**
   - `device_map="auto"` causes segfaults on ROCm - load to CPU first with `device_map=None`, then `.to("cuda")`
   - GPU 0 = 7900 XTX (24GB), GPU 1 = CPU exposed as HIP - must set `CUDA_VISIBLE_DEVICES=0`
   - bf16 detection with fp16 fallback for GPUs that don't support bf16

3. **QLoRA (ROCm-native, no bitsandbytes)**
   - `Linear4bit` class now inherits from `nn.Linear` so PEFT recognizes it as valid LoRA target
   - Added `_prepare_model_for_kbit_training_rocm()` function
   - Quantization: weights stored as 4-bit with per-row scaling, dequantized for forward pass
   - Base weights frozen (`requires_grad=False`), LoRA adapters remain trainable

4. **Streaming (Layer Streaming for Memory Efficiency)**
   - Rewrote `StreamWrapper` to keep layers on GPU through entire forward/backward pass
   - Previous approach moved layers off GPU during forward, breaking gradient computation
   - Added validation to reject streaming+LoRA combinations (incompatible)

### Key Files Changed

- `mud_puppy/trainer.py` - Core training logic, ROCm fixes, streaming
- `mud_puppy/bnb_rocm.py` - ROCm-native 4-bit quantization
- `mud_puppy/config.py` - Validation for streaming+LoRA incompatibility
- `tests/pressure_test.py` - Comprehensive test suite

### Pressure Test Results (All Pass)

```
Total: 11 passed, 0 failed, 0 skipped

1. LoRA + fp16
2. LoRA + bf16
3. LoRA + gradient_checkpointing
4. Full fine-tuning + fp16
5. QLoRA (4-bit ROCm-native)
6. LoRA + higher rank (r=32)
7. LoRA + streaming validation (correctly rejected)
8. Full fine-tuning + streaming
9. LoRA + cosine scheduler
10. LoRA + early stopping
11. GPTQ quantization
```

### Commits

```
259efb2 Fix ROCm and transformers 5.0 compatibility
e8013f0 Add streaming+LoRA validation and pressure test suite
d616f70 feat: ROCm-native QLoRA and streaming support
```

### Gotchas for Future Sessions

- **Don't use device_map="auto" on ROCm (with iGPU)** - The 9900X iGPU (gfx1036) is exposed as cuda:1 but segfaults on basic ops. Set `HIP_VISIBLE_DEVICES=0` to hide it, or disable iGPU in BIOS
- **Linear4bit must inherit nn.Linear** - PEFT won't recognize it as LoRA target otherwise
- **Streaming + LoRA don't mix** - LoRA modules are added after streaming hooks
- **Streaming backward pass** - layers must stay on GPU through forward+backward, not just forward
- **transformers 5.0** - many parameter names changed, check deprecation warnings

### Voice Intent Classifier

Successfully trained on 84 examples using TinyLlama-1.1B with LoRA:
- Loss: 0.72 → 0.16
- Integrated into `~/Projects/scripts/voice-control.py`
