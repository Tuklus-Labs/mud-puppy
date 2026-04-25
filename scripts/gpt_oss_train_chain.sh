#!/usr/bin/env bash
# Serial chain: train gpt-oss-20b LoRA -> merge -> heretic abliterate -> GGUF.
#
# Runs on the 7900 XTX using coder_sft.py (gpt-oss MoE-aware, MXFP4 attention,
# bf16 experts). After training lands, merges LoRA into the base, runs heretic
# to remove the policy-refusal direction, then exports GGUF Q4_K_M.

set -euo pipefail

MP_DIR="${MP_DIR:-/home/aegis/Projects/mud-puppy}"
OUT_ROOT="${OUT_ROOT:-$MP_DIR/outputs/gpt-oss-coder-chain}"
BASE_MODEL="${BASE_MODEL:-/home/aegis/Models/gpt-oss-20b-hf}"
DATA_GLOB="${DATA_GLOB:-$MP_DIR/training_data_sets/coder/mix.jsonl}"

SFT_OUT="$OUT_ROOT/sft"
MERGED_OUT="$OUT_ROOT/merged"
HERETIC_OUT="$OUT_ROOT/heretic"
GGUF_OUT="$OUT_ROOT/gguf"

mkdir -p "$OUT_ROOT" "$SFT_OUT" "$MERGED_OUT" "$HERETIC_OUT" "$GGUF_OUT"

log() { printf '[gpt-oss-chain %(%H:%M:%S)T] %s\n' -1 "$*"; }

# Charon helper (best-effort)
ch() { charon-emit milestone "gpt-oss-chain: $*" 2>/dev/null || true; }

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

log "===================================================================="
log "gpt-oss-20b training chain"
log "  base:      $BASE_MODEL"
log "  data:      $DATA_GLOB"
log "  outputs:   $OUT_ROOT"
log "===================================================================="

# ------------------------------------------------------------------
# Phase 1: LoRA SFT
# ------------------------------------------------------------------
log "Phase 1: LoRA SFT via scripts/coder_sft.py"
ch "phase 1 start (LoRA SFT)"

cd "$MP_DIR"
python3 scripts/coder_sft.py \
    --model "$BASE_MODEL" \
    --data-glob "$DATA_GLOB" \
    --output "$SFT_OUT" \
    --lr 5e-5 \
    --grad-accum 16 \
    --per-device-batch 1 \
    --max-seq-length 2048 \
    --lora-r 32 --lora-alpha 64 \
    --save-steps 200 --logging-steps 20 \
    --mxfp4-attn

ch "phase 1 done (LoRA adapter at $SFT_OUT)"
log "Phase 1 complete."

# ------------------------------------------------------------------
# Phase 2: merge LoRA into base to produce an HF-loadable merged dir
# ------------------------------------------------------------------
log "Phase 2: merge LoRA adapter into base model"
ch "phase 2 start (merge)"

python3 - <<PYEOF
import torch
from transformers import AutoTokenizer
from peft import PeftModel
from mud_puppy.model_loader import load_model_graceful

base = load_model_graceful(
    "$BASE_MODEL",
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=None,
    low_cpu_mem_usage=True,
).model

peft_model = PeftModel.from_pretrained(base, "$SFT_OUT")
merged = peft_model.merge_and_unload()

tok = AutoTokenizer.from_pretrained("$BASE_MODEL", trust_remote_code=True)

merged.save_pretrained("$MERGED_OUT", safe_serialization=True, max_shard_size="5GB")
tok.save_pretrained("$MERGED_OUT")
print("Merged model written to: $MERGED_OUT")
PYEOF

ch "phase 2 done (merged at $MERGED_OUT)"
log "Phase 2 complete."

# ------------------------------------------------------------------
# Phase 3: heretic abliteration
# ------------------------------------------------------------------
log "Phase 3: heretic abliteration"
ch "phase 3 start (heretic)"

python3 scripts/heretic_auto.py \
    --save-dir "$HERETIC_OUT" \
    --merge-strategy merge \
    --quantization BNB_4BIT \
    --fail-on-no-trials \
    -- \
    --model "$MERGED_OUT" \
    --n-trials 20 \
    --trust-remote-code true

ch "phase 3 done (abliterated at $HERETIC_OUT)"
log "Phase 3 complete."

# ------------------------------------------------------------------
# Phase 4: GGUF export (Q4_K_M)
# ------------------------------------------------------------------
log "Phase 4: GGUF export (Q4_K_M)"
ch "phase 4 start (gguf)"

python3 - <<PYEOF
from mud_puppy.gguf_export import export_to_gguf, ExportConfig

export_cfg = ExportConfig(
    source_dir="$HERETIC_OUT",
    out_path="$GGUF_OUT/gpt-oss-20b-coder-abliterated.gguf",
    quant="Q4_K_M",
    optimize_with_kernel_anvil=False,
)
result = export_to_gguf(export_cfg)
print(f"GGUF: {result.gguf_path}")
for step in result.steps:
    print(f"  - {step}")
if result.serve_command:
    print(f"Serve: {result.serve_command}")
PYEOF

ch "phase 4 done (gguf at $GGUF_OUT)"
log "Phase 4 complete."

log "===================================================================="
log "ALL PHASES COMPLETE"
log "  LoRA adapter:     $SFT_OUT"
log "  Merged model:     $MERGED_OUT"
log "  Abliterated:      $HERETIC_OUT"
log "  GGUF (deploy):    $GGUF_OUT"
log "===================================================================="
ch "ALL phases complete"
