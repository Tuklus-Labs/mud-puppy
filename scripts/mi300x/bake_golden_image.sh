#!/usr/bin/env bash
# One-shot provisioning for the golden MI300X droplet.
#
# Run this ONCE on a fresh MI300X pod (spun from DO's "ROCm Software 7.2"
# marketplace image). Everything it does becomes part of the custom image
# you'll snapshot afterward, so subsequent droplets boot ready to train.
#
# Runs setup_pod.sh + model pre-downloads + heretic install + a smoke test,
# then cleans up machine-specific state so the snapshot is stateless.

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch}"
WORK_DIR="${SCRATCH}/work"
MODELS_DIR="${SCRATCH}/models"
VENV_DIR="${WORK_DIR}/venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf '[bake %(%H:%M:%S)T] %s\n' -1 "$*"; }

# --- 1) standard pod setup (apt + venv + torch-ROCm + deps + repo preclone) ---
log "=== step 1/6: setup_pod.sh ==="
bash "${SCRIPT_DIR}/setup_pod.sh"

# shellcheck disable=SC1091
source "${WORK_DIR}/env.sh"

# --- 2) install heretic and its deps into mud-puppy's env ---
log "=== step 2/6: heretic install ==="
pip install --no-deps heretic-llm==1.2.0
pip install 'optuna>=4.5' 'questionary>=2.1' 'psutil>=7.1' \
            'pydantic-settings>=2.10' 'hf-transfer>=0.1' 'kernels>=0.11' \
            'rich>=14.1,<16'

# Verify heretic imports under our stub path
python3 -c "
import sys, types, importlib.machinery
bnb = types.ModuleType('bitsandbytes')
fn = types.ModuleType('bitsandbytes.functional')
fn.dequantize_4bit = lambda *a, **kw: None
bnb.functional = fn
nn_mod = types.ModuleType('bitsandbytes.nn')
nn_mod.Linear4bit = object
bnb.nn = nn_mod
for m in (bnb, fn, nn_mod):
    m.__spec__ = importlib.machinery.ModuleSpec(name=m.__name__ if hasattr(m,'__name__') else 'bitsandbytes', loader=None)
sys.modules['bitsandbytes'] = bnb
sys.modules['bitsandbytes.functional'] = fn
sys.modules['bitsandbytes.nn'] = nn_mod
import heretic.model, heretic.main
print('heretic imports OK')
"

# --- 3) pre-download base models from HF Hub (once, baked into image) ---
log "=== step 3/6: pre-download base models ==="
mkdir -p "${MODELS_DIR}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# gpt-oss-20b in native MXFP4 HF format (13GB)
if [ ! -d "${MODELS_DIR}/gpt-oss-20b-mxfp4-hf" ]; then
    log "downloading openai/gpt-oss-20b (~13GB, MXFP4 native)"
    huggingface-cli download openai/gpt-oss-20b \
        --local-dir "${MODELS_DIR}/gpt-oss-20b-mxfp4-hf" \
        --exclude "*.md" "*.png"
fi

# Ministral-3-14B-Reasoning base (14B bf16, ~28GB) -- optional; skip if not needed
# Uncomment if you want to bake the reviewer base too. Costs ~$1.70/month in
# image storage.
# if [ ! -d "${MODELS_DIR}/Ministral-3-14B-Instruct" ]; then
#     log "downloading mistralai/Ministral-3-14B-Instruct-2512 (~28GB)"
#     huggingface-cli download mistralai/Ministral-3-14B-Instruct-2512 \
#         --local-dir "${MODELS_DIR}/Ministral-3-14B-Instruct"
# fi

# TinyLlama for fast smoke tests (~2GB, useful for image sanity)
if [ ! -d "${MODELS_DIR}/TinyLlama-1.1B-Chat-v1.0" ]; then
    log "downloading TinyLlama for smoke tests (~2GB)"
    huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --local-dir "${MODELS_DIR}/TinyLlama-1.1B-Chat-v1.0" \
        --exclude "*.bin" "*.md" "*.png"  # prefer safetensors
fi

# --- 4) smoke test: heretic runs end-to-end on TinyLlama ---
log "=== step 4/6: heretic smoke test (TinyLlama, 2 trials) ==="
rm -rf /tmp/bake-smoke-out
cd "${WORK_DIR}/mud-puppy"
timeout 900 python3 scripts/heretic_auto.py \
    --save-dir /tmp/bake-smoke-out \
    --quantization NONE \
    -- \
    --model "${MODELS_DIR}/TinyLlama-1.1B-Chat-v1.0" \
    --n-trials 2 \
    --trust-remote-code true \
    2>&1 | tail -20
if [ -f /tmp/bake-smoke-out/config.json ]; then
    log "smoke test PASSED"
else
    log "smoke test FAILED -- abort bake"
    exit 2
fi
rm -rf /tmp/bake-smoke-out

# --- 5) clean up machine-specific state so the snapshot is stateless ---
log "=== step 5/6: strip machine-specific state ==="

# Python bytecode (regenerated on first run)
find "${WORK_DIR}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# HF download temporary junk
rm -rf "${SCRATCH}/.cache/huggingface/hub"/*/blobs.* 2>/dev/null || true

# apt caches
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

# journal logs (DO image shouldn't have logs from the bake session)
sudo journalctl --vacuum-time=1s

# Shell histories
cat /dev/null > ~/.bash_history 2>/dev/null || true
cat /dev/null > /root/.bash_history 2>/dev/null || true
history -c 2>/dev/null || true

# SSH host keys (DO regenerates on new droplet creation)
sudo rm -f /etc/ssh/ssh_host_* 2>/dev/null || true

# Machine ID (regenerated on boot)
sudo truncate -s 0 /etc/machine-id 2>/dev/null || true

# Cloud-init state (so new droplets run first-boot steps)
sudo rm -rf /var/lib/cloud/* 2>/dev/null || true

# --- 6) summary ---
log "=== step 6/6: image contents summary ==="
echo "  scratch:   ${SCRATCH}  ($(du -sh ${SCRATCH} 2>/dev/null | cut -f1))"
echo "  models:    ${MODELS_DIR}  ($(du -sh ${MODELS_DIR} 2>/dev/null | cut -f1))"
echo "  work:      ${WORK_DIR}  ($(du -sh ${WORK_DIR} 2>/dev/null | cut -f1))"
echo "  venv:      ${VENV_DIR}  ($(du -sh ${VENV_DIR} 2>/dev/null | cut -f1))"

log "BAKE COMPLETE"
log ""
log "Next steps on your workstation:"
log "  1. In DO web UI: Droplets -> <this droplet> -> Power Off"
log "  2. Droplets -> <this droplet> -> Take Snapshot (name it 'mud-puppy-mi300x-<date>')"
log "  3. (Optional) Backups & Snapshots -> Snapshots -> move to Custom Images"
log "  4. Destroy this droplet once the snapshot shows 'Available'"
log "  5. Future training runs: Create Droplet -> Custom Images -> <your snapshot>"
log ""
log "Estimated image size: ~30-50 GB. Storage: $0.06/GiB/month."
