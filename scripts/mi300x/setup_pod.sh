#!/usr/bin/env bash
# Prepares a DigitalOcean MI300X training pod for the agent-router reviewer
# and coder training runs. Run this on the pod AFTER Gary has SSHed in.
#
# Idempotent. Safe to re-run.

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch}"
WORK_DIR="${SCRATCH}/work"
MODELS_DIR="${SCRATCH}/models"
DATA_DIR="${SCRATCH}/training_data_sets"
REPO_CACHE="${SCRATCH}/coder_repos"
VENV_DIR="${WORK_DIR}/venv"

log() { printf '[setup_pod %(%H:%M:%S)T] %s\n' -1 "$*"; }

log "Detecting ROCm version and GPU count"
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname 2>&1 | grep -E "GPU|Card Series" | head -10 || true
else
    log "rocm-smi not found; continuing anyway"
fi

if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>&1 | grep -E "Name:.*gfx" | head -4 || true
fi

log "Installing system packages"
if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update -y
    sudo apt-get install -y \
        bubblewrap git build-essential cmake ninja-build \
        python3.12 python3.12-venv python3-pip \
        tmux rsync curl jq htop
else
    log "apt-get not found; install packages manually"
    exit 1
fi

log "Creating scratch layout at ${SCRATCH}"
mkdir -p "${WORK_DIR}" "${MODELS_DIR}" "${DATA_DIR}" "${REPO_CACHE}"
mkdir -p "${WORK_DIR}/outputs"

if [ ! -d "${VENV_DIR}" ]; then
    log "Creating Python venv at ${VENV_DIR}"
    python3.12 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
# Pin build tooling too -- a compromised wheel or setuptools can pwn the pod
# at install time. Bump these deliberately, not automatically.
pip install --upgrade 'pip==24.3.1' 'wheel==0.45.1' 'setuptools==75.6.0'

log "Installing torch for ROCm 7.2 (pinned)"
# Torch is pinned so a silent upstream rebuild cannot alter the training stack
# under us. If the pinned wheel is not yet mirrored for ROCm 7.2, fall back to
# the default index at the same version rather than a floating 'torch'.
TORCH_PIN="torch==2.5.1"
pip install "${TORCH_PIN}" --index-url https://download.pytorch.org/whl/rocm7.2/ || \
    pip install "${TORCH_PIN}"

log "Installing training deps from pinned requirements-pod.txt"
# All package versions are pinned in requirements-pod.txt next to this script.
# This defends against malicious or accidentally broken PyPI releases landing
# mid-training run. Bump versions in that file, not by loosening this install.
REQ_FILE="$(dirname "$(readlink -f "$0")")/requirements-pod.txt"
if [ ! -f "${REQ_FILE}" ]; then
    log "ERROR: pinned requirements file not found at ${REQ_FILE}"
    exit 3
fi
pip install -r "${REQ_FILE}"

log "Installing mud-puppy (editable) from uploaded source"
if [ -d "${WORK_DIR}/mud-puppy" ]; then
    pip install -e "${WORK_DIR}/mud-puppy"
else
    log "ERROR: mud-puppy source not found at ${WORK_DIR}/mud-puppy"
    log "Run the upload step from the workstation first."
    exit 2
fi

log "Verifying torch + ROCm"
python3 -c "
import torch
print('torch:', torch.__version__)
print('cuda/rocm available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'  [{i}]', torch.cuda.get_device_name(i))
"

log "Pre-cloning coder repos to ${REPO_CACHE}"
declare -A REPOS=(
    [llamacpp]="https://github.com/ggml-org/llama.cpp"
    [redis]="https://github.com/redis/redis"
    [sqlite]="https://github.com/sqlite/sqlite"
    [leveldb]="https://github.com/google/leveldb"
)
for name in "${!REPOS[@]}"; do
    dest="${REPO_CACHE}/${name}"
    if [ ! -d "${dest}/.git" ]; then
        log "  cloning ${name}"
        git clone --depth 1000 "${REPOS[$name]}" "${dest}"
    else
        log "  ${name} already cloned"
    fi
done

log "Setting HF cache to ${SCRATCH}/.cache/huggingface"
mkdir -p "${SCRATCH}/.cache/huggingface"
cat > "${WORK_DIR}/env.sh" <<EOF
# Source this before running training: source ${WORK_DIR}/env.sh
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${SCRATCH}/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="${SCRATCH}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${SCRATCH}/.cache/huggingface/datasets"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VIRTUAL_ENV}/bin:\${PATH}"
EOF

log "Environment summary"
echo "  scratch:     ${SCRATCH}"
echo "  work:        ${WORK_DIR}"
echo "  models:      ${MODELS_DIR} ($(du -sh ${MODELS_DIR} 2>/dev/null | cut -f1))"
echo "  data:        ${DATA_DIR} ($(du -sh ${DATA_DIR} 2>/dev/null | cut -f1))"
echo "  repo cache:  ${REPO_CACHE} ($(du -sh ${REPO_CACHE} 2>/dev/null | cut -f1))"
echo "  venv:        ${VENV_DIR}"

log "Pod setup complete. Next: source ${WORK_DIR}/env.sh then run launch.sh"
