#!/usr/bin/env bash
# Fast-path launch for a droplet booted from the baked custom image.
#
# The image already has: python venv, mud-puppy, torch-ROCm, heretic,
# pre-downloaded base models, pre-cloned coder repos, pinned deps.
#
# This script only does the DELTA per run: rsync the latest training data
# + any mud-puppy source changes from the workstation, then fire training.
#
# Runs on the pod after SSH-ing in. Training-data upload happens OUT OF BAND
# via scripts/mi300x/upload.sh from the workstation beforehand.

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch}"
WORK_DIR="${SCRATCH}/work"
MP_DIR="${WORK_DIR}/mud-puppy"

# shellcheck disable=SC1091
source "${WORK_DIR}/env.sh"

log() { printf '[launch-img %(%H:%M:%S)T] %s\n' -1 "$*"; }

# Verify the baked image actually has what we need (sanity on first boot).
log "verifying image pre-bakes"
for probe in \
    "${MP_DIR}/scripts/heretic_auto.py" \
    "${WORK_DIR}/venv/bin/python" \
    "${SCRATCH}/models/gpt-oss-20b-mxfp4-hf/config.json" \
    "${SCRATCH}/coder_repos/llamacpp/.git" \
    ; do
    if [ ! -e "${probe}" ]; then
        log "ERROR: baked image missing ${probe}"
        log "This droplet may not be from the golden image. Run setup_pod.sh first."
        exit 2
    fi
done
log "baked image OK"

# Pick up any code changes the user rsynced since the bake
if [ -d "${WORK_DIR}/mud-puppy-updates" ]; then
    log "applying delta updates from ${WORK_DIR}/mud-puppy-updates"
    rsync -aP --delete \
        --exclude "outputs/" \
        --exclude ".git/" \
        --exclude "__pycache__/" \
        "${WORK_DIR}/mud-puppy-updates/" "${MP_DIR}/"
    cd "${MP_DIR}"
    pip install -e . --no-deps
fi

# Hand off to the real launch script with all 8 GPUs via FSDP.
log "handing off to launch.sh"
exec bash "${MP_DIR}/scripts/mi300x/launch.sh" "$@"
