#!/usr/bin/env bash
# Pulls trained LoRA adapters, tokenizers, and tensorboard events back
# to Gary's workstation. Runs on the workstation.
#
# Does NOT delete anything on the pod. Pod teardown is a manual step
# in the DO web UI so Gary controls when money stops flowing.

set -euo pipefail

POD_HOST="${POD_HOST:-}"
POD_USER="${POD_USER:-root}"
SSH_KEY="${SSH_KEY:-}"
LOCAL_OUT="${LOCAL_OUT:-/home/aegis/Projects/mud-puppy/outputs/mi300x}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) POD_HOST="$2"; shift 2 ;;
        --user) POD_USER="$2"; shift 2 ;;
        --key)  SSH_KEY="$2"; shift 2 ;;
        --out)  LOCAL_OUT="$2"; shift 2 ;;
        --help|-h)
            cat <<EOF
Usage: download_checkpoints.sh --host <pod-ip> [--user <user>] [--key <ssh-key>] [--out <dir>]

Pulls from the pod:
  adapter_model.safetensors, adapter_config.json, tokenizer files
  runs/ (tensorboard event files)
  latest checkpoint-* inside each run directory

Writes to: ${LOCAL_OUT:-<default>}

The pod itself is not touched. After this completes, you can destroy
the pod from the DO web UI to stop billing.
EOF
            exit 0
            ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

if [ -z "${POD_HOST}" ]; then echo "pass --host"; exit 2; fi

SSH_OPTS=("-o" "StrictHostKeyChecking=accept-new" "-o" "ServerAliveInterval=30")
if [ -n "${SSH_KEY}" ]; then SSH_OPTS+=("-i" "${SSH_KEY}"); fi

mkdir -p "${LOCAL_OUT}"

log() { printf '[download %(%H:%M:%S)T] %s\n' -1 "$*"; }

log "Enumerating runs on pod"
RUNS=$(ssh "${SSH_OPTS[@]}" "${POD_USER}@${POD_HOST}" "ls /scratch/work/mud-puppy/outputs/ 2>/dev/null" || true)
if [ -z "${RUNS}" ]; then
    log "No runs found at /scratch/work/mud-puppy/outputs/"
    exit 0
fi

echo "${RUNS}" | while read -r run; do
    [ -z "${run}" ] && continue
    log "Pulling run: ${run}"
    mkdir -p "${LOCAL_OUT}/${run}"

    # Adapter + tokenizer files at the top level of the run.
    rsync -aP \
        -e "ssh ${SSH_OPTS[*]}" \
        --include="adapter_model.safetensors" \
        --include="adapter_config.json" \
        --include="tokenizer*" \
        --include="special_tokens_map.json" \
        --include="training_args.bin" \
        --include="README.md" \
        --include="*.json" \
        --exclude="*" \
        "${POD_USER}@${POD_HOST}:/scratch/work/mud-puppy/outputs/${run}/" \
        "${LOCAL_OUT}/${run}/" || log "  adapter pull failed for ${run}"

    # Tensorboard runs/ subtree.
    rsync -aP \
        -e "ssh ${SSH_OPTS[*]}" \
        "${POD_USER}@${POD_HOST}:/scratch/work/mud-puppy/outputs/${run}/runs/" \
        "${LOCAL_OUT}/${run}/runs/" 2>/dev/null || true

    # Latest checkpoint dir (highest-numbered checkpoint-*).
    LATEST_CKPT=$(ssh "${SSH_OPTS[@]}" "${POD_USER}@${POD_HOST}" \
        "ls -d /scratch/work/mud-puppy/outputs/${run}/checkpoint-* 2>/dev/null | sort -V | tail -1" || true)
    if [ -n "${LATEST_CKPT}" ]; then
        CKPT_NAME=$(basename "${LATEST_CKPT}")
        log "  latest checkpoint: ${CKPT_NAME}"
        rsync -aP \
            -e "ssh ${SSH_OPTS[*]}" \
            "${POD_USER}@${POD_HOST}:${LATEST_CKPT}/" \
            "${LOCAL_OUT}/${run}/${CKPT_NAME}/" || log "  checkpoint pull failed"
    fi
done

log "Done. Artifacts under ${LOCAL_OUT}"
log "Pod is still alive. Destroy it from the DO web UI when you are satisfied."
