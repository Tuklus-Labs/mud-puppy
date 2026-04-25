#!/usr/bin/env bash
# Upload workstation-unique artifacts to the MI300X pod.
#
# By default we push ONLY things that are uniquely yours:
#   - training_data_sets/ (~4 GB, your curated data)
#   - mud-puppy source code (few MB)
#
# Base models (gpt-oss-20b MXFP4, Ministral, etc.) are downloaded directly
# from HuggingFace Hub on the pod, since DO datacenter <-> HF CDN is an
# order of magnitude faster than a residential uplink.
#
# Use --with-models if you have a LOCALLY-MODIFIED model you want to push
# instead of downloading from HF. That's rare and usually a mistake.

set -euo pipefail

POD_HOST="${POD_HOST:-}"
POD_USER="${POD_USER:-root}"
SSH_KEY="${SSH_KEY:-}"
WITH_MODELS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) POD_HOST="$2"; shift 2 ;;
        --user) POD_USER="$2"; shift 2 ;;
        --key)  SSH_KEY="$2"; shift 2 ;;
        --with-models) WITH_MODELS=1; shift ;;
        --help|-h)
            cat <<EOF
Usage: upload.sh --host <pod-ip> [--user <user>] [--key <ssh-key>] [--with-models]

Default uploads (~5 GB total, fast over residential uplink):
  ~/Projects/mud-puppy/training_data_sets/    -> /scratch/training_data_sets/
  ~/Projects/mud-puppy/ (filtered source)     -> /scratch/work/mud-puppy/

With --with-models (66+ GB, slow), additionally pushes:
  ~/Models/Ministral-3-14B-Reasoning/         -> /scratch/models/
  ~/Models/gpt-oss-20b-hf/                    -> /scratch/models/

The default path is what you want. Base models come from HuggingFace on
the pod side (bake_golden_image.sh handles that). Only use --with-models
if you have workstation-only modifications that aren't on the Hub.

On a 1 Gbps residential uplink:
  default:       ~1-2 minutes
  --with-models: ~10-15 minutes
EOF
            exit 0
            ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

if [ -z "${POD_HOST}" ]; then
    echo "ERROR: pass --host <pod-ip>"
    exit 2
fi

SSH_OPTS=("-o" "StrictHostKeyChecking=accept-new" "-o" "ServerAliveInterval=30")
if [ -n "${SSH_KEY}" ]; then
    SSH_OPTS+=("-i" "${SSH_KEY}")
fi

RSYNC_BASE=(rsync -aP --partial --inplace --compress-level=0)

log() { printf '[upload %(%H:%M:%S)T] %s\n' -1 "$*"; }

log "Pre-flight: pod reachable?"
ssh "${SSH_OPTS[@]}" "${POD_USER}@${POD_HOST}" \
    "mkdir -p /scratch/models /scratch/training_data_sets /scratch/work"

log "Uploading training_data_sets (~4 GB)"
"${RSYNC_BASE[@]}" -e "ssh ${SSH_OPTS[*]}" \
    --exclude "_work/raw_downloads/" \
    --exclude "_work/comment_generation.zip" \
    --exclude "_work/repos/" \
    /home/aegis/Projects/mud-puppy/training_data_sets/ \
    "${POD_USER}@${POD_HOST}:/scratch/training_data_sets/"

log "Uploading mud-puppy source (filtered, small)"
"${RSYNC_BASE[@]}" -e "ssh ${SSH_OPTS[*]}" \
    --exclude "outputs/" \
    --exclude ".git/" \
    --exclude "__pycache__/" \
    --exclude "*.egg-info/" \
    --exclude "venv/" \
    --exclude ".venv/" \
    --exclude "runs/" \
    --exclude "wandb/" \
    --exclude "training_data_sets/" \
    /home/aegis/Projects/mud-puppy/ \
    "${POD_USER}@${POD_HOST}:/scratch/work/mud-puppy/"

if [ "${WITH_MODELS}" = "1" ]; then
    log "--with-models: uploading Ministral-3-14B-Reasoning (~26 GB)"
    "${RSYNC_BASE[@]}" -e "ssh ${SSH_OPTS[*]}" \
        /home/aegis/Models/Ministral-3-14B-Reasoning/ \
        "${POD_USER}@${POD_HOST}:/scratch/models/Ministral-3-14B-Reasoning/"

    log "--with-models: uploading gpt-oss-20b-hf (~40 GB, bf16)"
    "${RSYNC_BASE[@]}" -e "ssh ${SSH_OPTS[*]}" \
        /home/aegis/Models/gpt-oss-20b-hf/ \
        "${POD_USER}@${POD_HOST}:/scratch/models/gpt-oss-20b-hf/"
fi

log "Symlinking data + models into work tree on pod"
ssh "${SSH_OPTS[@]}" "${POD_USER}@${POD_HOST}" "
    ln -sfn /scratch/training_data_sets /scratch/work/mud-puppy/training_data_sets
    ln -sfn /scratch/models /scratch/work/models
"

log "Upload complete."
if [ "${WITH_MODELS}" = "0" ]; then
    log "Base models will be downloaded from HuggingFace during the bake."
    log "Next: ssh in and run bake_golden_image.sh (first time) or launch.sh (subsequent runs)."
fi
