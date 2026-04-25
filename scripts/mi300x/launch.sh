#!/usr/bin/env bash
# Launches training jobs on the MI300X pod, using FSDP full_shard across
# all 8 GPUs for each run. Runs are SERIALIZED (one at a time): each
# training takes the whole cluster for speed, then hands off to the next.
#
# Sequence:
#   1. reviewer SFT    (FSDP x8, ~30-45 min)
#   2. reviewer GRPO   (FSDP x8, ~1-2h)
#   3. coder SFT       (FSDP x8, ~45-90 min)
#   4. coder GRPO      (FSDP x8, ~1.5-2.5h, sandbox-bound)
#
# Runs inside a single tmux session so Gary can detach and reattach.
#
# Usage:
#   bash launch.sh [--dry-run] [--skip-reviewer] [--skip-coder]

set -euo pipefail

DRY_RUN=0
SKIP_REV=0
SKIP_COD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --skip-reviewer) SKIP_REV=1; shift ;;
        --skip-coder) SKIP_COD=1; shift ;;
        -h|--help)
            sed -n '2,15p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

SCRATCH="${SCRATCH:-/scratch}"
WORK_DIR="${SCRATCH}/work"
MP_DIR="${WORK_DIR}/mud-puppy"
OUT_DIR="${MP_DIR}/outputs"
LAUNCH_LOG="${WORK_DIR}/launch.log"
NPROC="${NPROC:-8}"

mkdir -p "${OUT_DIR}"

log() { printf '[launch %(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "${LAUNCH_LOG}"; }

# shellcheck disable=SC1091
if [ -f "${WORK_DIR}/env.sh" ]; then source "${WORK_DIR}/env.sh"; fi

REV_SFT_OUT="${OUT_DIR}/reviewer-sft-mi300x"
REV_GRPO_OUT="${OUT_DIR}/reviewer-grpo-mi300x"
COD_SFT_OUT="${OUT_DIR}/coder-sft-mi300x"
COD_GRPO_OUT="${OUT_DIR}/coder-grpo-mi300x"

# Build the serial command chain.
CHAIN=()

if [ "${SKIP_REV}" = "0" ]; then
    CHAIN+=(
        "torchrun --nproc_per_node=${NPROC} scripts/reviewer_sft.py \
            --base /scratch/models/Ministral-3-14B-Reasoning \
            --data training_data_sets/reviewer/codereviewer-sft-warmup.messages.jsonl \
            --out ${REV_SFT_OUT} \
            --fsdp full_shard --fsdp-wrap-class MistralDecoderLayer"
        "torchrun --nproc_per_node=${NPROC} scripts/reviewer_grpo.py \
            --sft-checkpoint ${REV_SFT_OUT} \
            --data training_data_sets/reviewer/reviewer-grpo.jsonl \
            --out ${REV_GRPO_OUT} \
            --fsdp full_shard --fsdp-wrap-class MistralDecoderLayer"
    )
fi

if [ "${SKIP_COD}" = "0" ]; then
    CHAIN+=(
        "torchrun --nproc_per_node=${NPROC} scripts/coder_sft.py \
            --model /scratch/models/gpt-oss-20b-hf \
            --data-glob 'training_data_sets/coder/*-commits.jsonl' \
            --output ${COD_SFT_OUT} \
            --fsdp full_shard --fsdp-wrap-class GptOssDecoderLayer"
        "torchrun --nproc_per_node=${NPROC} scripts/coder_grpo.py \
            --base-model /scratch/models/gpt-oss-20b-hf \
            --sft-checkpoint ${COD_SFT_OUT} \
            --data-glob 'training_data_sets/coder/*-commits.jsonl' \
            --repo-cache-root /scratch/coder_repos \
            --output ${COD_GRPO_OUT} \
            --fsdp full_shard --fsdp-wrap-class GptOssDecoderLayer"
    )
fi

# Join into a single shell chain (one command && next && ...).
FULL_CMD="cd ${MP_DIR} && source ${WORK_DIR}/env.sh"
for c in "${CHAIN[@]}"; do
    # Strip leading whitespace from multi-line continuation.
    clean=$(echo "$c" | tr '\n' ' ' | tr -s ' ')
    FULL_CMD="${FULL_CMD} && ${clean}"
done

log "Launch plan (dry-run=${DRY_RUN}, nproc=${NPROC})"
log "======================================"
log "Sequence (${#CHAIN[@]} jobs, serialized, FSDP x${NPROC} each):"
i=1
for c in "${CHAIN[@]}"; do
    first_line=$(echo "$c" | head -1 | tr -s ' ')
    log "  ${i}. ${first_line}"
    i=$((i+1))
done

if [ "${DRY_RUN}" = "1" ]; then
    log "Dry run. Full chain would be:"
    echo "${FULL_CMD}" | sed 's/&&/\n  &&/g' | sed 's/^/  /'
    exit 0
fi

# Launch in a single long-lived tmux session.
SESSION="training"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    log "session '${SESSION}' already exists; attach or kill before re-running"
    exit 3
fi

log "Starting tmux session '${SESSION}' with the serialized chain"
tmux new-session -d -s "${SESSION}" "${FULL_CMD}; echo 'CHAIN COMPLETE'; bash"

log "Launched. Attach with: tmux attach -t ${SESSION}"
log "Check GPU utilization:      watch -n 5 rocm-smi --showuse --showmeminfo vram"
log "Check training progress:    tail -F ${OUT_DIR}/*/run.log (or tensorboard)"
log "Log: ${LAUNCH_LOG}"
