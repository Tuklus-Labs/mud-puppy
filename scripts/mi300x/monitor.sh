#!/usr/bin/env bash
# Remote monitor: SSH into the pod and print GPU util, tmux sessions,
# recent log tails, and the latest loss from each run's tensorboard.
# Runs on Gary's workstation. Refreshes every 60s. Ctrl-C to exit.

set -euo pipefail

POD_HOST="${POD_HOST:-}"
POD_USER="${POD_USER:-root}"
SSH_KEY="${SSH_KEY:-}"
INTERVAL="${INTERVAL:-60}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) POD_HOST="$2"; shift 2 ;;
        --user) POD_USER="$2"; shift 2 ;;
        --key)  SSH_KEY="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: monitor.sh --host <pod-ip> [--user <user>] [--key <ssh-key>] [--interval <sec>]"
            exit 0
            ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

if [ -z "${POD_HOST}" ]; then echo "pass --host"; exit 2; fi

SSH_OPTS=("-o" "StrictHostKeyChecking=accept-new" "-o" "ServerAliveInterval=30")
if [ -n "${SSH_KEY}" ]; then SSH_OPTS+=("-i" "${SSH_KEY}"); fi

REMOTE_SCRIPT=$(cat <<'REMOTE'
set -u
echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
echo "-- GPUs --"
rocm-smi --showuse --showmeminfo vram 2>/dev/null | grep -E "GPU use|Used Memory" | head -16 || true
echo
echo "-- tmux sessions --"
tmux ls 2>/dev/null || echo "  none"
echo
echo "-- tensorboard scalars (latest loss per run) --"
for d in /scratch/work/mud-puppy/outputs/*/runs/*; do
    [ -d "$d" ] || continue
    python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys, os
p = sys.argv[1]
ea = EventAccumulator(p); ea.Reload()
tags = ea.Tags()['scalars']
name = os.path.relpath(p, '/scratch/work/mud-puppy/outputs')
if 'train/loss' in tags:
    ls = ea.Scalars('train/loss')
    gn = ea.Scalars('train/grad_norm') if 'train/grad_norm' in tags else []
    rw = ea.Scalars('train/reward') if 'train/reward' in tags else []
    if ls:
        tail = ls[-1]
        gnv = gn[-1].value if gn else None
        rwv = rw[-1].value if rw else None
        extras = []
        if gnv is not None: extras.append(f'gn={gnv:.2e}')
        if rwv is not None: extras.append(f'rew={rwv:.3f}')
        extra_s = ' '.join(extras)
        print(f'  {name:50s}  step={tail.step:5d}  loss={tail.value:.4f}  {extra_s}')
" "$d" 2>/dev/null
done
echo
echo "-- last 8 lines of most recent run log --"
latest=$(ls -t /scratch/work/launch.log /scratch/work/mud-puppy/outputs/*/run.log 2>/dev/null | head -1)
if [ -n "$latest" ]; then
    echo "  $latest"
    tail -n 8 "$latest" | sed 's/^/    /'
fi
REMOTE
)

while true; do
    clear
    # shellcheck disable=SC2029
    ssh "${SSH_OPTS[@]}" "${POD_USER}@${POD_HOST}" "bash -s" <<< "${REMOTE_SCRIPT}" || {
        echo "SSH failed; retrying in ${INTERVAL}s"
    }
    sleep "${INTERVAL}"
done
