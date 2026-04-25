# MI300X cloud training workflow

End-to-end runbook for training reviewer + coder + ablations on a rented
DigitalOcean MI300X pod. Scripts live in this directory. Pod lifecycle
(create, destroy) is a manual step so Gary controls when money starts
and stops flowing.

## Cost estimate

- 8 GPU pod: $1.99/GPU/hr × 8 = $15.92/hr
- Expected wall-clock: 12-18 hours (bounded by coder SFT + GRPO)
- Expected total: $190-290

Monitor with `monitor.sh` and destroy the pod when you are satisfied
with the downloaded artifacts.

## Steps

### 0. Create the pod (manual, via DO web UI)

- Plan: MI300X x8 (1.5 TB VRAM, 1920 GB RAM, 40 TB scratch)
- Image: ROCm 7.2 on Ubuntu, or bare Ubuntu 24.04 with ROCm to install
- SSH key: attach yours
- Region: wherever gets you the lowest RTT, probably NYC3 or SFO3

Note the pod's public IPv4. Set it in env:
```
export POD_HOST=<ip>
# optional
export POD_USER=root
export SSH_KEY=~/.ssh/id_ed25519
```

### 1. Upload from the workstation

```
bash /home/aegis/Projects/mud-puppy/scripts/mi300x/upload.sh --host $POD_HOST
```

Uploads ~52 GB (models + training data + mud-puppy source). On a 1 Gbps
uplink this is roughly 8-10 minutes. Safe to re-run; rsync resumes.

### 2. Set up the pod

SSH in:
```
ssh -i $SSH_KEY $POD_USER@$POD_HOST
```

On the pod:
```
bash /scratch/work/mud-puppy/scripts/mi300x/setup_pod.sh
```

Installs system packages, creates Python 3.12 venv, installs torch-ROCm
and training deps, pre-clones the four coder repos. Idempotent. ~5-10
minutes first run.

### 3. Launch training

Still on the pod:
```
cd /scratch/work/mud-puppy
bash scripts/mi300x/launch.sh --dry-run    # review what will run
bash scripts/mi300x/launch.sh              # actually launch
```

Creates one tmux session per GPU job. GPU 0 runs reviewer SFT then GRPO.
GPU 1 runs coder SFT then GRPO. GPUs 2-3 run ablation stubs. You can
`tmux attach -t rev-pipeline` to watch any session live. Detach with
`Ctrl-b d`.

After launching, close the SSH session. The tmux sessions keep running.

### 4. Monitor from the workstation

```
bash /home/aegis/Projects/mud-puppy/scripts/mi300x/monitor.sh --host $POD_HOST
```

Refreshes every 60 seconds. Shows GPU utilization, tmux session status,
latest loss per run from tensorboard, and log tails. Ctrl-C to exit; does
not affect the pod.

### 5. Download checkpoints

When you see the runs reporting `Training complete!` in the logs, pull
artifacts back:

```
bash /home/aegis/Projects/mud-puppy/scripts/mi300x/download_checkpoints.sh --host $POD_HOST
```

Lands under `/home/aegis/Projects/mud-puppy/outputs/mi300x/<run-name>/`.

### 6. Destroy the pod (manual, via DO web UI)

**Only after you are satisfied with the downloaded artifacts.** Once
destroyed, the 40 TB scratch disk and everything on it are gone.

## What each script does

| Script | Runs on | Purpose |
|--------|---------|---------|
| `setup_pod.sh` | pod | apt install, venv, torch-ROCm, mud-puppy, repo pre-clones |
| `upload.sh` | workstation | rsync models + data + mud-puppy source to pod |
| `launch.sh` | pod | start all training jobs in tmux, pinned to GPUs |
| `monitor.sh` | workstation | SSH-based liveness + loss dashboard |
| `download_checkpoints.sh` | workstation | rsync LoRA adapters + tensorboard events back |

## Expected run outputs on the pod

```
/scratch/work/mud-puppy/outputs/
├── reviewer-sft-mi300x/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── runs/<timestamp>/events.out.tfevents.*
│   └── checkpoint-*/
├── reviewer-grpo-mi300x/
│   ├── adapter_model.safetensors
│   └── ...
├── coder-sft-mi300x/
│   └── ...
└── coder-grpo-mi300x/
    └── ...
```

## Troubleshooting

**`torch.cuda.is_available()` returns False on the pod**: ROCm is not
fully installed. Try `apt install rocm-hip-libraries` and re-run setup.
If stuck, the pod may need a different image.

**llama.cpp / redis / sqlite / leveldb tests take too long**: the
coder GRPO reward uses the tests. Every generation runs them. If the
test wall exceeds ~60s it will hurt GRPO throughput. Override per-repo
test commands in `mud_puppy/coder_sandbox.py` to use a faster subset.

**Out of scratch disk**: ~80 GB of checkpoints can land depending on
LoRA rank and save frequency. `df -h /scratch` on the pod; trim old
`checkpoint-*` dirs.

**Upload resume after interruption**: rsync is resumable. Just re-run
`upload.sh` with the same args.
