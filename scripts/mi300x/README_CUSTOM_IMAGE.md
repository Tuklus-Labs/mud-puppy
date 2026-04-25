# Custom Image workflow for MI300X training

One-time setup pays ~30 min of boot-time setup back on every training run.

## Cost

- Snapshot / custom image storage: **$0.06 / GiB-month**
- Expected image size: **30-50 GB** (OS + venv + models + repos)
- Monthly: **$2-3**
- No additional cost when a droplet is running from the image; you pay
  the normal $1.99/GPU/hr.

## One-time bake (requires 1 droplet of your quota)

### 1. Create the golden droplet

DO web UI: Create Droplet.
- Image: "ROCm Software 7.2" (marketplace)
- Plan: MI300X x1 or x8 (x1 is enough for baking; x8 wastes money on idle GPUs during setup)
- SSH keys: yours
- Region: wherever your future training will run (atl1 recommended)

Note the IPv4.

### 2. Upload training data + mud-puppy source

From your workstation:
```bash
export POD_HOST=<ip>
bash /home/aegis/Projects/mud-puppy/scripts/mi300x/upload.sh --host $POD_HOST
```

~8-10 min. Sends:
- training_data_sets/ (~4 GB)
- mud-puppy/ source tree
- NOT the base models -- those download on the pod from HF Hub

### 3. Run the bake

SSH in:
```bash
ssh root@$POD_HOST
bash /scratch/work/mud-puppy/scripts/mi300x/bake_golden_image.sh
```

Takes ~20-40 min:
- apt installs + venv + torch-ROCm (~5 min)
- Training deps + heretic (~2 min)
- Model downloads from HF Hub (~5-15 min depending on region)
- Repo pre-clones (~2 min)
- Heretic smoke test on TinyLlama (~3 min)
- Cleanup of machine-specific state (<1 min)

Verify the bake succeeded -- script prints "BAKE COMPLETE" and a summary.

### 4. Snapshot the droplet

In DO web UI:
- Droplets -> <your golden droplet> -> More -> Power Off
  (wait for Off state, ~30 s)
- Same menu -> Take Snapshot
  - Name: `mud-puppy-mi300x-<date>` (e.g., `mud-puppy-mi300x-20260423`)
  - Expect 5-15 min for the snapshot to complete

### 5. (Optional) Promote snapshot to Custom Image

Backups & Snapshots -> Snapshots tab -> find your snapshot -> More -> "Move to Custom Images"
(keeps it in a dedicated Custom Images list, no functional difference)

### 6. Destroy the golden droplet

Once the snapshot/image is "Available", destroy the golden droplet from
the UI. Stops billing on it. The image persists until you manually delete it.

## Every subsequent training run

### 1. Create droplet from the image

DO web UI: Create Droplet.
- Image tab: "Snapshots" or "Custom Images"
- Select your `mud-puppy-mi300x-<date>`
- Plan: MI300X x8 (what you actually need)
- SSH keys: yours
- Region: same as before

### 2. Upload ONLY the deltas

From workstation:
```bash
export POD_HOST=<new_ip>
# Small rsync: only the changes since the last bake
rsync -aP --exclude outputs/ --exclude .git/ --exclude __pycache__/ \
    /home/aegis/Projects/mud-puppy/ \
    root@$POD_HOST:/scratch/work/mud-puppy-updates/
```

Seconds to a minute. Only sends what changed.

### 3. Fire training

SSH in, go:
```bash
ssh root@$POD_HOST
bash /scratch/work/mud-puppy/scripts/mi300x/launch_from_image.sh
# (or launch.sh directly if no code changes since bake)
```

Training starts in ~1-2 minutes. No apt, no model downloads, no pip.

### 4. Monitor + exfil as usual

```bash
# From workstation
bash scripts/mi300x/monitor.sh --host $POD_HOST
bash scripts/mi300x/download_checkpoints.sh --host $POD_HOST
```

### 5. Destroy droplet when done

DO UI. Stops billing. Image stays around for the next run.

## Maintenance

**When to re-bake**:
- mud-puppy dep versions change (torch, transformers, peft, trl, heretic-llm)
- You want a newer ROCm version in the base image
- You added a new base model you'll train on frequently
- You want to preserve a known-good configuration as a fallback after a bad change

**Delete old images**:
- DO web UI -> Backups & Snapshots -> Snapshots/Custom Images -> Delete
- Images you don't delete keep billing at $0.06/GiB/month forever

## What's inside the baked image

After `bake_golden_image.sh`:

```
/scratch/
├── work/
│   ├── venv/                  Python 3.12 + torch-ROCm + transformers + peft +
│   │                          trl + heretic-llm + all mud-puppy deps (pinned)
│   ├── mud-puppy/             Editable install of the framework
│   ├── training_data_sets/    symlink -> /scratch/training_data_sets (rsynced)
│   ├── models/                symlink -> /scratch/models
│   └── env.sh                 Source this before running training
├── models/
│   ├── gpt-oss-20b-mxfp4-hf/  ~13 GB, MXFP4 HF format (from openai/gpt-oss-20b)
│   └── TinyLlama-1.1B-Chat-v1.0/  ~2 GB, for smoke tests
├── coder_repos/
│   ├── llamacpp/              shallow clone, for coder GRPO sandbox
│   ├── redis/
│   ├── sqlite/
│   └── leveldb/
└── training_data_sets/        (rsynced during bake, re-rsynced per-run for fresh data)
```

## Gotchas

- **The snapshot captures the WHOLE disk**, including `/scratch`. That's
  why the image is 30-50 GB instead of just the base OS. This is by
  design: it's exactly why future droplets boot ready.
- **Machine-specific state is scrubbed** (SSH host keys, machine-id,
  cloud-init state, journal logs) so the image is portable across regions.
- **DO rebuilds SSH host keys on droplet creation** from the snapshot.
  You'll see a fresh host-key fingerprint on first SSH; that's correct.
- **Base models live at `/scratch/models/`** inside the image. Do not
  put them at `~/Models` as on the workstation; the paths differ.
- **Training data gets re-rsynced** per run because it's actively edited.
  Models don't change, so they live in the image.
