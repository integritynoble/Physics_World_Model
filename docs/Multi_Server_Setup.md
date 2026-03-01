# Multi-Server Benchmark Setup Guide

This guide explains how to set up new servers to run PWM benchmark experiments in parallel across different imaging modalities, with GPU algorithms running on [Modal](https://modal.com/).

## Architecture Overview

```
       ┌────────────────────────────┐     ┌────────────────────┐
       │  GCS Bucket                │     │  Modal Volume      │
       │  pwm-benchmark-datasets    │     │  pwm-models        │
       │                            │     │                    │
       │  datasets/sd_cassi/   2.6G │     │  checkpoint/       │
       │  datasets/cacti/      1.6G │     │    ELP-Unfolding/  │
       │  datasets/spc_kronecker/   │     │    MST-HDNet/      │
       │  challenge-data/v1.0/ 5.2G │     │    DRUNet/         │
       └──────────┬─────────────────┘     │    EfficientSCI/   │
                  │                       │    HATNet-SPI/      │
     ┌────────────┼────────────┐          │    ... (17 GB)      │
     ▼            ▼            ▼          └─────────┬──────────┘
┌─────────┐ ┌─────────┐ ┌─────────┐                │
│Server A │ │Server B │ │Server C │    ┌────────────┼────────────┐
│ CACTI   │ │SD-CASSI │ │  SPC    │    ▼            ▼            ▼
└────┬────┘ └────┬────┘ └────┬────┘  ┌───┐       ┌───┐       ┌───┐
     │           │           │       │GPU│       │GPU│       │GPU│
     └───────────┼───────────┘       │A10│       │A10│       │A10│
                 │                   └───┘       └───┘       └───┘
           modal run ...              Modal (on-demand GPUs)
```

**Data flow:**
- **Benchmark datasets** (HDF5, images): stored on GCS, downloaded to each server via `gsutil`
- **Model checkpoints** (17 GB): stored on Modal volume `pwm-models`, mounted instantly by GPU functions — no download needed
- **GPU compute**: on-demand via Modal (A10, A100, H100) — no GPU server management

---

## Step 1: Provision the Server

On a new GCP VM (or any Linux server):

```bash
gcloud compute ssh <instance-name> --zone <zone>
```

## Step 2: Install Prerequisites

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git

# Install Google Cloud SDK (pre-installed on GCP VMs, skip if already available)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

## Step 3: Authenticate with GCS

**Option A — Interactive login (personal servers):**

```bash
gcloud auth login
```

**Option B — Service account (automated/headless servers, recommended):**

1. In GCP Console: IAM & Admin → Service Accounts → Create Key → JSON
2. Copy the key file to the server
3. Activate:

```bash
gcloud auth activate-service-account --key-file=/path/to/service-account-key.json
```

**Verify access:**

```bash
gsutil ls gs://pwm-benchmark-datasets/
```

## Step 4: Clone the Repo

```bash
cd ~
git clone git@github.com:integritynoble/Physics_World_Model.git
cd Physics_World_Model
```

## Step 5: Set Up Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e packages/pwm_core
pip install numpy scipy h5py matplotlib
```

## Step 6: Download Benchmark Data

```bash
# See what's available and sizes
./scripts/setup_benchmark_data.sh --list

# Download ONLY the dataset you need for your modality
./scripts/setup_benchmark_data.sh cacti            # Server A: CACTI
./scripts/setup_benchmark_data.sh sd_cassi         # Server B: SD-CASSI
./scripts/setup_benchmark_data.sh spc_kronecker    # Server C: SPC Kronecker

# Or download just the challenge HDF5 files (smaller, no preview images)
./scripts/setup_benchmark_data.sh --challenge cacti
```

## Step 7: Set Up Modal (for GPU algorithms)

All GPU algorithms (ELP-Unfolding, MST, EfficientSCI, etc.) run on Modal. Model checkpoints (17 GB) are stored on a shared Modal volume — no need to download them to your server.

```bash
# Install Modal
pip install modal

# Authenticate with Modal (one-time, uses same account as main server)
modal setup
```

**Verify checkpoints are available:**

```bash
# List checkpoints on the shared Modal volume
modal volume ls pwm-models /checkpoint/

# Run verification on a Modal GPU (loads every checkpoint)
modal run scripts/verify_modal_checkpoints.py
```

### Available Checkpoints on Modal Volume

All checkpoints are pre-uploaded to the `pwm-models` volume and shared across all servers using the same Modal account.

| Checkpoint | Size | Algorithm |
|-----------|------|-----------|
| `ELP-Unfolding` | 13 GB | SD-CASSI / CACTI deep unfolding |
| `HATNet-SPI` | 3.7 GB | SPC hybrid attention transformer |
| `DRUNet` | 250 MB | PnP denoiser (all modalities) |
| `EfficientSCI` | 34 MB | CACTI snapshot compressive imaging |
| `MST-HDNet` | 35 MB | SD-CASSI spectral transformer |
| `ProxUnroll` | 38 MB | General proximal unrolling |
| `PnP-SCI` | 42 MB | CACTI PnP reconstruction |
| `ISTA-Net` | 21 MB | Compressive sensing |
| `PnP-CASSI` | 7.5 MB | SD-CASSI PnP reconstruction |
| `DnCNN` | 2.2 MB | Gaussian denoiser prior |

**Cost:** Modal volume storage is free. You only pay for GPU seconds when running functions.

## Step 8: Run Experiments

**CPU-only algorithms (run locally):**

```bash
python3 scripts/run_cacti_experiment.py          # Server A
python3 scripts/run_cassi_experiment.py          # Server B
python3 scripts/run_spc_experiment.py            # Server C
```

**GPU algorithms (run on Modal):**

```bash
modal run scripts/modal_run_elp.py               # ELP-Unfolding on GPU
```

---

## GPU Algorithms on Modal

### How It Works

Modal functions mount the `pwm-models` volume at `/models/`. Checkpoints are available instantly at `/models/checkpoint/` — no download, no waiting.

```python
import modal

app = modal.App("pwm-benchmark")
vol = modal.Volume.from_name("pwm-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "numpy", "scipy", "h5py")
)

@app.function(
    image=image,
    gpu="A100",
    volumes={"/models": vol},
    timeout=3600,
)
def run_elp_reconstruction(challenge_data: bytes):
    import torch, h5py, io

    # Checkpoints mounted instantly — no download needed
    weights = "/models/checkpoint/ELP-Unfolding/ckptall.pth"
    ckpt = torch.load(weights, map_location="cuda", weights_only=False)

    # Load challenge data and reconstruct
    f = h5py.File(io.BytesIO(challenge_data), "r")
    results = {}
    for key in sorted(f.keys()):
        y = torch.tensor(f[key]["y"][()]).cuda()
        H = torch.tensor(f[key]["H_ideal"][()]).cuda()
        # ... run reconstruction ...
        results[key] = y.cpu().numpy()  # placeholder
    return results

@app.local_entrypoint()
def main():
    with open("datasets/benchmark/sd_cassi/public/sd_cassi_challenge_public.h5", "rb") as f:
        data = f.read()
    results = run_elp_reconstruction.remote(data)
    print(f"Reconstructed {len(results)} scenes")
```

### Run Multiple Modalities in Parallel

```python
@app.local_entrypoint()
def main():
    variants = ["sd_cassi", "cacti", "spc_kronecker"]
    futures = []
    for v in variants:
        path = f"datasets/benchmark/{v}/public/{v}_challenge_public.h5"
        with open(path, "rb") as f:
            futures.append(run_elp_reconstruction.remote(f.read()))
    # All three run in parallel on separate GPUs
    for v, result in zip(variants, futures):
        print(f"{v}: {len(result)} scenes")
```

### Algorithm → Checkpoint Path Mapping

When writing Modal functions, use these paths (mounted at `/models/`):

| Algorithm | Checkpoint Path |
|-----------|----------------|
| ELP-Unfolding | `/models/checkpoint/ELP-Unfolding/ckptall.pth` |
| ELP-Unfolding (small) | `/models/checkpoint/ELP-Unfolding/ckptallS.pth` |
| MST-S | `/models/checkpoint/MST-HDNet/mst/mst_s.pth` |
| MST-L | `/models/checkpoint/MST-HDNet/mst/mst_l.pth` |
| HDNet | `/models/checkpoint/MST-HDNet/hdnet/hdnet.pth` |
| DRUNet (color) | `/models/checkpoint/DRUNet/drunet_deepinv_color_finetune_22k.pth` |
| DRUNet (gray) | `/models/checkpoint/DRUNet/drunet_deepinv_gray_finetune_26k.pth` |
| EfficientSCI | `/models/checkpoint/EfficientSCI/efficientsci_base.pth` |
| HATNet-SPI (cr=0.25) | `/models/checkpoint/HATNet-SPI/2024_pretraiend_weights/cr_0.25.pth` |
| PnP-CASSI | `/models/checkpoint/PnP-CASSI/deep_denoiser.pth` |
| PnP-SCI (FFDNet) | `/models/checkpoint/PnP-SCI/ffdnet/net_gray.pth` |
| PnP-SCI (FastDVDnet) | `/models/checkpoint/PnP-SCI/fastdvdnet/model.pth` |
| DnCNN | `/models/checkpoint/DnCNN/dncnn_25.pth` |
| ProxUnroll (ADMM) | `/models/checkpoint/ProxUnroll/admm_proxunroll.pth` |
| ProxUnroll (HQS) | `/models/checkpoint/ProxUnroll/hqs_proxunroll.pth` |
| ISTA-Net+ (ratio=25) | `/models/checkpoint/ISTA-Net/CS_ISTA_Net_plus_layer_9_group_1_ratio_25_lr_0.0001/net_params_200.pkl` |

---

## Managing Checkpoints (Main Server Only)

The main server is the source of truth for checkpoints. To add or update checkpoints:

```bash
# Upload a new checkpoint from main server to Modal volume
modal volume put pwm-models /path/to/local/checkpoint /checkpoint/NewModel/

# List what's on the volume
modal volume ls pwm-models /checkpoint/

# Verify all checkpoints load on GPU
modal run scripts/verify_modal_checkpoints.py
```

All other servers using the same Modal account see the updated volume immediately.

---

## GCS Data Structure

| Path | Size | Content |
|------|------|---------|
| `datasets/sd_cassi/` | 2.6 GB | SD-CASSI source data (all tiers + images) |
| `datasets/cacti/` | 1.6 GB | CACTI source data (all tiers + images) |
| `datasets/spc_kronecker/` | 151 MB | SPC Kronecker source data |
| `challenge-data/v1.0/` | 5.2 GB | All 507 challenge HDF5 files |

Model checkpoints are **NOT** on GCS. They are on Modal volume `pwm-models` (17 GB, free storage).

---

## Parallel Deployment Example

| Server | Modality | Data Setup | CPU Experiment | GPU Experiment |
|--------|----------|------------|----------------|----------------|
| A | CACTI | `./scripts/setup_benchmark_data.sh cacti` | `python3 scripts/run_cacti_experiment.py` | `modal run scripts/modal_run_elp.py` |
| B | SD-CASSI | `./scripts/setup_benchmark_data.sh sd_cassi` | `python3 scripts/run_cassi_experiment.py` | `modal run scripts/modal_run_elp.py` |
| C | SPC | `./scripts/setup_benchmark_data.sh spc_kronecker` | `python3 scripts/run_spc_experiment.py` | `modal run scripts/modal_run_elp.py` |

---

## Quick One-Liner Setup

```bash
git clone git@github.com:integritynoble/Physics_World_Model.git && \
cd Physics_World_Model && \
python3 -m venv .venv && source .venv/bin/activate && \
pip install -e packages/pwm_core modal && \
modal setup && \
./scripts/setup_benchmark_data.sh cacti && \
python3 scripts/run_cacti_experiment.py
```

---

## Upload Results Back

After experiments finish, push results back so the main server can access them:

**Via GCS:**

```bash
gsutil -m cp -r results/ gs://pwm-benchmark-datasets/results/<server-name>/
```

**Via Git:**

```bash
git add results/
git commit -m "results: CACTI experiment from server-A"
git push origin master
```
