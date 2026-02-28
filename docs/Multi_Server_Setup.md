# Multi-Server Benchmark Setup Guide

This guide explains how to set up new servers to run PWM benchmark experiments in parallel across different imaging modalities, including GPU-accelerated algorithms via [Modal](https://modal.com/).

## Architecture Overview

```
                    ┌──────────────────────────────┐
                    │   GCS Bucket                 │
                    │   pwm-benchmark-datasets     │
                    │                              │
                    │   datasets/sd_cassi/    2.6G │
                    │   datasets/cacti/       1.6G │
                    │   datasets/spc_kronecker/151M│
                    │   checkpoint/            17G │
                    │   challenge-data/v1.0/  5.2G │
                    └──────┬───────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │ Server A  │   │ Server B  │   │ Server C  │
    │ CACTI     │   │ SD-CASSI  │   │ SPC       │
    │ (CPU)     │   │ (GPU)     │   │ (CPU)     │
    └───────────┘   └─────┬─────┘   └───────────┘
                          │
                    ┌─────▼─────┐
                    │  Modal    │
                    │  (GPU)    │
                    │ ELP, MST  │
                    │ HDNet ... │
                    └───────────┘
```

Benchmark datasets and model checkpoints are stored on Google Cloud Storage (GCS). Each server clones the repo and downloads only the data it needs. The `datasets/` and `checkpoint/` directories are gitignored — large data never goes into git.

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

## Step 7: Download Model Checkpoints

Pretrained model weights (17 GB total) are required for GPU-accelerated algorithms like ELP-Unfolding, MST, EfficientSCI, etc.

```bash
# Download all checkpoints
./scripts/setup_benchmark_data.sh --checkpoints

# Or download only the checkpoint you need
./scripts/setup_benchmark_data.sh --checkpoint ELP-Unfolding     # 13 GB
./scripts/setup_benchmark_data.sh --checkpoint EfficientSCI      # 34 MB
./scripts/setup_benchmark_data.sh --checkpoint MST-HDNet         # 35 MB

# Download everything at once (datasets + checkpoints)
./scripts/setup_benchmark_data.sh --all
```

### Available Checkpoints

| Checkpoint | Size | GPU Required | Used By |
|-----------|------|-------------|---------|
| `ELP-Unfolding` | 13 GB | Yes | SD-CASSI, CACTI reconstruction |
| `HATNet-SPI` | 3.7 GB | Yes | SPC reconstruction |
| `DRUNet` | 250 MB | Yes | PnP denoiser (all modalities) |
| `EfficientSCI` | 34 MB | Yes | CACTI reconstruction |
| `MST-HDNet` | 35 MB | Yes | SD-CASSI reconstruction |
| `ProxUnroll` | 38 MB | Yes | General inverse problems |
| `PnP-SCI` | 42 MB | Yes | CACTI PnP reconstruction |
| `ISTA-Net` | 21 MB | Yes | Compressive sensing |
| `PnP-CASSI` | 7.5 MB | Yes | SD-CASSI PnP reconstruction |
| `DnCNN` | 2.2 MB | Yes | Denoising prior |

## Step 8: Run Your Experiment

```bash
# Each modality has its own experiment script
python3 scripts/run_cacti_experiment.py          # Server A
python3 scripts/run_cassi_experiment.py          # Server B
python3 scripts/run_spc_experiment.py            # Server C
```

---

## GPU Setup with Modal

For algorithms that require GPU (ELP-Unfolding, MST, EfficientSCI, HATNet-SPI, etc.), you can use [Modal](https://modal.com/) to run GPU workloads without managing GPU servers.

### Why Modal?

- No GPU server management — spin up A100/H100 GPUs on demand
- Pay per second of GPU usage
- Checkpoints can be mounted as Modal volumes (no re-download)
- Works from any CPU-only server

### Modal Setup

```bash
# 1. Install Modal
pip install modal

# 2. Authenticate (one-time, opens browser)
modal setup

# 3. Create a Modal volume for checkpoints (one-time)
modal volume create pwm-checkpoints
```

### Upload Checkpoints to Modal Volume

```bash
# Upload all checkpoints to Modal volume (one-time)
modal volume put pwm-checkpoints checkpoint/ /checkpoint/

# Or upload specific ones
modal volume put pwm-checkpoints checkpoint/ELP-Unfolding/ /checkpoint/ELP-Unfolding/
modal volume put pwm-checkpoints checkpoint/MST-HDNet/ /checkpoint/MST-HDNet/
```

### Example: Run ELP-Unfolding on Modal GPU

Create a file `modal_run_elp.py`:

```python
import modal

app = modal.App("pwm-benchmark")

# Mount checkpoint volume and repo code
vol = modal.Volume.from_name("pwm-checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "numpy", "scipy", "h5py", "matplotlib")
    .pip_install("scikit-image")
)

@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoint": vol},
    timeout=3600,
    mounts=[modal.Mount.from_local_dir("packages/pwm_core", remote_path="/root/pwm_core")],
)
def run_elp_reconstruction(challenge_data: bytes, variant: str = "sd_cassi"):
    """Run ELP-Unfolding reconstruction on GPU."""
    import sys
    sys.path.insert(0, "/root")

    import numpy as np
    import h5py
    import io
    import torch

    # Load challenge data
    f = h5py.File(io.BytesIO(challenge_data), "r")

    results = {}
    for sample_key in sorted(f.keys()):
        sample = f[sample_key]
        y = sample["y"][()]
        H = sample["H_ideal"][()]

        # Load ELP model with GPU checkpoint
        from pwm_core.recon.elp_unfolding import build_elp_unfolding
        solver = build_elp_unfolding(
            weights_path="/checkpoint/ELP-Unfolding/ckptall.pth",
            device="cuda",
        )

        # Reconstruct
        x_recon = solver(torch.tensor(y).cuda(), torch.tensor(H).cuda())
        results[sample_key] = x_recon.cpu().numpy()

    return results


@app.local_entrypoint()
def main():
    # Read local challenge file
    with open("datasets/benchmark/sd_cassi/public/sd_cassi_challenge_public.h5", "rb") as f:
        data = f.read()

    results = run_elp_reconstruction.remote(data, "sd_cassi")
    print(f"Reconstructed {len(results)} scenes")

    # Save results
    import numpy as np
    np.savez("results/elp_results.npz", **results)
```

Run it:

```bash
# Run on Modal GPU from any machine (even CPU-only)
modal run modal_run_elp.py
```

### Example: Run Multiple Modalities in Parallel on Modal

```python
@app.local_entrypoint()
def main():
    import concurrent.futures

    variants = ["sd_cassi", "cacti", "spc_kronecker"]
    futures = []
    for variant in variants:
        with open(f"datasets/benchmark/{variant}/public/{variant}_challenge_public.h5", "rb") as f:
            data = f.read()
        futures.append(run_elp_reconstruction.remote(data, variant))

    # All three run in parallel on separate GPUs
    for variant, result in zip(variants, futures):
        print(f"{variant}: {len(result)} scenes reconstructed")
```

### Modal vs GCS Checkpoints: When to Use Which

| Scenario | Use GCS | Use Modal Volume |
|----------|---------|-----------------|
| Server has GPU | Download via `--checkpoints` | Not needed |
| Server is CPU-only, need GPU | Not needed | Upload once, mount in Modal functions |
| Running on multiple GPU servers | Download to each | Share one volume across all |
| One-off GPU experiment | Not needed | Upload + run, no server setup |

---

## GCS Data Structure

| Path | Size | Content |
|------|------|---------|
| `datasets/sd_cassi/` | 2.6 GB | SD-CASSI source data (all tiers + images) |
| `datasets/cacti/` | 1.6 GB | CACTI source data (all tiers + images) |
| `datasets/spc_kronecker/` | 151 MB | SPC Kronecker source data |
| `checkpoint/` | 17 GB | Pretrained model weights (11 algorithms) |
| `challenge-data/v1.0/` | 5.2 GB | All 507 challenge HDF5 files |

---

## Parallel Deployment Example

| Server | Modality | Setup Command | Run Command |
|--------|----------|---------------|-------------|
| A (CPU) | CACTI | `./scripts/setup_benchmark_data.sh cacti` | `python3 scripts/run_cacti_experiment.py` |
| B (GPU) | SD-CASSI | `./scripts/setup_benchmark_data.sh --all sd_cassi` | `python3 scripts/run_cassi_experiment.py` |
| C (CPU) | SPC | `./scripts/setup_benchmark_data.sh spc_kronecker` | `python3 scripts/run_spc_experiment.py` |
| D (Modal) | All GPU | `modal run modal_run_elp.py` | Runs on Modal A100 |

---

## Quick One-Liner Setup

**CPU server (data + experiment):**

```bash
git clone git@github.com:integritynoble/Physics_World_Model.git && \
cd Physics_World_Model && \
python3 -m venv .venv && source .venv/bin/activate && \
pip install -e packages/pwm_core && \
./scripts/setup_benchmark_data.sh cacti && \
python3 scripts/run_cacti_experiment.py
```

**GPU server (data + checkpoints + experiment):**

```bash
git clone git@github.com:integritynoble/Physics_World_Model.git && \
cd Physics_World_Model && \
python3 -m venv .venv && source .venv/bin/activate && \
pip install -e packages/pwm_core && \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
./scripts/setup_benchmark_data.sh --all && \
python3 scripts/run_cassi_experiment.py
```

**Modal (no GPU server needed):**

```bash
git clone git@github.com:integritynoble/Physics_World_Model.git && \
cd Physics_World_Model && \
pip install modal && modal setup && \
./scripts/setup_benchmark_data.sh sd_cassi && \
modal run modal_run_elp.py
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
