# Multi-Server Benchmark Setup Guide

This guide explains how to set up new servers to run PWM benchmark experiments in parallel across different imaging modalities.

## Architecture Overview

```
                    ┌──────────────────────────┐
                    │   GCS Bucket             │
                    │   pwm-benchmark-datasets │
                    │                          │
                    │   datasets/sd_cassi/     │  2.6 GB
                    │   datasets/cacti/        │  1.6 GB
                    │   datasets/spc_kronecker/│  151 MB
                    │   challenge-data/v1.0/   │  5.2 GB
                    └──────┬───────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Server A │ │ Server B │ │ Server C │
        │ CACTI    │ │ SD-CASSI │ │ SPC      │
        └──────────┘ └──────────┘ └──────────┘
```

Benchmark datasets are stored on Google Cloud Storage (GCS). Each server clones the repo and downloads only the data it needs. The `datasets/` directory is gitignored — data never goes into git.

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

## Step 7: Run Your Experiment

```bash
# Each modality has its own experiment script
python3 scripts/run_cacti_experiment.py          # Server A
python3 scripts/run_cassi_experiment.py          # Server B
python3 scripts/run_spc_experiment.py            # Server C
```

---

## GCS Data Structure

| Path | Size | Content |
|------|------|---------|
| `datasets/sd_cassi/` | 2.6 GB | SD-CASSI source data (all tiers + images) |
| `datasets/cacti/` | 1.6 GB | CACTI source data (all tiers + images) |
| `datasets/spc_kronecker/` | 151 MB | SPC Kronecker source data |
| `challenge-data/v1.0/` | 5.2 GB | All 507 challenge HDF5 files |

---

## Parallel Deployment Example

| Server | Modality | Download Command | Run Command |
|--------|----------|------------------|-------------|
| A | CACTI | `./scripts/setup_benchmark_data.sh cacti` | `python3 scripts/run_cacti_experiment.py` |
| B | SD-CASSI | `./scripts/setup_benchmark_data.sh sd_cassi` | `python3 scripts/run_cassi_experiment.py` |
| C | SPC | `./scripts/setup_benchmark_data.sh spc_kronecker` | `python3 scripts/run_spc_experiment.py` |

---

## Quick One-Liner Setup

Set up a new server with a single command chain:

```bash
git clone git@github.com:integritynoble/Physics_World_Model.git && \
cd Physics_World_Model && \
python3 -m venv .venv && source .venv/bin/activate && \
pip install -e packages/pwm_core && \
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
