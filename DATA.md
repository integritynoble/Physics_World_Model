# Dataset & Model Weight Storage

All heavy datasets, model weights, and reconstruction results are stored in Google Cloud Storage (GCS).
This repository contains only code, configs, and lightweight metadata.

## GCS Bucket

**Bucket**: `gs://pwm-benchmark-datasets/`
**Project**: `subtle-app-431618-i1`

## Directory Structure on GCS

```
gs://pwm-benchmark-datasets/
├── benchmark/                    # 168-modality benchmark datasets (~8.7 GB)
│   ├── {modality}/
│   │   ├── public/               # 12 public samples (h5 + images)
│   │   ├── dev/                  # 20 dev samples
│   │   └── hidden/               # 20 hidden samples (ground truth withheld)
├── inversenet/                   # InverseNet paper datasets
│   ├── cacti/                    # CACTI samples
│   ├── cassi/                    # CASSI samples
│   └── spc/                      # SPC samples (~6.5 MB mask)
├── lip_arena/                    # LIP Arena evaluation samples
│   └── {modality}/x_gt.npy, y.npy
├── weights/                      # Pre-trained model weights
│   ├── care/care_2d.pth          # CARE 2D fluorescence restoration
│   ├── care/care_3d.pth          # CARE 3D confocal restoration
│   ├── redcnn/redcnn.pth         # RED-CNN low-dose CT denoising
│   ├── efficientsci/             # EfficientSCI CACTI reconstruction
│   ├── flatnet/flatnet.pth       # FlatNet lensless imaging
│   ├── phasenet/phasenet.pth     # PhaseNet holography reconstruction
│   ├── mst/                      # MST hyperspectral reconstruction
│   ├── hdnet/hdnet.pth           # HDNet hyperspectral reconstruction
│   ├── modl/modl.pth             # MoDL MRI reconstruction
│   └── ...
└── results/                      # Paper reconstruction results
    ├── inversenet/cacti/         # CACTI reconstructions (100-130 MB each)
    ├── inversenet/cassi/         # CASSI reconstructions
    ├── inversenet/spc/           # SPC reconstructions
    └── pwmi_cassi/               # PWMi-CASSI reconstructions
```

## Download Instructions

### Prerequisites

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth application-default login
```

### Download Benchmark Datasets (168 modalities)

```bash
# Download all public tier data (~2 GB)
gsutil -m cp -r gs://pwm-benchmark-datasets/benchmark/*/public/ datasets/benchmark/

# Or download a specific modality
gsutil -m cp -r gs://pwm-benchmark-datasets/benchmark/mri/ datasets/benchmark/mri/
```

### Download Pre-trained Model Weights

```bash
# Download all weights (~2 GB)
gsutil -m cp -r gs://pwm-benchmark-datasets/weights/ \
    packages/pwm_core/pwm_core/weights/

# Or individual models
gsutil cp gs://pwm-benchmark-datasets/weights/care/care_2d.pth \
    packages/pwm_core/pwm_core/weights/care/care_2d.pth
```

### Download InverseNet / LIP Arena Data

```bash
gsutil -m cp -r gs://pwm-benchmark-datasets/inversenet/ datasets/inversenet_cacti/
gsutil -m cp -r gs://pwm-benchmark-datasets/lip_arena/ datasets/lip_arena/
```

### Download Paper Results

```bash
# InverseNet reconstructions
gsutil -m cp -r gs://pwm-benchmark-datasets/results/inversenet/ \
    papers/inversenet/results/

# PWMi-CASSI reconstructions
gsutil -m cp -r gs://pwm-benchmark-datasets/results/pwmi_cassi/ \
    papers/pwmi_cassi/results/
```

## Upload Script

To upload local datasets to GCS (run when credentials are available):

```bash
python scripts/upload_to_gcs.py
```

## Python Download Helper

```python
from google.cloud import storage

def download_benchmark_data(modality, tier="public", local_dir="datasets/benchmark"):
    """Download benchmark data for a specific modality from GCS."""
    client = storage.Client()
    bucket = client.bucket("pwm-benchmark-datasets")
    prefix = f"benchmark/{modality}/{tier}/"

    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        local_path = Path(local_dir) / blob.name.replace(f"benchmark/", "")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        print(f"Downloaded: {local_path}")
```

## Dataset Sizes

| Dataset | Size | Description |
|---------|------|-------------|
| benchmark/ | ~8.7 GB | 168-modality benchmark (public+dev+hidden) |
| weights/ | ~2 GB | Pre-trained DL model weights |
| results/inversenet/ | ~1.5 GB | InverseNet paper reconstruction outputs |
| results/pwmi_cassi/ | ~1.3 GB | PWMi-CASSI reconstruction outputs |
| inversenet/ | ~7 MB | InverseNet demo samples |
| lip_arena/ | ~400 KB | LIP Arena evaluation samples |
