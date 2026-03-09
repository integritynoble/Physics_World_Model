# Modify Plan: confocal_endomicroscopy

## Status: COMPLETED (2026-03-09)

All required changes have been implemented.

## Changes Made

### 1. Phantom Generator (`benchmarks/datasets/downloaders.py`)
Added `generate_confocal_endomicroscopy_phantom()` — colonic crypt phantom with:
- Hexagonal crypt grid (epithelium walls + dark lumens)
- Background stroma
- Forward model: fibre bundle honeycomb artefact + PSF blur + Rayleigh speckle noise
- `target_shape` support for dataset generator compatibility

### 2. Registry Entry (`benchmarks/datasets/registry.py`)
Added `confocal_endomicroscopy_generated` DatasetEntry:
- `source_type="generated"`, `applies_to=["confocal_endomicroscopy"]`
- `converter="generate_confocal_endomicroscopy_phantom"`, `x_shape=[128, 128]`

### 3. Algorithm Override (`_algorithm_catalog.py` `_VARIANT_OVERRIDES`)
Replaced 4-algorithm OCT-adjacent set with 9 CLE-specific algorithms:
- NLM-Speckle, BM3D-CLE (Classical)
- DnCNN-CLE, U-Net-CLE, CARE-CLE (Deep Learning)
- SwinIR-CLE, Restormer-CLE (Transformer)
- PINN-CLE (Physics-Informed)
- DiffusionEndo (Diffusion)

### 4. Score Table (`_algorithm_catalog.py` `CATEGORY_REAL_SCORES`)
Added `confocal_endomicroscopy` entry with 9 PSNR/SSIM scores (25.5–39.4 dB).
Removed stale alias `confocal_endomicroscopy -> fiber_endoscopy` from `_VARIANT_SCORE_ALIASES`.

### 5. Runner Routing (`generate_challenge_datasets.py`)
Added `"confocal_endomicroscopy": "identity"` to `_VARIANT_TO_RUNNER`.
Added `generate_confocal_endomicroscopy_phantom` to both import blocks and both generator maps.

### 6. GCS Upload
Generated and uploaded 3 HDF5 challenge tiers:
```
gs://pwm-benchmark-datasets/challenge-data/v1.0/confocal_endomicroscopy_challenge_public.h5
gs://pwm-benchmark-datasets/challenge-data/v1.0/confocal_endomicroscopy_challenge_dev.h5
gs://pwm-benchmark-datasets/challenge-data/v1.0/confocal_endomicroscopy_challenge_hidden.h5
```
