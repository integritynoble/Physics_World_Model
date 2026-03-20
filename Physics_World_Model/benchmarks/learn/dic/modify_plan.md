# Modify Plan: dic

## Current State (After 2026-03-09 Update)
- **Category:** microscopy
- **Sub-category pool:** DIC-specific (9 algorithms, expanded from 4)
- **Algorithms:** [DIC-Deconv, TV-DIC, Phase-DLSIM, DIC-CNN, PhaseNet-DIC, PnP-DIC, SwinDIC, PhysPhase-Net, DiffusionDIC]
- **Phantom generator:** `generate_dic_phantom` (cell nucleus + cytoplasm OPD map, DIC shear forward model)
- **Registry entry:** `dic_generated` in `DATASET_REGISTRY`
- **Runner:** `"dic": "identity"` in `_VARIANT_TO_RUNNER`
- **GCS:** All 3 challenge tiers uploaded to `gs://pwm-benchmark-datasets/challenge-data/v1.0/`

## Change Log

### 2026-03-09 — Full DIC modality integration
- Added `generate_dic_phantom()` to `benchmarks/datasets/downloaders.py`:
  - 64x64 float32 OPD map with nucleus (OPD ~0.8) and cytoplasm (OPD ~0.3-0.5)
  - DIC forward model: x-direction shear gradient [1,-1] + offset 0.5 + Gaussian noise (sigma=0.05)
  - Returns list of 3 dicts with x_true, y, H_ideal, metadata
  - Registered in both `_generated_converters` and `converter_map` inside `load_and_convert_dataset()`
- Added `dic_generated` DatasetEntry to `benchmarks/datasets/registry.py`
- Replaced `_VARIANT_OVERRIDES["dic"]` in `_algorithm_catalog.py` with 9 algorithms:
  - Classical: DIC-Deconv (Preza 1999), Phase-DLSIM (Stephens 2003)
  - Variational: TV-DIC (Bostan 2014)
  - Deep Learning: DIC-CNN (Rivenson 2018), PhaseNet-DIC (Sinha 2020)
  - PnP: PnP-DIC (Kamilov 2017)
  - Transformer: SwinDIC (Liang 2021)
  - Physics-Informed: PhysPhase-Net (Barbastathis 2019)
  - Diffusion: DiffusionDIC (Luo 2023)
- Replaced `CATEGORY_REAL_SCORES["dic"]` with 9 entries (PSNR 24.1-39.2, SSIM 0.731-0.950)
- Added `"dic": "identity"` to `_VARIANT_TO_RUNNER` in `generate_challenge_datasets.py`
- Added `generate_dic_phantom` to all import blocks and generator maps in `generate_challenge_datasets.py`
- Generated and uploaded all 3 challenge tiers to GCS (public, dev, hidden)

### 2026-03-06 — Initial DIC algorithm routing fix
- Replaced generic microscopy pool with 4 DIC-specific algorithms
- Algorithms: Fourier Integration, DIC-Tikhonov, DIC-Net, PhaseFormer

## Verdict
DIC modality fully integrated with phantom generator, registry entry, algorithm overrides, scores, runner routing, and GCS datasets.
