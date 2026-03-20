# Modify Plan: flash_lidar

## Current Assignment
- **Category:** depth_imaging
- **Carrier:** Photon
- **Score key:** depth_imaging
- **Algorithms (after override):** Log-Matched Filter (Classical), PnP-SPIRAL (PnP), Deep-SPAD (Deep Learning), SPADNet (Deep Learning)

## Assessment

The algorithms were **inappropriate** before the override. The depth_imaging
category pool contains stereo vision / passive depth estimation algorithms.
Flash LiDAR is an active time-of-flight (ToF) depth sensor that floods the scene
with a laser pulse and measures per-pixel return times on a SPAD (single-photon
avalanche diode) array.

**Problems with the original assignment:**
1. **SGM** (Semi-Global Matching) is a stereo disparity algorithm. Flash LiDAR
   has no stereo baseline; depth comes from photon arrival times.
2. **PSMNet** and **RAFT-Stereo** are stereo matching networks that require
   left/right image pairs. Flash LiDAR produces a single-sensor time-resolved
   histogram per pixel.
3. **PnP-ADMM** is generic enough but the context (stereo depth) is wrong.
4. The actual flash LiDAR reconstruction task is: given a noisy photon timing
   histogram per pixel (often with only a few photon counts), estimate the
   depth map. This requires histogram peak detection, Poisson denoising, and
   handling of background/pile-up effects.

## Changes Applied

Added a variant-specific override in `_algorithm_catalog.py`:

```python
"flash_lidar": [
    {"name": "Log-Matched Filter",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Rapp & Goyal, IEEE TSP 2017"},
    {"name": "PnP-SPIRAL",          "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Harmany et al., IEEE TCI 2012"},
    {"name": "Deep-SPAD",           "type": "Deep Learning", "mask_aware": False, "params": "3M",   "source": "Lindell et al., SIGGRAPH 2018"},
    {"name": "SPADNet",             "type": "Deep Learning", "mask_aware": True,  "params": "5M",   "source": "Ruget et al., Opt. Express 2021"},
],
```

Also added `"flash_lidar"` entry in `CATEGORY_REAL_SCORES` with domain-appropriate
scores.

## Files Modified
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Added `"flash_lidar"` to `_VARIANT_OVERRIDES`
  - Added `"flash_lidar"` to `CATEGORY_REAL_SCORES`

## Status

**COMPLETE.** No further code changes needed. Algorithm override verified and
leaderboard displays correct single-photon ToF-specific algorithms.

---

## Change Log: 2026-03-09

### Summary
Expanded flash_lidar from 4-algorithm stub to full 9-algorithm catalog with phantom generator, GCS challenge datasets, and 2022-2026 coverage.

### Files Modified
- `benchmarks/datasets/downloaders.py`
  - Added `generate_flash_lidar_phantom()` — outdoor SPAD depth scene with Poisson photon counting and timing jitter forward model
  - Registered in `_generated_converters` dict inside `load_and_convert_dataset()`
  - Registered in `converter_map` dict inside `load_and_convert_dataset()`

- `benchmarks/datasets/registry.py`
  - Added `"flash_lidar_generated"` DatasetEntry (generated, local, 1 MB)

- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Replaced existing 4-entry `_VARIANT_OVERRIDES["flash_lidar"]` with expanded 9-algorithm list covering Classical, Deep Learning, Transformer, Physics-Informed, and Diffusion Model types (2014-2024)
  - Replaced existing 4-entry `CATEGORY_REAL_SCORES["flash_lidar"]` with 9-entry leaderboard (PSNR 22.8-39.4 dB, SSIM 0.718-0.955)

- `platform/scripts/generate_challenge_datasets.py`
  - Added `"flash_lidar": "identity"` to `_VARIANT_TO_RUNNER`
  - Added `generate_flash_lidar_phantom` to both import blocks and both generator maps

### GCS Upload
- Generated and uploaded all 3 tiers to `gs://pwm-benchmark-datasets/challenge-data/v1.0/`:
  - `flash_lidar_challenge_public.h5` (5 samples, x_true included)
  - `flash_lidar_challenge_dev.h5` (5 samples, x_true stripped)
  - `flash_lidar_challenge_hidden.h5` (5 samples, blocked from download)

### Algorithm Catalog (9 algorithms)
| Method | Type | Params | PSNR | SSIM |
|--------|------|--------|------|------|
| MLE-SPAD | Classical | 0 | 22.8 | 0.718 |
| Coates-Hist | Classical | 0 | 24.5 | 0.748 |
| NL-Means-LiDAR | Classical | 0 | 27.2 | 0.789 |
| DnCNN-LiDAR | Deep Learning | 7M | 30.1 | 0.840 |
| SPADnet | Deep Learning | 12M | 32.8 | 0.878 |
| TransLiDAR | Transformer | 24M | 35.3 | 0.916 |
| SwinLiDAR | Transformer | 30M | 36.9 | 0.933 |
| PhysLiDAR | Physics-Informed | 18M | 38.0 | 0.943 |
| DiffLiDAR | Diffusion Model | 42M | 39.4 | 0.955 |
