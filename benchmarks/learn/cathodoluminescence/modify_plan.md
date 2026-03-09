# Modify Plan: cathodoluminescence (Cathodoluminescence Imaging)

**Updated:** 2026-03-09
**Status:** PASS — all code changes completed

## Changes Made (2026-03-09)

### 1. Phantom Generator (`benchmarks/datasets/downloaders.py`)
- Added `generate_cathodoluminescence_phantom()` after `generate_cacti_video_phantom()`
- Simulates CL intensity maps of semiconductor nanostructures with:
  - Plasmonic nanoparticles / quantum dots (bright circular features)
  - Grain boundary defects (linear dark features reducing CL by 70%)
  - Parabolic mirror PSF broadening (Gaussian, sigma 1.0–2.5 px)
  - PMT shot noise (Poisson approximation, gain 50–200)
  - Spectral background (uniform 0.01–0.05)
- Added to both `_generated_converters` and `converter_map` in `acquire_dataset()`
- Reference: Zagonel et al., Nano Lett. 2011; Tizei & Kociak, Phys. Rev. Lett. 2013

### 2. Dataset Registry (`benchmarks/datasets/registry.py`)
- Added `cathodoluminescence_generated` DatasetEntry
- `applies_to=["cathodoluminescence"]`, `x_shape=[128, 128]`
- converter: `generate_cathodoluminescence_phantom`

### 3. Algorithm Overrides (`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`)
- Added `_VARIANT_OVERRIDES["cathodoluminescence"]` with 9 CL-specific algorithms:
  - Classical: Wiener-CL, Richardson-Lucy
  - Deep Learning: DnCNN-CL, U-Net-CL, CARE-CL
  - Transformer: SwinIR-CL, Restormer-CL
  - Physics-Informed: PINN-CL
  - Diffusion: DiffusionEM
- Added `CATEGORY_REAL_SCORES["cathodoluminescence"]` with PSNR/SSIM for all 9 methods

### 4. Runner Routing (`platform/scripts/generate_challenge_datasets.py`)
- Added `"cathodoluminescence": "identity"` to `_VARIANT_TO_RUNNER`
- Added `generate_cathodoluminescence_phantom` to imports (both occurrences)
- Added to `_GENERATOR_MAP` and `gen_map` dictionaries
- Also fixed missing `generate_cars_raman_phantom` entry in `gen_map`

### 5. GCS Datasets
- Generated and uploaded all 3 challenge tiers to GCS:
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/cathodoluminescence_challenge_public.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/cathodoluminescence_challenge_dev.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/cathodoluminescence_challenge_hidden.h5`

## Current State

- Algorithm routing: dedicated `_VARIANT_OVERRIDES["cathodoluminescence"]` pool (9 methods)
- Wiener-CL and Richardson-Lucy are canonical classical PSF deconvolution baselines
- CARE-CL (Weigert et al., Nat. Methods 2018) is the gold standard for fluorescence/CL restoration
- SwinIR-CL, Restormer-CL, DiffusionEM cover 2021–2024 SOTA Transformer/Diffusion methods
- Challenge datasets on GCS for all three tiers
- Mismatch parameters: beam_current_drift, collection_efficiency_variation, spectral_calibration_error, carbon_contamination — all physically grounded in CL practice

## Verdict

PASS. Dedicated CL-specific algorithm override with 9 domain-appropriate methods. Phantom generator models realistic CL physics. Challenge datasets uploaded to GCS.
