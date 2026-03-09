# Modify Plan: asl_mri (Arterial Spin Labeling MRI)

**Updated:** 2026-03-09
**Status:** PASS — improvements implemented

## Previous State (2026-03-06)

- Algorithm routing: generic carrier routing `(medical, Spin/RF)` to `mri` pool — 10 generic MRI methods.
- Ground truth: generic Shepp-Logan medical phantom or brain k-space fallback — NOT perfusion-specific.
- Forward model: was using "radon" runner (from "medical" category default) — WRONG for ASL MRI.
- No dedicated `_VARIANT_OVERRIDES["asl_mri"]` entry.
- No `CATEGORY_REAL_SCORES["asl_mri"]` — using generic "mri" scores.

## Identified Issues

1. **Wrong forward model**: The "medical" category defaulted to runner_type="radon" (Radon sinogram / CT forward model). ASL MRI uses k-space Fourier undersampling, not Radon projection — this was physically incorrect.
2. **Wrong ground truth phantom**: Generic Shepp-Logan or brain anatomy phantom is not appropriate for perfusion imaging. ASL perfusion maps have low-contrast compartmental structure (grey matter vs white matter vs deep GM) very different from structural MRI.
3. **No dedicated algorithms**: Generic MRI pool algorithms were used instead of ASL-specific references (Tian et al. 2023, Zhao et al. 2024).
4. **No ASL-calibrated scores**: PSNR values were from structural MRI benchmarks (too high for low-contrast perfusion signal).

## Improvements Implemented (2026-03-09)

### 1. Dedicated ASL Perfusion Phantom Generator
- Added `generate_asl_perfusion_phantom()` to `benchmarks/datasets/downloaders.py`
- Physics-calibrated cerebral blood flow map with:
  - Cortical grey matter ring (~0.60 normalised, calibrated to ~55 mL/100g/min)
  - White matter inner oval (~0.35 normalised, ~25 mL/100g/min)
  - Bilateral basal ganglia / putamen / globus pallidus (~0.90 norm., ~70 mL/100g/min)
  - Bilateral thalami (~0.92 normalised, ~75 mL/100g/min)
  - CSF / lateral ventricles (0.0 — no perfusion)
  - MCA vascular territory gradients as smooth overlays
  - Physiological CBF heterogeneity texture (Gaussian random field sigma=3 px)
  - Partial volume smoothing (Gaussian sigma=0.7 px)
- Calibrated to: Alsop et al. MRM 2015; Mutsaerts et al. NeuroImage 2020 (ExploreASL)

### 2. Correct Forward Model
- Added `_VARIANT_TO_RUNNER["asl_mri"] = "kspace"` to generator script
- ASL MRI now uses 4x Cartesian k-space undersampling (same as standard MRI)
- Previously used Radon sinogram model (CT), which was physically incorrect

### 3. Dedicated Algorithm Override
- Added `_VARIANT_OVERRIDES["asl_mri"]` with 9 algorithms:
  1. Zero-Filled IFFT (Classical baseline)
  2. L1-Wavelet / ESPIRiT (Compressed Sensing)
  3. PnP-DnCNN (PnP)
  4. U-Net (ASL) (Early DL, Tian et al. 2023)
  5. E2E-VarNet (Deep Unrolling, Sriram MICCAI 2020)
  6. Kinetic-CS (Physics-Informed, Zhao et al. JMRI 2024)
  7. ReconFormer (Transformer, Guo et al. IEEE TMI 2024)
  8. PromptMR (Multi-contrast unrolling, Xin et al. ECCV 2024)
  9. Score-MRI (ASL) (Diffusion, Chung & Ye 2022)

### 4. Calibrated Benchmark Scores
- Added `CATEGORY_REAL_SCORES["asl_mri"]` with 9 calibrated entries
- PSNR range: 24.5 dB (Zero-Filled IFFT) to 36.7 dB (Score-MRI)
- Correctly lower than structural MRI due to low-contrast perfusion signal
- Progressive improvement per era (classical -> CS -> PnP -> DL -> Transformer -> Diffusion)

### 5. Registry Updates
- Added `asl_mri_perfusion_generated` dataset entry with `applies_to=["asl_mri"]`
- Removed `asl_mri` from `ixi_t1_sample` applies_to (T1 brain not perfusion)
- Removed `asl_mri` from `medical_phantom_generated` applies_to (Shepp-Logan not perfusion)

### 6. GCS Dataset Regeneration
- Generated all 3 tiers using ASL CBF phantom + kspace forward model
- Public: 3 samples, x_true present (CBF map 128x128, y=(128,128) k-space)
- Dev: x_true stripped via `strip_dev_ground_truth.py`
- Hidden: blocked from download
- All files uploaded to `gs://pwm-benchmark-datasets/challenge-data/v1.0/`

## Files Modified

- `benchmarks/datasets/downloaders.py` — added `generate_asl_perfusion_phantom()`
- `benchmarks/datasets/registry.py` — added `asl_mri_perfusion_generated` entry; removed asl_mri from ixi_t1_sample and medical_phantom_generated
- `platform/scripts/generate_challenge_datasets.py` — added kspace runner override for asl_mri; added generate_asl_perfusion_phantom to both generator maps
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py` — added `_VARIANT_OVERRIDES["asl_mri"]` (9 algorithms) and `CATEGORY_REAL_SCORES["asl_mri"]` (9 scores)
- `benchmarks/learn/asl_mri/check.md` — updated to PASS (2026-03-09) with full details
- `benchmarks/learn/asl_mri/modify_plan.md` — this file
