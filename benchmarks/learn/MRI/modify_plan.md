# Modify Plan: MRI (Magnetic Resonance Imaging)

**Created:** 2026-03-03
**Status:** Done (items 1-3 fixed; items 4-5 deferred — require gallery regeneration)

## Changes Required

### 1. Add MRI-specific algorithm override (ERROR — FIXED)

**File:** `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
**Change:** Added `"mri"` key to `_VARIANT_OVERRIDES` with 8 MRI-specific algorithms:
- Zero-Filled IFFT (Classical), L1-Wavelet/ESPIRiT (CS), PnP-DnCNN (PnP)
- U-Net (Deep Learning), E2E-VarNet (Deep Unrolling), PromptMR (Deep Unrolling)
- ReconFormer (Transformer), Score-MRI (Diffusion)

### 2. Add MRI-specific real scores (ERROR — FIXED)

**File:** `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
**Change:** Added `"mri"` key to `CATEGORY_REAL_SCORES` with fastMRI 4x knee published values.

### 3. Fix leaderboard score lookup to check variant first (ERROR — FIXED)

**File:** `platform/pwm_platform/services/benchmark_database/_leaderboard_generator.py`
**Change:** `_generate_b2_leaderboard()` now checks `CATEGORY_REAL_SCORES[variant_key]` before `CATEGORY_REAL_SCORES[category]`.

### 4. Regenerate gallery with MRI forward model (WARNING — DEFERRED)

Gallery currently uses `medical_ct_radon` module. Needs regeneration with MRI k-space undersampling.

### 5. Fix gallery PSNR/SSIM values (WARNING — DEFERRED)

Gallery metrics are 13-18 dB PSNR — likely from wrong forward model. Will fix after gallery regeneration.

## Implementation

Items 1-3: Code changes in `_algorithm_catalog.py` and `_leaderboard_generator.py`.
Items 4-5: Require running `precompute_all_gallery.py` with corrected forward model.
