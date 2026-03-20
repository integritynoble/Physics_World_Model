# Modify Plan: dark_field

**Date:** 2026-03-09

## Change Log

### 2026-03-09 — Full dark_field integration

**Changes made:**

1. **`benchmarks/datasets/downloaders.py`** — Added `generate_dark_field_phantom()`:
   - Simulates dark-field optical microscopy: sparse Gaussian bright spots (0.8–1.0 intensity) on dark background (~0.02)
   - Noise model: Poisson (scale=100) + Gaussian (sigma=0.02)
   - Metadata: modality, particle_size_nm, wavelength_nm, NA
   - Returns list of 3 dicts with x_true, y, H_ideal, metadata
   - Registered in both `_generated_converters` and `converter_map` inside `load_and_convert_dataset()`

2. **`benchmarks/datasets/registry.py`** — Added `dark_field_generated` DatasetEntry:
   - source_type="generated", applies_to=["dark_field"], converter="generate_dark_field_phantom"
   - Citation: Siedentopf & Zsigmondy, Ann. Physik 1902

3. **`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`** — Added:
   - `_VARIANT_OVERRIDES["dark_field"]` with 9 domain-specific algorithms (Classical → Diffusion)
   - `CATEGORY_REAL_SCORES["dark_field"]` with 9 PSNR/SSIM score entries
   - Algorithms: Richardson-Lucy, Wiener-DF, TV-DF, BM3D-DF, CARE-DF, Noise2Void-DF, SwinIR-DF, Restormer-DF, DiffusionDF

4. **`platform/scripts/generate_challenge_datasets.py`** — Added:
   - `"dark_field": "identity"` to `_VARIANT_TO_RUNNER`
   - `generate_dark_field_phantom` to both import blocks and both generator maps

5. **GCS** — Generated and uploaded all 3 tiers:
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/dark_field_challenge_public.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/dark_field_challenge_dev.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/dark_field_challenge_hidden.h5`

## Previous State (2026-03-06)

- **Category:** microscopy
- **Carrier:** Photon
- **Routing:** Direct to `microscopy` pool (no carrier routing override)
- **Score key:** microscopy
- **Algorithms served:**
  1. Richardson-Lucy (Classical) -- Richardson 1972 / Lucy 1974
  2. PnP-FISTA (PnP) -- Bai et al., 2020
  3. CARE (Deep Learning) -- Weigert et al., Nat. Methods 2018
  4. Restormer (Transformer) -- Zamir et al., CVPR 2022

## Current State (2026-03-09)

- **Category:** microscopy
- **Carrier:** Photon
- **Routing:** `_VARIANT_OVERRIDES["dark_field"]` (9 domain-specific algorithms)
- **Score key:** dark_field (direct CATEGORY_REAL_SCORES lookup)
- **Phantom generator:** `generate_dark_field_phantom` (sparse sub-wavelength particle scattering)
- **Algorithms served:** Richardson-Lucy, Wiener-DF, TV-DF, BM3D-DF, CARE-DF, Noise2Void-DF, SwinIR-DF, Restormer-DF, DiffusionDF
- **GCS datasets:** 3 tiers uploaded and verified
