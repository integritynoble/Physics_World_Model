# Modify Plan: coded_exposure (Coded Exposure / Flutter Shutter)

**Updated:** 2026-03-09
**Status:** PASS — phantom, overrides, and GCS datasets added (2026-03-09)

## Current State

- Algorithm routing: `_VARIANT_OVERRIDES["coded_exposure"]` — 9 domain-specific motion deblurring methods.
- Phantom generator: `generate_coded_exposure_phantom` in `benchmarks/datasets/downloaders.py`.
- Registry entry: `coded_exposure_generated` in `benchmarks/datasets/registry.py`.
- `CATEGORY_REAL_SCORES["coded_exposure"]` added with 9 benchmark scores (PSNR 26.5–39.8, SSIM 0.791–0.961).
- Runner: `"coded_exposure": "identity"` in `_VARIANT_TO_RUNNER`.
- Challenge datasets: all 3 tiers on GCS at `gs://pwm-benchmark-datasets/challenge-data/v1.0/coded_exposure_challenge_{public,dev,hidden}.h5`.

## Changes Made (2026-03-09)

1. Added `generate_coded_exposure_phantom()` to `benchmarks/datasets/downloaders.py` — implements Raskar 52-bit flutter shutter code with horizontal motion convolution and read noise.
2. Added `coded_exposure_generated` DatasetEntry to `benchmarks/datasets/registry.py`.
3. Added `_VARIANT_OVERRIDES["coded_exposure"]` to `_algorithm_catalog.py` — 9 deblurring methods (Classical through Diffusion, 2006–2022).
4. Added `CATEGORY_REAL_SCORES["coded_exposure"]` to `_algorithm_catalog.py` — 9 benchmark scores.
5. Added `"coded_exposure": "identity"` to `_VARIANT_TO_RUNNER` in `generate_challenge_datasets.py`.
6. Added `generate_coded_exposure_phantom` to both import blocks and generator maps in `generate_challenge_datasets.py`.
7. Generated and uploaded all 3 challenge tiers to GCS.

## Verdict

PASS. Flutter shutter phantom implemented correctly. Domain-specific algorithm override added. GCS datasets verified.
