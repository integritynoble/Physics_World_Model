# Modify Plan: cars (CARS Microscopy)

**Updated:** 2026-03-09
**Status:** COMPLETE — all changes implemented

## Changes Implemented (2026-03-09)

### 1. Phantom Generator (`benchmarks/datasets/downloaders.py`)
- Added `generate_cars_raman_phantom()` after `generate_brillouin_vipa_phantom()`
- Simulates biological CARS images: lipid droplets (CH2 resonance ~2845 cm-1) and protein cytoplasm
- Forward model: CARS = chi_r^2 + 2*A_NRB*chi_r + A_NRB^2 (coherent superposition with NRB)
- Added to both `_generated_converters` and `converter_map` dicts in `acquire_dataset()`

### 2. Registry Entry (`benchmarks/datasets/registry.py`)
- Added `cars_raman_generated` DatasetEntry with `applies_to=["cars"]`, `converter="generate_cars_raman_phantom"`, `x_shape=[64, 64]`

### 3. Algorithm Overrides (`_algorithm_catalog.py`)
- Added `_VARIANT_OVERRIDES["cars"]` with 9 CARS-specific algorithms (2008-2024 coverage):
  - Classical: KK-Retrieval, MEM-CARS
  - Deep Learning: CNN-NRB, U-Net-CARS, ResNet-CARS
  - Physics-Informed: PINN-CARS
  - Transformer: SpecFormer-CARS
  - Diffusion: Diff-CARS, FMDiff-CARS

### 4. Real Scores (`CATEGORY_REAL_SCORES["cars"]`)
- Added 9 benchmark results with realistic PSNR/SSIM values (24.5 dB → 40.2 dB)
- Uses `"method"` key (required by leaderboard generator `_generate_b2_leaderboard`)

### 5. Runner Routing (`generate_challenge_datasets.py`)
- Added `"cars": "identity"` to `_VARIANT_TO_RUNNER`
- Added `generate_cars_raman_phantom` to imports and `_GENERATOR_MAP`

### 6. GCS Upload
- Generated and uploaded 3 challenge tiers to GCS:
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/cars_challenge_public.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/cars_challenge_dev.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/cars_challenge_hidden.h5`

## Verdict

COMPLETE. CARS now has a dedicated phantom generator, domain-specific algorithm override
with 9 algorithms (2008-2024), realistic benchmark scores, and GCS challenge datasets.
The identity runner is appropriate as the phantom `y` is already in measurement space
(CARS intensity with NRB mixing).
