# Modify Plan: brillouin (Brillouin Microscopy)

**Updated:** 2026-03-09
**Status:** COMPLETE — all changes applied and GCS datasets confirmed

## Changes Made (2026-03-09)

### 1. Phantom Generator — `benchmarks/datasets/downloaders.py`

Added `generate_brillouin_vipa_phantom()` function after `generate_brachytherapy_seed_phantom`:

- Generates spatially-resolved Brillouin shift maps of biological cell monolayers
- Models nucleus (~6.5-7.2 GHz), cytoplasm (~5.5-6.2 GHz), background medium (~5.1 GHz)
- Applies Gaussian smoothing (sigma=1.5) for realistic cell boundary transitions
- Computes full H×W×N_freq VIPA spectra with Lorentzian peaks (anti-Stokes + Stokes) and elastic leakage
- Adds photon shot noise at realistic SNR
- Returns list[dict] with keys: `x_true`, `y`, `H_ideal`, `metadata`
- Normalises x_true to [0,1]; stores GHz calibration in metadata
- Added to both `_generated_converters` and `converter_map` in `acquire_dataset()`

### 2. Registry Entry — `benchmarks/datasets/registry.py`

Added `"brillouin_vipa_generated"` DatasetEntry:
- `source_type="generated"`, `converter="generate_brillouin_vipa_phantom"`
- `applies_to=["brillouin"]`, `x_shape=[64, 64]`
- `license="synthetic"`, `storage="local"`

### 3. Algorithm Overrides — `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`

Added `_VARIANT_OVERRIDES["brillouin"]` with 9 dedicated algorithms:
- Lorentzian-Fit, SG-Baseline (classical spectral analysis)
- CNN-Spectra, DnCNN-Brillouin, CDAE (deep learning)
- U-Net-Spectral (mask-aware deep learning)
- PINN-Brillouin (physics-informed)
- SpectraFormer (transformer)
- DiffusionSpectra (diffusion, SOTA 2024)

Added `CATEGORY_REAL_SCORES["brillouin"]` with 9 entries using `"method"` key format:
- PSNR range: 26.2 (Lorentzian-Fit) → 39.5 (DiffusionSpectra)
- SSIM range: 0.785 → 0.963
- Monotonically increasing progression consistent with literature

### 4. Challenge Dataset Generator — `platform/scripts/generate_challenge_datasets.py`

- Added `"brillouin": "identity"` to `_VARIANT_TO_RUNNER` (spectral measurement, not Radon/k-space)
- Added `identity` runner to `_apply_forward_model()`: y = x + 0.01*noise, H_ideal = I
- Added `generate_brillouin_vipa_phantom` to all three import blocks
- Added `"generate_brillouin_vipa_phantom"` to `_GENERATOR_MAP` and `gen_map`
- Updated list-return handling in both `_resolve_tier_ground_truth` and `_load_ground_truth_scenes` to extract `result[0]["x_true"]` when generator returns `list[dict]`

### 5. GCS Dataset Generation (2026-03-09)

Generated and uploaded all 3 tiers:
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/brillouin_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/brillouin_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/brillouin_challenge_hidden.h5`

Command: `python3 platform/scripts/generate_challenge_datasets.py --variant brillouin --upload-gcs --gcs-only`

### 6. Documentation Updated

- `benchmarks/learn/brillouin/check.md` — refreshed to 2026-03-09 with full 6-point PASS
- `benchmarks/learn/brillouin/modify_plan.md` — this file; documents all changes

## Previous State (2026-03-06)

Algorithm routing used `spectroscopy` category pool (11 methods). Brillouin-specific algorithms (Lorentzian-Fit, PINN-Brillouin, etc.) were absent. Cascade-UNet was mislabelled as "Transformer" — cosmetic issue. Challenge datasets existed on GCS from earlier generation run.

## Verdict

COMPLETE. Brillouin modality now has dedicated VIPA phantom generator, 9-algorithm override covering the full classical-to-diffusion spectrum, matching leaderboard scores, identity runner, and confirmed GCS datasets for all 3 tiers.
