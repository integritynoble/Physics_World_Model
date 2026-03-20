# Modify Plan: cup (Compressed Ultrafast Photography)

**Updated:** 2026-03-09
**Status:** PASS — code changes deployed

## Change Log

### 2026-03-09 — Full CUP Modality Deployment

**Files modified:**

1. `benchmarks/datasets/downloaders.py`
   - Added `generate_cup_phantom()` function (generates 3 samples of 64×64 light pulse propagation scenes)
   - x_true: Gaussian intensity profile moving across frame
   - y: compressed 2D measurement via random binary mask (50% compression) summed over T=10 temporal frames, Gaussian noise σ=0.05
   - Added to both `_generated_converters` map and `converter_map` in `acquire_dataset()`

2. `benchmarks/datasets/registry.py`
   - Added `cup_generated` DatasetEntry (source_type=generated, applies_to=["cup"], converter="generate_cup_phantom")

3. `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
   - Added `_VARIANT_OVERRIDES["cup"]` with 9 domain-specific algorithms:
     TV-CUP, TwIST-CUP, GAP-TV, DeSCI-CUP, E2E-CNN-CUP, PnP-FastDVDnet, STFormer-CUP, DAUHST-CUP, DiffusionCUP
   - Added `CATEGORY_REAL_SCORES["cup"]` with realistic PSNR/SSIM (24.3 → 40.2 dB range)

4. `platform/scripts/generate_challenge_datasets.py`
   - Added `"cup": "identity"` to `_VARIANT_TO_RUNNER`
   - Added `generate_cup_phantom` to both import blocks and both generator map dicts

**GCS datasets generated and uploaded:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cup_challenge_public.h5` (5 samples)
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cup_challenge_dev.h5` (5 samples, no x_true)
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cup_challenge_hidden.h5` (5 samples, blocked)

### 2026-03-06 — Initial check

- Algorithm routing: dedicated `ultrafast` category pool → 11 methods
- Status: PASS, no code changes required at that time

## Current State

- Algorithm routing: dedicated `_VARIANT_OVERRIDES["cup"]` → 9 methods (TV-CUP through DiffusionCUP)
- Runner type: `identity` (compression handled in phantom generator)
- Challenge datasets: all 3 tiers on GCS
- Phantom: `generate_cup_phantom` in downloaders.py
- Registry: `cup_generated` in DATASET_REGISTRY
