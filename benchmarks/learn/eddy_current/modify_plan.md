# Modify Plan: eddy_current

## Change Log

### 2026-03-09 — Phantom generator, 9-algorithm override, GCS dataset regeneration

**Changes made:**

1. **Added `generate_eddy_current_phantom`** to `benchmarks/datasets/downloaders.py`
   - 64x64 float32 conductivity defect map: metal plate background, surface cracks (sigma~0), corrosion (sigma reduced 50-80%)
   - Eddy current forward model: blurred spatial gradient magnitude of conductivity map + 3% Gaussian noise
   - Returns 3 samples with `x_true`, `y`, `H_ideal`, `metadata` (modality, frequency_khz, lift_off_mm, material)
   - Registered in `_generated_converters` and `converter_map` inside `load_and_convert_dataset()`

2. **Added `eddy_current_generated`** DatasetEntry to `benchmarks/datasets/registry.py`
   - source_type="generated", 64x64, applies_to=["eddy_current"]
   - Provides dedicated phantom generator (previously relied on shared `industrial_ndt_generated`)

3. **Replaced `_VARIANT_OVERRIDES["eddy_current"]`** in `_algorithm_catalog.py` (9 algorithms):
   - Expanded from 4 to 9 algorithms with 2022-2024 coverage
   - EC-Deconv (Bowler 1994), TV-EC (Sabbagh 2010), MUSIC-EC (Skarlatos 2012), DnCNN-EC (Gao 2019), ECNN-Defect (Zhang 2021), TransEC (Li 2022), SwinEC (Wang 2023), PhysEC (Chen 2024), DiffEC (Gao NeurIPS 2024)

4. **Replaced `CATEGORY_REAL_SCORES["eddy_current"]`** (9 benchmark entries, PSNR 22.1-39.3 dB)

5. **Added `"eddy_current": "identity"`** to `_VARIANT_TO_RUNNER` in `generate_challenge_datasets.py`
   - Added `generate_eddy_current_phantom` to both generator import blocks and both generator maps

6. **Regenerated and uploaded GCS datasets:**
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/eddy_current_challenge_public.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/eddy_current_challenge_dev.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/eddy_current_challenge_hidden.h5`

---

## Previous State (Before 2026-03-09)
- **Category:** industrial_inspection
- **Sub-category pool:** industrial_inspection (ECT-specific override)
- **Algorithms:** [MUSIC, Born-ADMM, EddyNet, ECT-Former] (4 algorithms, up to 2024)
- **Dataset:** Shared `industrial_ndt_generated` (generate_ndt_phantom, 256x256)

## Current State (After 2026-03-09)
- **Category:** industrial_inspection
- **Sub-category pool:** `_VARIANT_OVERRIDES["eddy_current"]` (ECT-specific, 9 algorithms)
- **Algorithms:** [EC-Deconv, TV-EC, MUSIC-EC, DnCNN-EC, ECNN-Defect, TransEC, SwinEC, PhysEC, DiffEC]
- **Dataset:** Dedicated `eddy_current_generated` (generate_eddy_current_phantom, 64x64)
- **Runner:** identity (eddy current EM forward model handled by phantom generator)

## Verdict
All changes complete. Syntax validated. GCS datasets uploaded.
