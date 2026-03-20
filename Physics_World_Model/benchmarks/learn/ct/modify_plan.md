# Modify Plan: CT (X-ray Computed Tomography)

**Created:** 2026-03-03
**Last Updated:** 2026-03-09
**Status:** Done

## Changes Required

### 1. Fix FBP Citation (ERROR) — Done 2026-03-03

**File:** `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
Changed FBP source to `"Kak & Slaney, IEEE Press 1988"` in both `_VARIANT_OVERRIDES` and `CATEGORY_REAL_SCORES`.

### 2. Fix DuDoTrans Citation Venue (ERROR) — Done 2026-03-03

**File:** `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
Changed to `"Wang et al., MLMIR 2022"` in both sections.

### 3. Complete PnP-ADMM Citation (WARNING) — Done 2026-03-03

**File:** `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
Changed to `"Venkatakrishnan et al., IEEE GlobalSIP 2013"` in both sections.

### 4. Fix Dataset Size Description (WARNING) — Superseded 2026-03-09

Superseded by the full CT modality overhaul below.

---

## Change Log: 2026-03-09 — Full CT Modality Overhaul

### A. Add Shepp-Logan Phantom Generator

**File:** `benchmarks/datasets/downloaders.py`
Added `generate_ct_phantom()` function that generates 64×64 Shepp-Logan-style phantoms:
- `x_true`: ellipsoidal body outline, skull shell, liver, lung-left, lung-right, spine regions
- `y`: sinogram via Radon transform (128 angles), Beer-Lambert + Poisson noise at I₀=1e5, log-normalised to [0,1]
- `H_ideal`: identity matrix (Radon operator is implicit)
- `metadata`: modality, n_angles, detector_pixels, source_to_detector_mm
- Uses `skimage.transform.radon` when available; falls back to `scipy.ndimage.rotate` projection loop
- Registered in both `_generated_converters` and `converter_map` in `load_and_convert_dataset()`

### B. Add DatasetEntry

**File:** `benchmarks/datasets/registry.py`
Added `"ct_generated"` entry with `converter="generate_ct_phantom"`, `applies_to=["ct"]`,
`x_shape=[64, 64]`, `source_type="generated"`, `license="synthetic"`.

### C. Replace Algorithm Overrides (9 algorithms)

**File:** `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
Replaced old 10-algorithm CT entry in `_VARIANT_OVERRIDES` with:
  FBP, TV-ADMM, SART, FBPConvNet, RED-CNN, DuDoRNet, TransCT, CTformer, DiffusionMBIR

Replaced old CT entry in `CATEGORY_REAL_SCORES` with 9 entries matching the new algorithm list.

### D. Add Runner Routing

**File:** `platform/scripts/generate_challenge_datasets.py`
Added `"ct": "radon"` to `_VARIANT_TO_RUNNER`.
Added `generate_ct_phantom` to both import blocks and both generator maps
(`_GENERATOR_MAP` in `_resolve_ground_truth()` and `gen_map` in the gallery generator).

### E. GCS Dataset Regeneration

Ran: `python3 scripts/generate_challenge_datasets.py --variant ct --gcs-only`
Uploaded 3 HDF5 files to `gs://pwm-benchmark-datasets/challenge-data/v1.0/`:
- `ct_challenge_public.h5`
- `ct_challenge_dev.h5`
- `ct_challenge_hidden.h5`
