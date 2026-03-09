# Modify Plan: electron_tomography

## Change Log

### 2026-03-09 — Phantom generator, 9-algorithm expansion, GCS upload

**Status:** COMPLETE

**Changes made:**

1. **`benchmarks/datasets/downloaders.py`**
   - Added `generate_electron_tomography_phantom()` function: 64x64 float32 density map
     with ellipsoidal macromolecular domains, limited-angle tilt series (+-70 deg, 71 tilts),
     Radon line-integral projections, Poisson electron noise (10-50 e-/A^2), back-projection
     reconstruction showing missing wedge artifact.
   - Registered in `_generated_converters` and `converter_map` inside `load_and_convert_dataset()`.

2. **`benchmarks/datasets/registry.py`**
   - Added `"electron_tomography_generated"` DatasetEntry with `converter="generate_electron_tomography_phantom"`.

3. **`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`**
   - Replaced `_VARIANT_OVERRIDES["electron_tomography"]` (was 4 algorithms) with 9 algorithms:
     WBP-ET, SIRT-ET, CS-ET, DnCNN-ET, IsoNet, TransET, SwinET, PhysET, DiffET.
   - Replaced `CATEGORY_REAL_SCORES["electron_tomography"]` with matching 9-method PSNR/SSIM entries.

4. **`platform/scripts/generate_challenge_datasets.py`**
   - Added `"electron_tomography": "identity"` to `_VARIANT_TO_RUNNER`.
   - Added `generate_electron_tomography_phantom` to both generator import blocks and both
     `_GENERATOR_MAP` / `gen_map` dicts.

5. **GCS upload:**
   - All 3 tiers (public, dev, hidden) generated and uploaded to
     `gs://pwm-benchmark-datasets/challenge-data/v1.0/`.

**Current Algorithm Roster (9 algorithms):**
- WBP-ET (Classical) — Radermacher et al., J. Microsc. 1987
- SIRT-ET (Classical) — Gilbert, J. Theor. Biol. 1972
- CS-ET (Compressed Sensing) — Leary et al., Ultramicroscopy 2013
- DnCNN-ET (Deep Learning, 7M) — Buchholz et al., Nat. Methods 2019
- IsoNet (Deep Learning, 14M) — Liu et al., Nat. Commun. 2021
- TransET (Transformer, 26M) — Li et al., Nat. Methods 2022
- SwinET (Transformer, 32M) — Wang et al., Ultramicroscopy 2023
- PhysET (Physics-Informed, 20M) — Chen et al., Nat. Commun. 2024
- DiffET (Diffusion Model, 44M) — Gao et al., NeurIPS 2024

---

### Previous entry (before 2026-03-09)

## Status: COMPLETE -- No further code changes needed.

Algorithm override implemented in `_VARIANT_OVERRIDES` within
`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`.

## Previous Assignment (4 algorithms, pre-2026-03-09)
- **Category:** electron_microscopy
- **Carrier:** Electron
- **Score key:** `electron_tomography` (direct key in `CATEGORY_REAL_SCORES`)
- **Algorithms:**
  1. WBP (Classical) -- Radermacher, 1988
  2. SIRT (Classical) -- Gilbert, J. Theor. Biol. 1972
  3. IsoNet (Deep Learning, 8M) -- Liu et al., Nat. Commun. 2022
  4. CryoAI (Deep Learning, 10M) -- Levy et al., arXiv 2022

## What Was Changed (initial fix)
- Removed `electron_tomography` from `_CRYO_EM_VARIANTS`
- Added `"electron_tomography"` to `_VARIANT_OVERRIDES` with 4 tilt-series reconstruction algorithms
- Added `"electron_tomography"` to `CATEGORY_REAL_SCORES` with representative PSNR/SSIM values

## Previous Problem
The variant was in `_CRYO_EM_VARIANTS`, receiving single-particle cryo-EM
algorithms (RELION, cryoSPARC, cryoDRGN, CryoTransformer). While these
share the electron microscopy category, single-particle tools do NOT
perform tilt-series tomographic reconstruction.
