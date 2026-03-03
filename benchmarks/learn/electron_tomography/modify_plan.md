# Modify Plan: electron_tomography

## Status: COMPLETE -- No further code changes needed.

Algorithm override implemented in `_VARIANT_OVERRIDES` within
`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`.

## Current Assignment (After Fix)
- **Category:** electron_microscopy
- **Carrier:** Electron
- **Score key:** `electron_tomography` (direct key in `CATEGORY_REAL_SCORES`)
- **Algorithms:**
  1. WBP (Classical) -- Radermacher, 1988
  2. SIRT (Classical) -- Gilbert, J. Theor. Biol. 1972
  3. IsoNet (Deep Learning, 8M) -- Liu et al., Nat. Commun. 2022
  4. CryoAI (Deep Learning, 10M) -- Levy et al., arXiv 2022

## What Was Changed
- Removed `electron_tomography` from `_CRYO_EM_VARIANTS`
- Added `"electron_tomography"` to `_VARIANT_OVERRIDES` with 4 tilt-series reconstruction algorithms
- Added `"electron_tomography"` to `CATEGORY_REAL_SCORES` with representative PSNR/SSIM values

## Previous Problem
The variant was in `_CRYO_EM_VARIANTS`, receiving single-particle cryo-EM
algorithms (RELION, cryoSPARC, cryoDRGN, CryoTransformer). While these
share the electron microscopy category, single-particle tools do NOT
perform tilt-series tomographic reconstruction.
