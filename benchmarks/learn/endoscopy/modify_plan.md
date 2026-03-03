# Modify Plan: endoscopy

## Status: COMPLETE -- No further code changes needed.

Algorithm override implemented in `_VARIANT_OVERRIDES` within
`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`.

## Current Assignment (After Fix)
- **Category:** medical
- **Carrier:** Photon
- **Score key:** `endoscopy` -> `fiber_endoscopy` (via `_SCORE_KEY_ALIASES`)
- **Algorithms:**
  1. Interpolation (Classical) -- Elahi & Bhatt, BOE 2011
  2. PnP-BM3D (PnP) -- Danielyan et al., 2012
  3. FiberNet (Deep Learning, 3M) -- Ravi et al., MICCAI 2018
  4. EndoL2H (Deep Learning, 8M) -- Ravi et al., IEEE TMI 2022

## What Was Changed
- Added `"endoscopy"` to `_VARIANT_OVERRIDES` with 4 fiber-bundle-appropriate algorithms
- Added `"fiber_endoscopy"` to `CATEGORY_REAL_SCORES` with representative PSNR/SSIM values
- Added `"endoscopy": "fiber_endoscopy"` alias in `_SCORE_KEY_ALIASES`

## Previous Problem
Carrier-based routing sent endoscopy to the `clinical_optics` pool
(FFT-OCT, BM4D, Speckle-DenoiseNet, OCTA-Net), which contained OCT
and retinal imaging algorithms completely irrelevant to fiber bundle
endoscopy reconstruction.
