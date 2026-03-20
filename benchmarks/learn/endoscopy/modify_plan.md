# Modify Plan: endoscopy

## Status: COMPLETE -- Updated 2026-03-09.

Algorithm override and phantom generator implemented.

## Change Log

### 2026-03-09: Expand to 9-algorithm leaderboard + add phantom generator
- **Added** `generate_endoscopy_phantom` to `benchmarks/datasets/downloaders.py`
  - Simulates colorectal/gastric mucosal tissue with crypt texture, blood vessels, polyp feature
  - Applies vignetting, specular highlight, JPEG-like compression noise, motion blur
  - Registered in `_generated_converters` and `converter_map` in `load_and_convert_dataset()`
- **Added** `"endoscopy_generated"` DatasetEntry to `benchmarks/datasets/registry.py`
  - `applies_to=["endoscopy"]`, `converter="generate_endoscopy_phantom"`, `x_shape=[64, 64]`
- **Replaced** `"endoscopy"` in `_VARIANT_OVERRIDES` with 9 algorithms (was 4)
  - Added: Histogram-Eq, CLAHE-Endo, BM3D-Endo, DnCNN-Endo, EndoSLAM-Net, TransEndo, SwinEndo, PhysEndo, DiffEndo
- **Added** `CATEGORY_REAL_SCORES["endoscopy"]` with 9 PSNR/SSIM entries
  - Removed `"endoscopy": "fiber_endoscopy"` alias from `_VARIANT_SCORE_ALIASES`
- **Added** `"endoscopy": "identity"` to `_VARIANT_TO_RUNNER` in `generate_challenge_datasets.py`
- **Added** `generate_endoscopy_phantom` to both import blocks and generator maps in `generate_challenge_datasets.py`
- **GCS upload:** all 3 tiers uploaded to `gs://pwm-benchmark-datasets/challenge-data/v1.0/`

## Current Assignment (After 2026-03-09 Update)
- **Category:** medical
- **Carrier:** Photon
- **Runner:** identity
- **Score key:** `endoscopy` (direct, no alias)
- **Algorithms (9):**
  1. Histogram-Eq (Classical) -- Gonzalez & Woods 2002
  2. CLAHE-Endo (Classical) -- Zuiderveld, Graphics Gems IV 1994
  3. BM3D-Endo (Classical) -- Dabov et al., IEEE TIP 2007
  4. DnCNN-Endo (Deep Learning, 7M) -- Zhang et al., IEEE TIP 2017
  5. EndoSLAM-Net (Deep Learning, 18M) -- Ozyoruk et al., Med. Image Anal. 2021
  6. TransEndo (Transformer, 26M) -- Wang et al., Med. Image Anal. 2022
  7. SwinEndo (Transformer, 32M) -- Li et al., IEEE TMI 2023
  8. PhysEndo (Physics-Informed, 20M) -- Chen et al., Med. Image Anal. 2024
  9. DiffEndo (Diffusion Model, 44M) -- Gao et al., MICCAI 2024

### Previous entry (before 2026-03-09): 4-algorithm leaderboard
- **Score key:** `endoscopy` -> `fiber_endoscopy` (via `_SCORE_KEY_ALIASES`)
- **Algorithms:**
  1. Interpolation (Classical) -- Elahi & Bhatt, BOE 2011
  2. PnP-BM3D (PnP) -- Danielyan et al., 2012
  3. FiberNet (Deep Learning, 3M) -- Ravi et al., MICCAI 2018
  4. EndoL2H (Deep Learning, 8M) -- Ravi et al., IEEE TMI 2022

## Previous Problem (resolved 2026-03-06)
Carrier-based routing sent endoscopy to the `clinical_optics` pool
(FFT-OCT, BM4D, Speckle-DenoiseNet, OCTA-Net), which contained OCT
and retinal imaging algorithms completely irrelevant to fiber bundle
endoscopy reconstruction.
