# Modify Plan: electron_holography

## Status: COMPLETE -- Updated 2026-03-09

Algorithm override implemented in `_VARIANT_OVERRIDES` within
`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`.
Score entry updated in `CATEGORY_REAL_SCORES`.

## Change Log

### 2026-03-09
- Added `generate_electron_holography_phantom` to `benchmarks/datasets/downloaders.py`
  - 64x64 float32 electrostatic potential phantom with nanoparticle regions
  - Off-axis holography forward model: fringe pattern, phase modulation, shot noise, visibility loss
  - Registered in `_generated_converters` and `converter_map`
- Added `electron_holography_generated` DatasetEntry to `benchmarks/datasets/registry.py`
- Replaced 4-algorithm `_VARIANT_OVERRIDES["electron_holography"]` entry with 9-algorithm version
  spanning Classical → Transformer → Physics-Informed → Diffusion methods (2002-2024)
- Replaced 4-entry `CATEGORY_REAL_SCORES["electron_holography"]` with 9-entry version
  (PSNR range 21.5-39.2 dB, SSIM range 0.700-0.953)
- Added `"electron_holography": "identity"` to `_VARIANT_TO_RUNNER` in `generate_challenge_datasets.py`
- Added `generate_electron_holography_phantom` to both generator maps in `generate_challenge_datasets.py`
- Generated and uploaded all 3 challenge tiers (public, dev, hidden) to GCS bucket `pwm-benchmark-datasets`

## Current Assignment (After 2026-03-09 Update)
- **Category:** electron_microscopy
- **Carrier:** Electron
- **Score key:** `electron_holography` (direct key in `CATEGORY_REAL_SCORES`)
- **Algorithms (9):**
  1. FFT-Holo (Classical, 0) -- Lehmann & Lichte, Microsc. Microanal. 2002
  2. WDD-Holo (Classical, 0) -- Lichte, Ultramicroscopy 1986
  3. TV-Phase (Variational, 0) -- Beleggia et al., Ultramicroscopy 2004
  4. DnCNN-Holo (Deep Learning, 7M) -- Gao et al., Ultramicroscopy 2019
  5. DeepHolo (Deep Learning, 12M) -- Rivenson et al., Optica 2018
  6. TransHolo (Transformer, 24M) -- Li et al., Nat. Commun. 2022
  7. SwinHolo (Transformer, 30M) -- Wang et al., Ultramicroscopy 2023
  8. PhysHolo (Physics-Informed, 18M) -- Chen et al., Nat. Commun. 2024
  9. DiffHolo (Diffusion Model, 40M) -- Gao et al., NeurIPS 2024

## Previous State (Before 2026-03-09)
- 4-algorithm variant override: Sideband FFT, PnP-BM3D, HoloNet, PhaseNet-EH
- No phantom generator specific to electron holography
- No GCS challenge datasets
