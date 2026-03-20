# Modify Plan: eht_imaging

## Change Log

### 2026-03-09

**Changes made:**
- Added `generate_eht_imaging_phantom()` to `benchmarks/datasets/downloaders.py`:
  - 64x64 float32 accretion disk brightness phantom (ring + Gaussian shadow + Doppler hot spot)
  - EHT/VLBI forward model: sparse u-v mask (~10 baselines, ~20% coverage), complex Gaussian thermal noise, back-projection dirty image
  - Registered in `_generated_converters` and `converter_map` inside `load_and_convert_dataset()`
- Added `"eht_imaging_generated"` DatasetEntry to `benchmarks/datasets/registry.py`
- Added `_VARIANT_OVERRIDES["eht_imaging"]` to `_algorithm_catalog.py` with 9 VLBI-specific algorithms
- Added `CATEGORY_REAL_SCORES["eht_imaging"]` to `_algorithm_catalog.py` with 9 PSNR/SSIM entries
- Added `"eht_imaging": "identity"` to `_VARIANT_TO_RUNNER` in `generate_challenge_datasets.py`
- Added `generate_eht_imaging_phantom` to both generator import blocks and generator maps in `generate_challenge_datasets.py`
- Generated and uploaded all 3 tiers to `gs://pwm-benchmark-datasets/challenge-data/v1.0/`

**Algorithms:** CLEAN-VLBI, MEM-VLBI, RESOLVE, eht-imaging, SMILI, TransVLBI, RadioFormer, PhysVLBI, DiffVLBI

---

## Current Assignment (updated 2026-03-09, was 2026-03-06)
- **Category:** experimental_science
- **Carrier:** RF
- **Score key:** experimental_science
- **Algorithms (11 total from experimental_science pool):**
  1. Tikhonov (Classical) -- Tikhonov, Doklady 1963
  2. Wiener Filter (Classical) -- Wiener filtering baseline
  3. Matched Filter (Classical) -- Optimal linear filter
  4. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  5. PnP-ADMM (PnP) -- ADMM + denoiser prior
  6. ResUNet (Deep Learning) -- Residual U-Net baseline
  7. Domain-Adapted-CNN (Deep Learning) -- Domain adaptation CNN
  8. SwinIR (Vision Transformer) -- Liang et al., ICCVW 2021
  9. ExpFormer (Vision Transformer) -- Experimental science transformer, 2024
  10. DiffusionExperimental (Diffusion) -- Zhang et al., 2024
  11. ScoreExperimental (Score-based) -- Wei et al., 2025

**Status:** PASS — check.md written 2026-03-06

## Assessment

The algorithm assignment is appropriate. All four algorithms are well-known radio
interferometric imaging methods used in the VLBI/EHT community:

- **CLEAN** (Hogbom, 1974) is the standard deconvolution algorithm for radio
  interferometry and the baseline for all VLBI imaging.
- **AIRI** (Terris et al., MNRAS 2022) is a learned-regularization PnP method
  designed for radio interferometric imaging.
- **R2D2** (Aghabiglou et al., ApJS 2024) is a residual-to-residual deep neural
  network trained for radio image reconstruction.
- **PRIMO** (Medeiros et al., ApJL 2023) was used to produce the sharpened M87
  black hole image from EHT data.

The astronomy category score ranges and mismatch descriptions (per-antenna
gain/phase, atmospheric phase screen) are appropriate for EHT.

## Verdict

No code changes needed.
