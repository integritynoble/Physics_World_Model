# Modify Plan -- lensless

## Current State (Updated 2026-03-03)

- **Category:** computational_photography
- **Carrier:** Photon
- **Score key:** computational_photography
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["lensless"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. Wiener-ADMM (Classical) -- Antipa et al., Optica 2018
  2. PnP-ADMM (PnP) -- Monakhova et al., Opt. Express 2019
  3. FlatNet (Deep Learning) -- Khan et al., IEEE TPAMI 2020
  4. Uformer (Transformer) -- Wang et al., CVPR 2022

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the computational_photography pool where
HDR-CNN (an HDR tone-mapping network) was completely inappropriate for
lensless deconvolution. The new set includes the DiffuserCam team's
Wiener-ADMM baseline, PnP with learned denoisers, the landmark FlatNet
physics-informed architecture, and a transformer restoration network.

## Changes Applied

- Added `_VARIANT_OVERRIDES["lensless"]` with four lensless-appropriate algorithms
- Wiener-ADMM: standard DiffuserCam reconstruction with TV regularization
- PnP-ADMM: plug-and-play with learned denoisers for lensless imaging
- FlatNet: physics-informed end-to-end network incorporating the PSF
- Uformer: transformer-based image restoration (general but applicable)

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
