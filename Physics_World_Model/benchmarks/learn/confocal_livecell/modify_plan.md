# Modify Plan: confocal_livecell

**Date:** 2026-03-06

## Current State

- **Category:** microscopy
- **Carrier:** Photon
- **Routing:** Direct to `microscopy` pool (no carrier routing override)
- **Score key:** microscopy
- **Algorithms served:**
  1. Richardson-Lucy (Classical) -- Richardson 1972 / Lucy 1974
  2. PnP-FISTA (PnP) -- Bai et al., 2020
  3. CARE (Deep Learning) -- Weigert et al., Nat. Methods 2018
  4. Restormer (Transformer) -- Zamir et al., CVPR 2022

## Assessment

All four algorithms are appropriate for confocal live-cell microscopy:

- **Richardson-Lucy:** The standard deconvolution algorithm for fluorescence microscopy. Directly applicable to confocal PSF deconvolution in live-cell context. CORRECT.
- **PnP-FISTA:** Plug-and-play framework with FISTA acceleration. Suitable for microscopy image restoration with learned priors. Handles Poisson-Gaussian noise model. CORRECT.
- **CARE:** Content-Aware Image Restoration — the primary application demonstrated in the Nature Methods 2018 paper was denoising low-SNR live-cell confocal images of fluorescently labeled mitochondria, microtubules, and ER. The paper showed 60-fold photon budget reduction with CARE reconstruction. PERFECT FIT.
- **Restormer:** State-of-the-art transformer for image restoration, applicable to microscopy denoising/deconvolution. CORRECT.

### Distinction from confocal_3d

The confocal_livecell variant is functionally similar to confocal_3d (same algorithm pool) but represents a different use case:
- **confocal_livecell**: 2D time-series with photobleaching, temporal dynamics, extreme photon limitation (5–50 photons/pixel)
- **confocal_3d**: 3D static z-stack with moderate photon budget, depth-dependent PSF

The same algorithm pool is appropriate for both, but the dataset phantoms differ in dimensionality and SNR characteristics.

## Plan

No code changes needed. The microscopy pool is equally excellent for live-cell confocal as for 3D confocal z-stacks.

**Priority:** NONE — no changes needed.
