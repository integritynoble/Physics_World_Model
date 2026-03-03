# Modify Plan: confocal_endomicroscopy

## Current State

- **Category:** medical
- **Carrier:** Photon
- **Routing:** (medical, Photon) -> `clinical_optics` pool
- **Score key:** clinical_optics
- **Algorithms served:**
  1. FFT-OCT (Classical) -- Analytical baseline
  2. BM4D (PnP) -- Maggioni et al., IEEE TIP 2013
  3. Speckle-DenoiseNet (Deep Learning) -- Devalla et al., BOE 2019
  4. OCTA-Net (Transformer) -- Hybrid U-Net+Transformer, 2023

## Problem

The `clinical_optics` pool contains OCT-specific algorithms (FFT-OCT, Speckle-DenoiseNet, OCTA-Net) that are inappropriate for Confocal Laser Endomicroscopy (CLE). CLE is a fiber-bundle-based confocal fluorescence imaging technique used for real-time in vivo tissue microscopy. The reconstruction problem is fiber bundle pattern removal + deconvolution + mosaicking, not OCT spectral-domain processing.

- **FFT-OCT:** Spectral-domain OCT reconstruction via FFT. CLE does not use interferometry or spectral encoding. WRONG.
- **BM4D:** Generic volumetric denoiser. Acceptable as a general-purpose denoiser but not CLE-specific.
- **Speckle-DenoiseNet:** Designed for OCT speckle (coherent noise). CLE has photon shot noise and fiber honeycomb pattern, not OCT speckle. WRONG.
- **OCTA-Net:** OCT angiography network. Completely irrelevant to CLE. WRONG.

## Recommended Algorithms

CLE reconstruction involves: (1) fiber bundle pattern removal (honeycomb artifact), (2) PSF deconvolution, (3) super-resolution from fiber core spacing, (4) mosaicking for large-area imaging.

| Slot | Algorithm | Type | Reference | Rationale |
|------|-----------|------|-----------|-----------|
| Classical | Interpolation + Wiener | Classical | Elahi et al., J. Biomed. Opt. 2014 | Standard CLE pipeline: triangular interpolation to remove honeycomb pattern, then Wiener deconvolution |
| PnP | PnP-BM3D | PnP | Danielyan et al., 2012 | Plug-and-play prior with fiber pattern forward model |
| Deep Learning | FibreNet | Deep Learning | Shao et al., Med. Image Anal. 2019 | CNN trained specifically for fiber bundle image reconstruction |
| Transformer | Restormer | Transformer | Zamir et al., CVPR 2022 | General-purpose image restoration transformer applicable to CLE denoising/super-resolution |

## Required Code Changes

1. **`_algorithm_catalog.py`:** Add a CLE-specific entry to `_VARIANT_OVERRIDES` for `confocal_endomicroscopy`, or create a new sub-category routing rule.
2. **`_algorithm_catalog.py`:** Add real published scores for CLE algorithms to `CATEGORY_REAL_SCORES` if available.
3. **`_leaderboard_generator.py`:** No changes needed (score bands from `clinical_optics` PSNR range are reasonable for CLE).
