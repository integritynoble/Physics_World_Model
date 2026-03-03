# Modify Plan: coronagraphy

## Current State

- **Category:** astronomy
- **Carrier:** Photon
- **Routing:** Direct to `astronomy` pool (no carrier routing override for Photon astronomy)
- **Score key:** astronomy
- **Algorithms served:**
  1. CLEAN (Classical) -- Hogbom, A&AS 1974
  2. AIRI (PnP) -- Terris et al., MNRAS 2022
  3. R2D2 (Deep Learning) -- Aghabiglou et al., ApJS 2024
  4. PRIMO (Deep Learning) -- Medeiros et al., ApJL 2023

## Problem

The `astronomy` pool is tailored for radio interferometric imaging (aperture synthesis). Stellar coronagraphy is a fundamentally different problem: high-contrast imaging that suppresses starlight to reveal faint companions (exoplanets, circumstellar disks). The reconstruction task is point-source detection and separation, not visibility-to-image synthesis.

- **CLEAN:** Radio interferometry deconvolution (gridded visibilities -> sky image). Coronagraphy does not use visibility data. WRONG.
- **AIRI:** AI for Radio Interferometric imaging. Explicitly radio-only. WRONG.
- **R2D2:** Residual-to-Residual DNN for radio interferometry. WRONG.
- **PRIMO:** Principal-component Interferometric Modeling for the Event Horizon Telescope. Radio interferometry for black hole imaging. WRONG.

None of these four algorithms are applicable to coronagraphy.

## Recommended Algorithms

Coronagraphic image processing involves angular/spectral differential imaging to subtract the stellar PSF and reveal faint companions.

| Slot | Algorithm | Type | Reference | Rationale |
|------|-----------|------|-----------|-----------|
| Classical | cADI (classical ADI) | Classical | Marois et al., ApJ 2006 | Angular Differential Imaging -- the foundational technique for high-contrast imaging; rotates and subtracts median PSF |
| PnP | KLIP (Karhunen-Loeve Image Projection) | PnP | Soummer et al., ApJ 2012 | PCA-based PSF subtraction using Karhunen-Loeve eigenimages; standard in the field (~1,000 citations) |
| Deep Learning | SODINN | Deep Learning | Gomez Gonzalez et al., A&A 2018 | Supervised deep learning for exoplanet detection in ADI sequences |
| Transformer | ANDROMEDA | Model-fitting | Cantalloube et al., A&A 2015 | Maximum-likelihood model-fitting approach for point source detection in coronagraphic data |

## Required Code Changes

1. **`_algorithm_catalog.py`:** Add `coronagraphy` to `_VARIANT_OVERRIDES` with the four coronagraphy-specific algorithms above.
2. **`_algorithm_catalog.py`:** Add coronagraphy-specific real scores to `CATEGORY_REAL_SCORES` (contrast curves rather than PSNR if possible, else use SNR-based scores).
3. **Consider:** Adding a `("astronomy", "Photon")` entry to `_CARRIER_ROUTING` that maps to a `high_contrast_imaging` pool, so other optical astronomy modalities (e.g., adaptive_optics, lucky_imaging) can share it.
