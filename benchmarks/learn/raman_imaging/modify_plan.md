# Modify Plan -- raman_imaging

## Current State

- **Category:** spectroscopy
- **Carrier:** Photon
- **Routing:** No carrier routing for `("spectroscopy", "Photon")` -> falls to `_CATEGORY_ALGORITHMS["spectroscopy"]`
- **Score key:** spectroscopy
- **Algorithms assigned:**
  1. SG-ALS (Classical) -- Savitzky-Golay + ALS baseline
  2. PnP-DnCNN (PnP) -- Zhang et al., 2017
  3. CDAE (Deep Learning) -- Zhang et al., Sensors 2024
  4. Cascade-UNet (Transformer) -- Physics-informed UNet, 2025

## Assessment

**Appropriate: YES.**

Raman imaging/microscopy collects spatially-resolved Raman spectra. The key computational challenges are baseline removal (fluorescence background subtraction), denoising of weak Raman signals, and spectral unmixing. The spectroscopy algorithm pool is well-suited:

- **SG-ALS** (Savitzky-Golay + Asymmetric Least Squares): Standard baseline correction and smoothing for Raman spectra. This is the textbook classical approach.
- **PnP-DnCNN**: Plug-and-play denoising is applicable to Raman spectral image denoising.
- **CDAE** (Convolutional Denoising Autoencoder): Published specifically for Raman spectral denoising.
- **Cascade-UNet**: Physics-informed network for spectral reconstruction; applicable to Raman.

All four algorithms are directly relevant to Raman spectroscopy reconstruction and denoising.

## Plan

No code changes needed.
