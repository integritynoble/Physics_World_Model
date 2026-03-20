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

## 2026-03-06 Comprehensive Check Update

- Physics: y(nu) = sum_k c_k * s_k(nu) * A(nu) + b_fluorescence + n_shot; fluorescence background 10^3-10^6 x stronger than Raman
- Key mismatch: instrument response A(nu), fluorescence background model, laser power uniformity, reference spectra purity
- GCS datasets: 3 tiers confirmed
- Algorithm pool: PASS — SG-ALS (baseline removal), SVD (dimensionality reduction), CDAE (deep denoising), SpectraFormer (transformer) cover the full pipeline
- Note: Full catalog includes SpectraFormer (2024) and DiffusionSpectra as state-of-the-art
