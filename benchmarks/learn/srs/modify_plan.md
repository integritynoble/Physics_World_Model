# Modify Plan: srs

## Current State
- **Category:** spectroscopy
- **Carrier:** Photon
- **Score key:** spectroscopy
- **Algorithms:**
  1. SG-ALS (Classical) -- Savitzky-Golay + ALS baseline
  2. PnP-DnCNN (PnP) -- Zhang et al., 2017
  3. CDAE (Deep Learning) -- Zhang et al., Sensors 2024
  4. Cascade-UNet (Transformer) -- Physics-informed UNet, 2025

## Assessment

The algorithms are reasonable for SRS (Stimulated Raman Scattering) microscopy, though the match is imperfect. The spectroscopy pool focuses on spectral processing (baseline correction, denoising, unmixing), which partially applies to SRS:

- **SG-ALS** (Savitzky-Golay smoothing + Asymmetric Least Squares baseline) is a standard spectral preprocessing method applicable to Raman spectra, including SRS.
- **PnP-DnCNN** is a generic denoising method that can be applied to SRS images.
- **CDAE** (Convolutional Denoising Autoencoder) is designed for spectral denoising, applicable to SRS.
- **Cascade-UNet** is a physics-informed architecture that could handle SRS spectral unmixing.

SRS microscopy has a dual nature: it produces both spatial images and spectral information. The spectroscopy pool addresses the spectral dimension well. However, SRS-specific challenges (non-resonant background subtraction, lock-in phase optimization) are not directly captured. More SRS-specific algorithms might include:
- MIA (Multivariate Image Analysis) for spectral unmixing
- MCR-ALS (Multivariate Curve Resolution) for component separation

Overall, the spectroscopy pool is a reasonable approximation. The specific SRS mismatch parameters (lock-in phase error, cross-phase modulation, RIN) correctly capture the domain-specific challenges.

No code changes needed.

## Files to Modify
None.
