# Modify Plan: desi

## Current State

- **Category:** spectroscopy
- **Carrier:** Ion
- **Routing:** Direct to `spectroscopy` pool (no carrier routing override)
- **Score key:** spectroscopy
- **Algorithms served:**
  1. SG-ALS (Classical) -- Savitzky-Golay + ALS baseline
  2. PnP-DnCNN (PnP) -- Zhang et al., 2017
  3. CDAE (Deep Learning) -- Zhang et al., Sensors 2024
  4. Cascade-UNet (Transformer) -- Physics-informed UNet, 2025

## Assessment

The spectroscopy pool is a reasonable but imperfect fit for DESI Mass Spectrometry Imaging. DESI (Desorption Electrospray Ionization) MSI produces spatially-resolved mass spectra, creating hyperspectral-like datacubes (x, y, m/z). The reconstruction challenge involves both spectral processing (baseline correction, peak identification, denoising) and spatial image reconstruction (ion image denoising, spatial deconvolution).

- **SG-ALS (Savitzky-Golay + Asymmetric Least Squares):** Standard spectral preprocessing -- smoothing and baseline correction. Applicable to mass spectra. ACCEPTABLE.
- **PnP-DnCNN:** Generic denoiser in a plug-and-play framework. Can be applied to either spectral or spatial dimensions. ACCEPTABLE.
- **CDAE (Convolutional Denoising Autoencoder):** For spectral denoising. ACCEPTABLE.
- **Cascade-UNet:** Physics-informed UNet. Generic but applicable. ACCEPTABLE.

More domain-specific MSI algorithms exist (e.g., MCR-ALS for multivariate curve resolution, NMF for spectral unmixing, msImpute for missing value imputation, SCiLS for spatial segmentation), but the current pool covers the spectral denoising/reconstruction aspect adequately.

No code changes needed.
