# Modify Plan: ftir_imaging

## Current State

- **Category:** spectroscopy
- **Carrier:** IR
- **Score key:** spectroscopy
- **Algorithms assigned:**
  1. SG-ALS (Classical) -- Savitzky-Golay + ALS baseline
  2. PnP-DnCNN (PnP) -- Zhang et al., 2017
  3. CDAE (Deep Learning) -- Zhang et al., Sensors 2024
  4. Cascade-UNet (Transformer) -- Physics-informed UNet, 2025

## Assessment

**Appropriate: YES**

FTIR spectroscopic imaging is correctly placed in the spectroscopy category.
The inverse problem involves recovering clean spectra from noisy interferograms,
which is fundamentally a spectral denoising/deconvolution task.

- **SG-ALS** (Savitzky-Golay smoothing + Asymmetric Least Squares baseline
  correction) is the standard classical preprocessing for spectroscopic data.
- **PnP-DnCNN** is a reasonable generic PnP denoiser applied to spectral data.
- **CDAE** (Convolutional Denoising Autoencoder) from Zhang et al. 2024 is a
  published spectroscopy-specific deep learning method.
- **Cascade-UNet** is a physics-informed architecture appropriate for
  spectroscopic reconstruction.

All algorithms are appropriate for FTIR spectroscopic imaging reconstruction.

## Code Changes Needed

No code changes needed.
