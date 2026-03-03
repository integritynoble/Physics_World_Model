# Modify Plan -- libs

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

**Acceptable.** LIBS (Laser-Induced Breakdown Spectroscopy) imaging involves spectral analysis -- identifying elemental composition from emission spectra. The spectroscopy pool provides algorithms for spectral signal processing:

- **SG-ALS** (Savitzky-Golay smoothing + Asymmetric Least Squares baseline correction) is a standard preprocessing/reconstruction method for spectroscopic data. Appropriate for LIBS baseline correction and smoothing.
- **PnP-DnCNN** is a general-purpose denoising approach applicable to spectral data denoising.
- **CDAE** (Convolutional Denoising Autoencoder) for spectral data is reasonable.
- **Cascade-UNet** as a physics-informed network for spectral reconstruction is plausible.

While LIBS has some unique aspects (multi-elemental mapping, plasma emission dynamics), the core reconstruction task of spectral signal recovery and denoising is well-served by the spectroscopy pool. More LIBS-specific methods exist (e.g., calibration-free LIBS, CF-LIBS based on Boltzmann plots) but these are more about quantification than image/signal reconstruction, which is the benchmark focus.

## Recommendation

No code changes needed.
