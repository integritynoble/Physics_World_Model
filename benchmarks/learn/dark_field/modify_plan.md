# Modify Plan: dark_field

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

The generic microscopy pool is a reasonable fit for dark-field microscopy. Dark-field microscopy is an optical contrast technique where the unscattered beam is blocked and only scattered light forms the image. The reconstruction/restoration task is primarily denoising (dark-field images are typically low-SNR) and deconvolution.

- **Richardson-Lucy:** Standard deconvolution applicable to dark-field PSF deconvolution. ACCEPTABLE.
- **PnP-FISTA:** Plug-and-play framework suitable for dark-field image restoration. ACCEPTABLE.
- **CARE:** Content-Aware Image Restoration for fluorescence microscopy, but the architecture generalizes to dark-field denoising. ACCEPTABLE.
- **Restormer:** General-purpose image restoration transformer. ACCEPTABLE.

While more domain-specific algorithms exist (e.g., dark-field specific scattering models, Fourier ptychographic dark-field reconstruction), the generic microscopy pool is adequate since dark-field restoration is fundamentally a denoising + deconvolution problem similar to other optical microscopy modalities.

No code changes needed.
