# Modify Plan: confocal_livecell

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

- **Richardson-Lucy:** The standard deconvolution algorithm for fluorescence microscopy. Directly applicable to confocal PSF deconvolution. CORRECT.
- **PnP-FISTA:** Plug-and-play framework with FISTA acceleration. Suitable for microscopy image restoration with learned priors. CORRECT.
- **CARE:** Content-Aware Image Restoration -- literally designed for and validated on live-cell confocal microscopy data (Weigert et al., Nature Methods 2018). The flagship application was denoising low-SNR confocal images. PERFECT FIT.
- **Restormer:** State-of-the-art transformer for image restoration, applicable to microscopy denoising/deconvolution. CORRECT.

No code changes needed.
