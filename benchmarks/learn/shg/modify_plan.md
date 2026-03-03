# Modify Plan -- shg

## Current State

- **Category:** microscopy
- **Carrier:** Photon
- **Routing:** No carrier routing for `("microscopy", "Photon")` -> falls to `_CATEGORY_ALGORITHMS["microscopy"]`
- **Score key:** microscopy
- **Algorithms assigned:**
  1. Richardson-Lucy (Classical) -- Richardson 1972 / Lucy 1974
  2. PnP-FISTA (PnP) -- Bai et al., 2020
  3. CARE (Deep Learning) -- Weigert et al., Nat. Methods 2018
  4. Restormer (Transformer) -- Zamir et al., CVPR 2022

## Assessment

**Appropriate: YES.**

Second Harmonic Generation (SHG) microscopy is a nonlinear optical microscopy technique that generates contrast from non-centrosymmetric structures (e.g., collagen, myosin). The acquired images are degraded by the point spread function (PSF) and photon noise, just like other optical microscopy modalities. The reconstruction problem is image deconvolution and denoising.

- **Richardson-Lucy**: The standard iterative deconvolution algorithm for optical microscopy. Directly applicable to SHG image deconvolution.
- **PnP-FISTA**: Plug-and-play with FISTA for microscopy deconvolution. Appropriate.
- **CARE** (Content-Aware Image Restoration): Published specifically for microscopy image restoration including nonlinear microscopy modalities. Directly applicable.
- **Restormer**: Transformer-based image restoration. Applicable to any microscopy denoising/deconvolution task.

The microscopy pool is well-suited for SHG. SHG images are processed with the same deconvolution and denoising tools as fluorescence microscopy images.

## Plan

No code changes needed.
