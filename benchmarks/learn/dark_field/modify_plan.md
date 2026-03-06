# Modify Plan: dark_field

**Date:** 2026-03-06

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

The generic microscopy pool is a reasonable fit for dark-field microscopy. Dark-field microscopy is an optical contrast technique where the unscattered beam is blocked and only scattered light forms the image. The reconstruction/restoration task is primarily denoising (dark-field images are typically low-SNR due to the weak scattered signal) and deconvolution of the annular illumination PSF.

- **Richardson-Lucy:** Standard PSF deconvolution applicable to dark-field image restoration with the annular illumination PSF. ACCEPTABLE.
- **PnP-FISTA:** Plug-and-play framework for image restoration with forward model regularization. ACCEPTABLE.
- **CARE:** Content-Aware Image Restoration for fluorescence microscopy; the architecture generalizes to dark-field scattering image denoising. ACCEPTABLE.
- **Restormer:** General-purpose image restoration transformer. ACCEPTABLE.

### Scope Clarification

The benchmark covers **optical dark-field microscopy** (scattered-light contrast in a standard optical microscope with annular illumination stop). This is distinct from:
1. **Grating-based X-ray dark-field** (Talbot-Lau interferometry) — requires phase-stepping retrieval algorithms
2. **Dark-field electron microscopy** — uses diffracted electrons to form contrast

The optical dark-field restoration problem is structurally similar to fluorescence microscopy deconvolution (PSF + noise), making the microscopy pool appropriate. More domain-specific algorithms would be annular PSF deconvolution (Ring-DAS) or scattering-aware CNNs, but these represent enhancements rather than corrections.

## Plan

No code changes needed. The microscopy pool is adequate for optical dark-field image restoration.

**Priority:** NONE — no changes needed for the optical dark-field case. If the benchmark were extended to X-ray dark-field CT, a different algorithm pool would be required.
