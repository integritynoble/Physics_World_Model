# Modify Plan: confocal_endomicroscopy

## Current State

- **Category:** medical
- **Carrier:** Photon
- **Routing:** Override (if implemented) or (medical, Photon) -> `clinical_optics` pool [WRONG]
- **Score key:** clinical_optics or confocal_endomicroscopy (needs override)
- **Algorithms served (correct, via override):**
  1. Interpolation (Classical) -- Elahi et al., J. Biomed. Opt. 16, 026003 (2011)
  2. PnP-BM3D (PnP) -- Danielyan et al., IEEE TIP 21, 1322 (2012)
  3. FiberNet (Deep Learning) -- Shao et al., Med. Image Anal. 72, 102065 (2019)
  4. EndoL2H (Deep Learning) -- Ravi et al., IEEE TMI 42, 1488 (2022)

## Problem (Historical)

The `clinical_optics` pool contained OCT-specific algorithms (FFT-OCT, Speckle-DenoiseNet, OCTA-Net) that were inappropriate for Confocal Laser Endomicroscopy (CLE). CLE is a fiber-bundle-based confocal fluorescence imaging technique, not OCT:

- **FFT-OCT:** Spectral-domain OCT reconstruction. CLE does not use interferometry. WRONG.
- **Speckle-DenoiseNet:** Designed for OCT speckle. CLE has shot noise and honeycomb pattern, not OCT speckle. WRONG.
- **OCTA-Net:** OCT angiography. Completely irrelevant to CLE. WRONG.

## Resolution

A `_VARIANT_OVERRIDES` entry for `confocal_endomicroscopy` is required in `_algorithm_catalog.py` pointing to CLE-specific algorithms. The correct algorithm set is:

| Slot | Algorithm | Type | Reference | Rationale |
|------|-----------|------|-----------|-----------|
| Classical | Interpolation | Classical | Elahi et al., J. Biomed. Opt. 2011 | Standard CLE pipeline: triangular interpolation removes honeycomb artifact |
| PnP | PnP-BM3D | PnP | Danielyan et al., IEEE TIP 2012 | PnP with BM3D denoiser in fiber bundle forward model |
| Deep Learning | FiberNet | Deep Learning | Shao et al., Med. Image Anal. 2019 | CNN specifically trained on fiber bundle CLE images |
| Deep Learning | EndoL2H | Deep Learning | Ravi et al., IEEE TMI 2022 | Low-to-high quality endoscopy enhancement |

## Required Code Changes

1. **`_algorithm_catalog.py`:** Confirm `_VARIANT_OVERRIDES` entry exists for `confocal_endomicroscopy` pointing to the CLE-specific algorithm pool above.
2. **`_algorithm_catalog.py`:** Verify `_VARIANT_SCORE_ALIASES` maps `confocal_endomicroscopy` to an appropriate score pool (clinical_optics PSNR range ~30–40 dB is reasonable for CLE).

**Priority:** MEDIUM — the clinical_optics routing gives wrong algorithms if the override is absent.
