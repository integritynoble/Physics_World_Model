# Modify Plan: phase_contrast

## Current State
- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy
- **Algorithms:**
  1. Richardson-Lucy (Classical) -- Richardson 1972 / Lucy 1974
  2. PnP-FISTA (PnP) -- Bai et al., 2020
  3. CARE (Deep Learning) -- Weigert et al., Nat. Methods 2018
  4. Restormer (Transformer) -- Zamir et al., CVPR 2022

## Assessment

Phase contrast microscopy (PCM) converts phase shifts in transmitted light into intensity variations using an annular phase ring, producing images of transparent specimens. The category `microscopy` is correct. The reconstruction problem involves deconvolution, halo artifact removal, and quantitative phase recovery.

The generic microscopy algorithms are partially appropriate:
- **Richardson-Lucy** -- applicable for deconvolution of phase contrast images, but PCM introduces specific halo artifacts that Richardson-Lucy does not address. Acceptable as a baseline.
- **PnP-FISTA** -- generic PnP applicable here. Acceptable.
- **CARE** -- deep learning denoising/restoration. Applicable to PCM images. Acceptable.
- **Restormer** -- general-purpose image restoration transformer. Acceptable.

More specific algorithms exist (e.g., PhaseStain by Rivenson et al., Light: Sci. Appl. 2019, or QPI methods), but the current generic microscopy algorithms are defensible for a benchmark.

## Required Changes

No code changes needed. The generic microscopy algorithms are acceptable for phase contrast microscopy reconstruction.
