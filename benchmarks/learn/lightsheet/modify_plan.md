# Modify Plan — lightsheet

## Current State

- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy
- **Algorithms (from catalog):**
  1. Richardson-Lucy (Classical) -- Richardson 1972 / Lucy 1974
  2. PnP-FISTA (PnP) -- Bai et al., 2020
  3. CARE (Deep Learning) -- Weigert et al., Nat. Methods 2018
  4. Restormer (Transformer) -- Zamir et al., CVPR 2022
- **Leaderboard (live):** Richardson-Lucy, PnP-FISTA, CARE, Restormer (4 entries)

## Assessment

The algorithms are **appropriate** for Light-Sheet Fluorescence Microscopy (LSFM).

- Richardson-Lucy is the standard deconvolution method for fluorescence microscopy.
- PnP-FISTA is a well-established plug-and-play approach used in microscopy denoising/deconvolution.
- CARE (Content-Aware image REstoration) is a landmark deep learning method specifically designed for fluorescence microscopy by Weigert et al.
- Restormer is a strong general-purpose image restoration transformer that has been applied to microscopy.

All four methods are real, published, properly cited, and represent the standard progression from classical to deep learning for fluorescence microscopy reconstruction.

## Verdict

No code changes needed.
