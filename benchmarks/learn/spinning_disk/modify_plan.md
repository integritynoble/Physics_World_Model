# Modify Plan: spinning_disk

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

The algorithms are appropriate for spinning disk confocal microscopy. The primary reconstruction task is deconvolution of the confocal PSF and denoising of photon-limited images, which is exactly what this microscopy pool addresses:

- **Richardson-Lucy** is the standard deconvolution algorithm used in confocal microscopy, including spinning disk.
- **PnP-FISTA** is a plug-and-play method applicable to microscopy deconvolution.
- **CARE** (Content-Aware image REstoration) was specifically designed for fluorescence microscopy denoising/deconvolution, and spinning disk confocal is one of the modalities it was validated on.
- **Restormer** is a general-purpose image restoration transformer that can be applied to microscopy.

All four algorithms are well-suited to the spinning disk confocal reconstruction problem. The unique challenge of spinning disk (pinhole crosstalk, disk wobble) is captured in the mismatch parameters rather than requiring different reconstruction algorithms.

No code changes needed.

## Files to Modify
None.
