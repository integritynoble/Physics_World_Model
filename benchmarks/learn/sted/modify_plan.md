# Modify Plan: sted

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

The algorithms are appropriate for STED (Stimulated Emission Depletion) microscopy. STED is a super-resolution fluorescence microscopy technique where the reconstruction problem is PSF deconvolution and image restoration from noisy, photobleaching-limited data.

- **Richardson-Lucy** is the standard deconvolution method used in STED microscopy. The STED PSF (much sharper than confocal) is well-characterized, making RL deconvolution a natural baseline.
- **PnP-FISTA** is applicable to STED deconvolution with the STED-specific PSF.
- **CARE** was validated on multiple fluorescence microscopy modalities including STED data (Weigert et al., 2018).
- **Restormer** is a general restoration transformer applicable to STED.

Note: The project memory mentions that STED should get DECODE/ANNA-PALM (super-resolution localization methods). However, DECODE is for single-molecule localization microscopy (PALM/STORM), not STED. ANNA-PALM is also for localization microscopy. STED is a deterministic super-resolution method (not stochastic localization), so deconvolution-based methods (Richardson-Lucy, CARE) are more appropriate than localization algorithms. The current microscopy pool is correct for STED.

No code changes needed.

## Files to Modify
None.
