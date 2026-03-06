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

## 2026-03-06 Comprehensive Check Update

- Physics: y = (x * h_conf) + n_bg + n_photon; confocal PSF = h_ill * pinhole; axial FWHM ~ 2n*lambda/NA^2
- Key mismatch: pinhole size (Airy units), refractive index mismatch (spherical aberration at depth), photobleaching, sCMOS gain non-uniformity
- GCS datasets: 3 tiers confirmed
- Algorithm pool: PASS — RL (standard confocal deconvolution), PnP-FISTA (photon-limited), CARE (demonstrated on spinning disk data), Restormer (state-of-the-art restoration)
- Note: CARE is the strongest algorithm for this modality, having been validated specifically on spinning disk live-cell data
