# Modify Plan -- ism

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

**Appropriate.** Image Scanning Microscopy (ISM) is a confocal-family fluorescence microscopy technique. The reconstruction task is PSF deconvolution and pixel reassignment, which fits well within the general microscopy restoration framework:

- **Richardson-Lucy** is the standard deconvolution method used extensively in ISM/confocal for PSF deconvolution. Directly applicable.
- **PnP-FISTA** is appropriate as a plug-and-play method for microscopy deconvolution with learned denoisers.
- **CARE** (Content-Aware Image Restoration) is a flagship deep learning method specifically designed for fluorescence microscopy restoration. Directly applicable to ISM.
- **Restormer** is a strong general-purpose image restoration transformer that has been applied to microscopy denoising.

While ISM has some ISM-specific methods (e.g., pixel reassignment, multi-image ISM-SOFI), the generic microscopy pool covers the core deconvolution/restoration task well. The algorithm names are all credible and commonly used in this domain.

## Recommendation

No code changes needed.
