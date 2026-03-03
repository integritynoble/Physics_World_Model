# Modify Plan -- lattice_lightsheet

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

**Appropriate.** Lattice Light-Sheet Microscopy (LLSM) is a fluorescence microscopy technique where the primary reconstruction task is 3D deconvolution and denoising. The general microscopy pool is well-suited:

- **Richardson-Lucy** is the standard 3D deconvolution method used in light-sheet microscopy, including in the original Lattice Light-Sheet paper by Chen et al., Science 2014. Directly applicable.
- **PnP-FISTA** is appropriate for regularized deconvolution of volumetric microscopy data.
- **CARE** was explicitly designed for and validated on light-sheet microscopy data (including isotropic reconstruction and denoising). This is arguably the most domain-appropriate DL method possible for LLSM.
- **Restormer** provides strong 3D image restoration applicable to volumetric microscopy.

CARE in particular was demonstrated on lattice light-sheet data in the original Weigert et al., Nat. Methods 2018 paper, making this pool especially well-matched.

## Recommendation

No code changes needed.
