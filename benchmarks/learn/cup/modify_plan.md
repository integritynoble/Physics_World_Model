# Modify Plan: cup

## Current State

- **Category:** ultrafast
- **Carrier:** Photon
- **Routing:** Direct to `ultrafast` pool (no carrier routing override)
- **Score key:** ultrafast
- **Algorithms served:**
  1. TwIST (Classical) -- Bioucas-Dias & Figueiredo, IEEE TIP 2007
  2. PnP-FFDNet (PnP) -- Yuan et al., 2020
  3. CUP-Net (Deep Learning) -- Parker et al., 2021
  4. AL-DL (Hybrid) -- Yao et al., Photon. Res. 2021

## Assessment

All four algorithms are excellent matches for Compressed Ultrafast Photography:

- **TwIST (Two-step Iterative Shrinkage/Thresholding):** Standard compressed sensing algorithm widely used in CUP reconstruction. The original CUP paper (Gao et al., Nature 2014) used similar iterative CS approaches. CORRECT.
- **PnP-FFDNet:** Plug-and-play with FFDNet denoiser applied to CUP. Yuan et al. 2020 specifically demonstrated PnP methods for snapshot compressive imaging including ultrafast modalities. CORRECT.
- **CUP-Net:** Deep learning network designed specifically for CUP reconstruction. PERFECT FIT.
- **AL-DL (Augmented Lagrangian + Deep Learning):** Hybrid approach from Yao et al. (Photonics Research 2021) combining model-based optimization with deep learning for ultrafast imaging. CORRECT.

No code changes needed.
