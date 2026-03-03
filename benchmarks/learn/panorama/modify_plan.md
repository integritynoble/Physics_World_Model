# Modify Plan: panorama

## Current State
- **Category:** computational_photography
- **Carrier:** Photon
- **Score key:** computational_photography
- **Algorithms:**
  1. Wiener-Deconv (Classical) -- Analytical baseline
  2. PnP-FFDNet (PnP) -- Zhang et al., 2017
  3. HDR-CNN (Deep Learning) -- Eilertsen et al., ACM TOG 2017
  4. Uformer (Transformer) -- Wang et al., CVPR 2022

## Assessment

Panorama multi-focus fusion is a computational photography problem involving stitching and focus-stacking multiple images. The category `computational_photography` is reasonable. The algorithms are generic computational photography methods (Wiener deconvolution, HDR-CNN, etc.). While not perfectly specific to panorama/focus-fusion, they are defensible as general image reconstruction baselines for this category.

More specific algorithms would be multi-focus fusion methods (e.g., Laplacian pyramid fusion, MFIF-GAN, IFCNN), but the current algorithms are not fundamentally wrong for a computational photography benchmark. The leaderboard shows (Uformer, PnP-FFDNet, SRResNet, FISTA-TV) which are also reasonable generic choices.

## Required Changes

No code changes needed. The algorithms are acceptable for this computational photography modality.
