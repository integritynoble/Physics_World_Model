# Modify Plan: hdr_imaging

## Current State

- **Category:** computational_photography
- **Carrier:** Photon
- **Score key:** computational_photography
- **Algorithms assigned:**
  1. Wiener-Deconv (Classical) -- Analytical baseline
  2. PnP-FFDNet (PnP) -- Zhang et al., 2017
  3. HDR-CNN (Deep Learning) -- Eilertsen et al., ACM TOG 2017
  4. Uformer (Transformer) -- Wang et al., CVPR 2022

## Assessment

**Appropriate: YES**

HDR imaging is a computational photography technique. The pool is well-chosen:

- **Wiener-Deconv**: A reasonable classical baseline for exposure fusion /
  image restoration. While HDR-specific classical methods exist (Debevec &
  Malik 1997), Wiener deconvolution is acceptable as a generic baseline.
- **PnP-FFDNet**: A PnP denoiser that can handle the noise amplification in
  HDR tone mapping. Appropriate.
- **HDR-CNN**: Eilertsen et al., ACM TOG 2017 is THE seminal deep learning
  paper for single-image HDR reconstruction. Perfectly chosen.
- **Uformer**: A vision transformer for image restoration (Wang et al., CVPR
  2022). Applicable to HDR reconstruction/denoising.

The computational_photography pool was clearly designed with HDR in mind
(HDR-CNN is literally an HDR-specific method). This is a good fit.

## Current Algorithm Count (updated 2026-03-06)

Full pool (14 algorithms): Wiener-Deconv, Laplacian Pyramid, Lucy-Richardson, PnP-FFDNet, PnP-ADMM, HDR-CNN, U-Net, LaplacianFormer, Uformer, DeblurGaussian, HDRFormer, PhotoFormer, DiffusionPhoto, ScorePhoto.

**Status:** PASS — check.md written 2026-03-06

## Code Changes Needed

No code changes needed.
