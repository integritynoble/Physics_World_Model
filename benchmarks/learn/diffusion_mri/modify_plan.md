# Modify Plan: diffusion_mri (Diffusion MRI / DTI)

## Current State

- **Category:** medical
- **Carrier:** Spin/RF
- **Score key:** mri (routed via carrier)
- **Algorithms served (8 total):**
  1. Zero-Filled IFFT (Classical) -- Zbontar et al., arXiv 2018
  2. L1-Wavelet / ESPIRiT (Compressed Sensing) -- Lustig et al., MRM 2007
  3. PnP-DnCNN (PnP) -- Ahmad et al., IEEE SPM 2020
  4. U-Net (Deep Learning) -- Zbontar et al., arXiv 2018
  5. E2E-VarNet (Deep Unrolling) -- Sriram et al., MICCAI 2020
  6. PromptMR (Deep Unrolling) -- Bai et al., ECCV 2024
  7. ReconFormer (Transformer) -- Guo et al., IEEE TMI 2024
  8. Score-MRI (Diffusion) -- Chung & Ye, Med. Image Anal. 2022

## Assessment

Excellent match. Diffusion MRI acquires k-space data just like structural MRI,
with the addition of diffusion-encoding gradients. The undersampled k-space
reconstruction problem is identical in structure: accelerated parallel imaging
with Cartesian or non-Cartesian trajectories. All 8 MRI algorithms are directly
applicable:

- Zero-Filled IFFT and ESPIRiT are standard baselines for accelerated MRI.
- E2E-VarNet and PromptMR are state-of-the-art on fastMRI leaderboards.
- Score-MRI is a diffusion-model approach validated on MRI reconstruction.

The carrier-based routing (`("medical", "Spin/RF") -> "mri"`) correctly directs
diffusion MRI to the MRI algorithm pool rather than the generic medical/CT pool.

## Verdict

No code changes needed.
