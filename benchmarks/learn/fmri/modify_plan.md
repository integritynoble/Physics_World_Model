# Modify Plan: fmri

## Current State

- **Category:** medical
- **Carrier:** Spin/RF
- **Score key:** mri (routed via `("medical", "Spin/RF") -> "mri"`)
- **Algorithms assigned (8, full MRI pool):**
  1. Zero-Filled IFFT (Classical) -- Zbontar et al., arXiv 2018
  2. L1-Wavelet / ESPIRiT (Compressed Sensing) -- Lustig et al., MRM 2007
  3. PnP-DnCNN (PnP) -- Ahmad et al., IEEE SPM 2020
  4. U-Net (Deep Learning) -- Zbontar et al., arXiv 2018
  5. E2E-VarNet (Deep Unrolling) -- Sriram et al., MICCAI 2020
  6. PromptMR (Deep Unrolling) -- Bai et al., ECCV 2024
  7. ReconFormer (Transformer) -- Guo et al., IEEE TMI 2024
  8. Score-MRI (Diffusion) -- Chung & Ye, Med. Image Anal. 2022

## Assessment

**Appropriate: YES**

fMRI acquires k-space data using EPI sequences -- it is fundamentally MRI
reconstruction with the added dimension of temporal BOLD dynamics. The MRI
reconstruction pool is exactly right: Zero-Filled IFFT, L1-Wavelet/ESPIRiT,
VarNet, etc. are all directly applicable to accelerated fMRI reconstruction
(e.g., undersampled EPI). The carrier routing `("medical", "Spin/RF") -> "mri"`
correctly sends fMRI to the MRI algorithm pool.

All 8 algorithms are real, published MRI reconstruction methods with correct
citations.

## Code Changes Needed

No code changes needed.
