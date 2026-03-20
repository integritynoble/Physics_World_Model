# Modify Plan: fmri

**Date:** 2026-03-06

## Current State

- **Category:** medical
- **Carrier:** Spin/RF
- **Score key:** mri (routed via `("medical", "Spin/RF") -> "mri"`)
- **Algorithms assigned (full MRI pool):**
  1. Zero-Filled IFFT (Classical) -- Zbontar et al., arXiv 2018
  2. L1-Wavelet / ESPIRiT (Compressed Sensing) -- Lustig et al., MRM 2007; Uecker et al., MRM 2014
  3. PnP-DnCNN (PnP) -- Ahmad et al., IEEE SPM 2020
  4. U-Net (Deep Learning) -- Zbontar et al., arXiv 2018
  5. E2E-VarNet (Deep Unrolling) -- Sriram et al., MICCAI 2020
  6. PromptMR (Deep Unrolling) -- Bai et al., ECCV 2024
  7. ReconFormer (Transformer) -- Guo et al., IEEE TMI 2024
  8. Score-MRI (Diffusion) -- Chung & Ye, Med. Image Anal. 2022

## Assessment

**Appropriate: YES — EXCELLENT FIT**

fMRI acquires k-space data using EPI (Echo Planar Imaging) sequences — it is fundamentally MRI reconstruction with the added dimension of temporal BOLD dynamics. The MRI reconstruction pool is exactly right:

- **Zero-Filled IFFT**: Universal MRI baseline. CORRECT.
- **L1-Wavelet/ESPIRiT**: Compressed sensing + parallel imaging baseline. Lustig 2007 (compressed sensing MRI) and Uecker 2014 (ESPIRiT) are real, foundational papers. CORRECT.
- **PnP-DnCNN**: Ahmad et al., IEEE SPM 2020 is a real PnP paper for MRI. CORRECT.
- **E2E-VarNet**: Sriram et al., MICCAI 2020 — fastMRI challenge winner. CORRECT.
- **PromptMR**: Bai et al., ECCV 2024 — state-of-the-art prompting-based MRI reconstruction. CORRECT.
- **ReconFormer**: Guo et al., IEEE TMI 2024 — transformer for MRI reconstruction. CORRECT.
- **Score-MRI**: Chung & Ye, Med. Image Anal. 2022 — diffusion model for MRI. CORRECT.

The carrier routing `("medical", "Spin/RF") -> "mri"` correctly sends fMRI to the MRI algorithm pool.

### fMRI-Specific Metric Note

For fMRI, tSNR (temporal signal-to-noise ratio) is clinically critical — it determines statistical power for activation detection. The benchmark should report tSNR in addition to standard PSNR/SSIM. No algorithm changes needed, only metric documentation.

## Code Changes Needed

No code changes needed.

**Priority:** NONE — algorithms correct. Consider adding tSNR metric documentation to benchmark page.
