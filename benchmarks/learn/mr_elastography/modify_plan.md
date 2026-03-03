# Modify Plan — mr_elastography

## Current State

- **Category:** medical
- **Carrier:** Spin/RF
- **Score key:** mri (routed via _CARRIER_ROUTING: medical + Spin/RF -> mri)
- **Algorithms (from catalog, 8 total):**
  1. Zero-Filled IFFT (Classical) -- Zbontar et al., arXiv 2018
  2. L1-Wavelet / ESPIRiT (Compressed Sensing) -- Lustig et al., MRM 2007
  3. PnP-DnCNN (PnP) -- Ahmad et al., IEEE SPM 2020
  4. U-Net (Deep Learning) -- Zbontar et al., arXiv 2018
  5. E2E-VarNet (Deep Unrolling) -- Sriram et al., MICCAI 2020
  6. PromptMR (Deep Unrolling) -- Bai et al., ECCV 2024
  7. ReconFormer (Transformer) -- Guo et al., IEEE TMI 2024
  8. Score-MRI (Diffusion) -- Chung & Ye, Med. Image Anal. 2022
- **Leaderboard (live):** All 8 algorithms (8 entries)

## Assessment

The algorithms are **partially appropriate** for MR Elastography (MRE).

MRE is a two-stage process:
1. **MRI acquisition** of wave propagation images (motion-encoding gradient-sensitized phase images)
2. **Elastogram inversion** to estimate tissue stiffness (shear modulus) from the wave displacement field

The MRI pool algorithms (Zero-Filled IFFT, L1-Wavelet, VarNet, etc.) are appropriate for **Stage 1** -- reconstructing the MR images from undersampled k-space data. This is a valid and important reconstruction step.

However, the **distinctive** reconstruction challenge in MRE is Stage 2: the elastogram inversion. Domain-specific algorithms include:
- **LFE (Local Frequency Estimation)** -- Manduca et al., Med. Image Anal. 2001
- **Direct Inversion** -- Oliphant et al., MRM 2001
- **FEM-based inversion** -- Van Houten et al., MRM 2001
- **NNLS (Non-Negative Least Squares)** -- Papazoglou et al., MRM 2012

The MRI reconstruction algorithms are defensible because MRE does rely on MRI acquisition (and accelerated MRE acquisition is an active research area), but they represent only the first half of the MRE pipeline.

## Verdict

No code changes needed. The MRI pool algorithms correctly address the k-space reconstruction stage of MRE, which is a valid inverse problem. The elastogram inversion stage is a separate downstream task. Listing all 8 MRI algorithms provides a comprehensive benchmark set.
