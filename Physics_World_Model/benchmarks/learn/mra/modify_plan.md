# Modify Plan — mra

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

The algorithms are **appropriate** for MR Angiography (MRA).

MRA is fundamentally an MRI technique that images blood vessels. The reconstruction problem is the same core inverse problem as standard MRI: recovering images from undersampled k-space data. MRA-specific concerns include:

- **Contrast-enhanced MRA (CE-MRA):** Uses gadolinium contrast agents; reconstruction is standard MRI reconstruction with emphasis on vascular structures.
- **Time-of-Flight MRA (TOF-MRA):** Flow-sensitive MRI; reconstruction is standard k-space inversion.
- **Phase-Contrast MRA (PC-MRA):** Velocity-encoded; similar k-space reconstruction.
- **4D-Flow MRA:** Time-resolved; accelerated acquisition benefits directly from compressed sensing and deep learning MRI reconstruction.

All 8 MRI pool algorithms are directly applicable:
- Zero-Filled IFFT and L1-Wavelet are standard baselines for any MRI k-space problem.
- VarNet, PromptMR, and the other deep methods have been successfully applied to MRA reconstruction (e.g., Hammernik et al. demonstrated VarNet on cardiac MRI including MRA-like sequences).
- Compressed sensing MRA is an active clinical research area (Lustig et al.'s original CS-MRI work included angiography applications).

## Current Algorithm Count (updated 2026-03-06)

Full pool (10 algorithms, MRI pool): Zero-Filled IFFT, L1-Wavelet (ESPIRiT), PnP-DnCNN, U-Net, E2E-VarNet, PromptMR, ReconFormer, MRI-DiffusionNet, Score-MRI, MRDynamo.

**Status:** PASS — check.md written 2026-03-06

## Verdict

No code changes needed.
