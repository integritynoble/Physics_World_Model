# Modify Plan -- cest_mri

**Date:** 2026-03-03
**Category:** medical | **Carrier:** Spin/RF | **Score key:** mri

## Current Algorithms (from catalog)

| # | Algorithm            | Type               | Source                             |
|---|----------------------|--------------------|------------------------------------|
| 1 | Zero-Filled IFFT     | Classical          | Zbontar et al., arXiv 2018        |
| 2 | L1-Wavelet (ESPIRiT) | Compressed Sensing | Lustig et al., MRM 2007           |
| 3 | PnP-DnCNN           | PnP                | Ahmad et al., IEEE SPM 2020       |
| 4 | U-Net                | Deep Learning      | Zbontar et al., arXiv 2018        |
| 5 | E2E-VarNet           | Deep Unrolling     | Sriram et al., MICCAI 2020        |
| 6 | PromptMR             | Deep Unrolling     | Bai et al., ECCV 2024             |
| 7 | ReconFormer          | Transformer        | Guo et al., IEEE TMI 2024         |
| 8 | Score-MRI            | Diffusion          | Chung & Ye, Med. Image Anal. 2022 |

## Assessment

### Are algorithms domain-appropriate?
YES, with a nuance. CEST MRI (Chemical Exchange Saturation Transfer) is an MRI technique, and the carrier-based routing correctly sends (medical, Spin/RF) to the `mri` variant override pool. The 8 MRI algorithms are all real, well-cited MRI reconstruction methods.

However, CEST MRI has a unique reconstruction pipeline:
1. First, k-space undersampled images at many saturation offsets must be reconstructed (standard MRI recon -- this is what the algorithms address)
2. Then, Z-spectra must be fitted to extract CEST contrast maps (Lorentzian fitting, AREX, etc.)

The benchmark algorithms address step 1 (MRI image reconstruction from undersampled k-space), which is the appropriate scope for a physics-based benchmark.

- Zero-Filled IFFT: Standard MRI baseline -- correct
- L1-Wavelet (ESPIRiT): Lustig et al., MRM 2007 -- THE compressed sensing MRI paper, correct
- PnP-DnCNN: Ahmad et al., IEEE SPM 2020 -- PnP for MRI, correct
- U-Net: Zbontar et al., arXiv 2018 -- fastMRI baseline, correct
- E2E-VarNet: Sriram et al., MICCAI 2020 -- End-to-end variational network, correct
- PromptMR: Bai et al., ECCV 2024 -- recent prompt-based MRI recon, correct
- ReconFormer: Guo et al., IEEE TMI 2024 -- Transformer MRI recon, correct
- Score-MRI: Chung & Ye, Med. Image Anal. 2022 -- diffusion-based MRI recon, correct

### Are citations correct?
YES. All 8 citations are accurate and correspond to real, well-known MRI reconstruction papers.

### Other issues
- check.md reports only 4 algorithms (PromptMR, PnP-ADMM, VarNet, Zero-filled IFFT) but the actual catalog returns 8 from the MRI hand-crafted override. The check.md is stale.
- The MRI pool is the richest pool in the catalog (8 algorithms vs typical 4), which is appropriate given MRI's prominence in medical imaging.
- The check.md learning materials mention `z_spectrum_fit` as the default solver, which addresses CEST-specific step 2. This is distinct from the leaderboard algorithms which address step 1.

## Plan

No code changes needed. The MRI pool is the gold standard of the algorithm catalog -- 8 real, correctly-cited, domain-appropriate algorithms spanning classical to diffusion-based methods. The carrier-based routing (medical, Spin/RF) -> mri works perfectly for CEST MRI.
