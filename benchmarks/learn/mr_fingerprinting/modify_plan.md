# Modify Plan: mr_fingerprinting

## Current State (After Fix)
- **Category:** medical
- **Sub-category pool:** mri_recon (MRF-specific override)
- **Algorithms:** [SVD-MRF, MANTIS, MRF-Net, MRF-Former]

## Assessment
Algorithms are now domain-appropriate.

The previous generic MRI pool (Zero-Filled IFFT, L1-Wavelet/ESPIRiT, PnP-DnCNN, U-Net, E2E-VarNet, PromptMR, ReconFormer, Score-MRI) addressed only the k-space reconstruction stage, missing the dictionary matching / quantitative parameter estimation stage that is the core of MRF. The replacement algorithms address the complete MRF pipeline:
- **SVD-MRF** — SVD subspace compression accelerating dictionary matching by 300x (McGivney et al., IEEE TMI 2014); bridges classical MRI recon with MRF-specific parameter fitting
- **MANTIS** — Model-Augmented Neural neTwork with Incoherent k-space Sampling, combining deep learning with MRF-specific subspace constraints (Liu et al., MRM 2019)
- **MRF-Net** — deep learning CNN that directly maps fingerprint time series to T1/T2 parameter maps, bypassing explicit dictionary matching (Cohen et al., MRM 2018)
- **MRF-Former** — transformer-based temporal signal analysis for simultaneous multi-parameter mapping (Luo et al., IEEE TMI 2023)

## Verdict
No further code changes needed.
