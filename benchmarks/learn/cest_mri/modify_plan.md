# Modify Plan: cest_mri

## Current State (After Fix)
- **Category:** medical
- **Sub-category pool:** mri_recon (CEST-specific override)
- **Algorithms:** [Lorentzian Fit, WASSR, DeepCEST, CEST-Former]

## Assessment
Algorithms are now domain-appropriate.

The previous MRI pool (Zero-Filled IFFT, L1-Wavelet/ESPIRiT, PnP-DnCNN, U-Net, E2E-VarNet, PromptMR, ReconFormer, Score-MRI) targeted generic k-space reconstruction but missed the CEST-specific Z-spectrum fitting and B0/B1 correction steps. The replacement algorithms are CEST-native:
- **Lorentzian Fit** — multi-pool Lorentzian fitting of the Z-spectrum to extract individual CEST contributions (Woessner et al., MRM 2005); the canonical CEST analysis method
- **WASSR** — Water Saturation Shift Referencing for B0 inhomogeneity correction, addressing the dominant mismatch source (Kim et al., MRM 2009)
- **DeepCEST** — deep learning CNN for robust CEST quantification at high field (Glang et al., MRM 2020)
- **CEST-Former** — attention-based transformer for multi-offset Z-spectrum reconstruction and acceleration (Chen et al., IEEE TMI 2024)

## Verdict
No further code changes needed.
