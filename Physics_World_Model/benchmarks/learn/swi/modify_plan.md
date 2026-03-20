# Modify Plan: swi

## Current State
- **Category:** medical
- **Carrier:** Spin/RF
- **Score key:** mri (routed via `_CARRIER_ROUTING`)
- **Algorithms:**
  1. Zero-Filled IFFT (Classical) -- Zbontar et al., arXiv 2018
  2. L1-Wavelet (ESPIRiT) (Compressed Sensing) -- Lustig et al., MRM 2007
  3. PnP-DnCNN (PnP) -- Ahmad et al., IEEE SPM 2020
  4. U-Net (Deep Learning) -- Zbontar et al., arXiv 2018
  5. E2E-VarNet (Deep Unrolling) -- Sriram et al., MICCAI 2020
  6. PromptMR (Deep Unrolling) -- Bai et al., ECCV 2024
  7. ReconFormer (Transformer) -- Guo et al., IEEE TMI 2024
  8. Score-MRI (Diffusion) -- Chung & Ye, Med. Image Anal. 2022

## Assessment

The algorithms are appropriate for SWI (Susceptibility-Weighted Imaging). SWI is an MRI contrast mechanism that uses gradient-echo sequences to generate phase and magnitude images sensitive to magnetic susceptibility differences. The reconstruction problem is fundamentally MRI k-space to image reconstruction, which is exactly what the MRI pool addresses.

- **Zero-Filled IFFT** is the standard MRI reconstruction baseline, applicable to SWI.
- **L1-Wavelet (ESPIRiT)** is the standard compressed sensing MRI method, applicable to accelerated SWI acquisition.
- **E2E-VarNet** and other fastMRI methods work on the same k-space reconstruction problem.
- All other MRI methods in the pool are directly applicable.

SWI-specific post-processing (phase unwrapping, background field removal, QSM dipole inversion) is captured in the mismatch parameters rather than needing different reconstruction algorithms. The MRI pool correctly handles the image reconstruction step.

The carrier-based routing `("medical", "Spin/RF") -> "mri"` correctly identifies SWI as an MRI modality.

No code changes needed.

## Files to Modify
None.

## 2026-03-06 Comprehensive Check Update

- Physics: s(k) = integral rho(r)*exp(i*phi(r))*exp(-R2*TE)*exp(-i2pi*k.r) dr; phi = gamma*TE*chi(k)*D(k) dipole kernel
- Key mismatch: B_0 field inhomogeneity, k-space undersampling pattern, coil sensitivity maps, TE mismatch
- GCS datasets: 3 tiers confirmed
- Algorithm pool: PASS — MRI pool (Zero-Filled IFFT, L1-Wavelet, E2E-VarNet, Score-MRI) is correct; SWI uses same k-space reconstruction framework as standard MRI
- Note: SWI phase accuracy critical for susceptibility mapping; Score-MRI provides principled uncertainty quantification for phase estimates
