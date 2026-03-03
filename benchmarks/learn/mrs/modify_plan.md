# Modify Plan: mrs (MR Spectroscopy)

## Current State

- **Category:** medical
- **Carrier:** Spin/RF
- **Score key:** mri (routed via `_CARRIER_ROUTING[("medical", "Spin/RF")]`)
- **Algorithms served (8):**
  1. Zero-Filled IFFT (Classical) -- Zbontar et al., arXiv 2018
  2. L1-Wavelet (ESPIRiT) (Compressed Sensing) -- Lustig et al., MRM 2007
  3. PnP-DnCNN (PnP) -- Ahmad et al., IEEE SPM 2020
  4. U-Net (Deep Learning) -- Zbontar et al., arXiv 2018
  5. E2E-VarNet (Deep Unrolling) -- Sriram et al., MICCAI 2020
  6. PromptMR (Deep Unrolling) -- Bai et al., ECCV 2024
  7. ReconFormer (Transformer) -- Guo et al., IEEE TMI 2024
  8. Score-MRI (Diffusion) -- Chung & Ye, Med. Image Anal. 2022

## Assessment

**Acceptable.** MRS (MR Spectroscopy) is routed to the MRI algorithm pool because
it shares the Spin/RF carrier. While MRS reconstruction is fundamentally a 1D spectral
fitting problem (recovering metabolite concentrations from FID data), the routing is
physically motivated:

- Both MRS and MRI share k-space Fourier encoding physics
- The inverse problem structure (recover signal from undersampled Fourier measurements) is the same
- Compressed sensing and deep learning approaches from MRI are applicable to accelerated MRSI
- The benchmark correctly frames MRS as a Fourier inverse problem

Domain-specific MRS algorithms (LCModel, TARQUIN, QUEST, FID-Net) are spectral fitting
methods that operate differently from image-domain methods, but the current pool is not
incorrect for the benchmark's inverse-problem framing.

## Verdict

**PASS -- no code changes needed.** The carrier routing `(medical, Spin/RF) -> mri`
is physically justified. The MRI pool algorithms correctly test the Fourier inverse
problem shared by both modalities.

## Recommended Changes

None required. Optional future enhancement: add a dedicated MRS override with
spectral fitting algorithms for improved domain specificity, but this is not
necessary for correct benchmark operation.
