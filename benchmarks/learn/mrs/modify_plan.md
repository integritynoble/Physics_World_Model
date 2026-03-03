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

**Acceptable, but not ideal.** MRS (MR Spectroscopy) is routed to the MRI algorithm
pool because it shares the Spin/RF carrier. However, MRS reconstruction is fundamentally
different from MRI image reconstruction:

- MRS recovers 1D frequency-domain spectra, not 2D spatial images.
- The classical baseline should be FFT + phasing/baseline correction, not "Zero-Filled IFFT".
- Domain-standard algorithms include LCModel (Prior et al., 1993), TARQUIN, QUEST, and
  jMRUI, which are spectral fitting methods, not image-domain methods.
- Deep learning approaches include DeepSPICE (Lee et al., NMR Biomed 2020) and
  FID-Net (Chen et al., MRM 2022), which operate on spectral/FID data.
- E2E-VarNet, PromptMR, and ReconFormer are image-domain MRI methods that have no
  published application to MRS spectral fitting.

The mismatch is moderate: the leaderboard shows MRI image reconstruction algorithms
applied to a spectroscopy problem, which is misleading but not harmful for the
benchmark's inverse-problem framing (since both share k-space sampling physics).

## Recommended Changes

**Option A (ideal):** Add a dedicated MRS override to `_VARIANT_OVERRIDES` in
`_algorithm_catalog.py` with spectral fitting algorithms:
```python
"mrs": [
    {"name": "FFT + Phase Corr", "type": "Classical",      ...},
    {"name": "LCModel",          "type": "Classical",      ...},
    {"name": "TARQUIN",          "type": "PnP",            ...},
    {"name": "DeepSPICE",        "type": "Deep Learning",  ...},
]
```
Plus corresponding entry in `CATEGORY_REAL_SCORES`.

**Option B (minimal):** Leave as-is. The MRI pool is defensible since MRS shares
k-space acquisition physics and the benchmark frames it as an inverse problem
with the same forward model structure (Fourier encoding + subsampling).

## Verdict

Changes would improve domain accuracy but are not strictly required.
The current MRI routing is physically motivated (same carrier/encoding).
