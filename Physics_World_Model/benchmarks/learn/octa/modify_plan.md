# Modify Plan: octa (OCT Angiography)

## Current State

- **Category:** medical
- **Carrier:** Photon
- **Score key:** clinical_optics (routed via `_CARRIER_ROUTING[("medical", "Photon")]`)
- **Algorithms served (4):**
  1. FFT-OCT (Classical) -- Analytical baseline
  2. BM4D (PnP) -- Maggioni et al., IEEE TIP 2013
  3. Speckle-DenoiseNet (Deep Learning) -- Devalla et al., BOE 2019
  4. OCTA-Net (Transformer) -- Hybrid U-Net+Transformer, 2023

## Assessment

**Excellent.** OCTA (OCT Angiography) is directly within the clinical_optics domain,
and the algorithm pool is specifically designed for OCT/OCTA:

- **FFT-OCT** is the standard spectral-domain OCT reconstruction baseline, which
  is the first step in OCTA processing (reconstruct OCT volumes, then compute
  inter-frame decorrelation for angiography).
- **BM4D** is widely used for volumetric OCT denoising (speckle reduction).
- **Speckle-DenoiseNet** (Devalla et al., BOE 2019) is a published OCT/OCTA
  denoising network -- directly relevant.
- **OCTA-Net** is an OCTA-specific reconstruction/segmentation network -- directly
  relevant.

This is one of the best-matched modality-algorithm combinations in the catalog.
The clinical_optics pool was essentially designed with OCT/OCTA in mind.

## Verdict

No code changes needed.
