# Modify Plan: proton_radiography

## Current State
- **Category:** scientific_instrumentation
- **Carrier:** Proton
- **Score key:** scientific_instrumentation
- **Algorithms:**
  1. Deconv (Classical) -- Analytical baseline
  2. PnP-BM3D (PnP) -- Danielyan et al., 2012
  3. ResNet-Calib (Deep Learning) -- ResNet for calibration, 2022
  4. CalibFormer (Transformer) -- Transformer calibration, 2024

## Assessment

Proton radiography uses proton beams to image the internal density/areal density of objects by measuring energy loss and multiple Coulomb scattering. It is used in both high-energy physics (e.g., pRad at LANL) and medical proton CT. The category `scientific_instrumentation` is reasonable.

The algorithms are generic "scientific instrumentation" methods:
- **Deconv** -- generic deconvolution baseline. Minimally applicable. Proton radiography reconstruction involves more than deconvolution; it requires reconstructing density from energy loss and scattering angle measurements.
- **PnP-BM3D** -- generic denoising prior. Applicable as a regularizer but not domain-specific.
- **ResNet-Calib** -- generic calibration network. Not specific.
- **CalibFormer** -- generic calibration transformer. Not specific.

More domain-specific algorithms:
- Most-Likely Path (MLP) estimation (Schulte et al., Med. Phys. 2008) -- reconstructs proton path from entry/exit measurements
- Filtered Back-Projection for proton CT (Penfold et al., Med. Phys. 2010)
- Algebraic Reconstruction (ART/SIRT) (Penfold et al., 2009) -- iterative proton CT
- CNN-pCT (Krah et al., Phys. Med. Biol. 2019) -- deep learning proton CT

The mismatch is moderate. The generic algorithms do not reflect the specific physics of proton radiography (energy loss, scattering) but are defensible as generic reconstruction baselines.

## Required Changes

No code changes needed. The generic scientific_instrumentation algorithms serve as acceptable baselines for a proton radiography reconstruction benchmark, though domain-specific algorithms would be more informative. This is a low-priority improvement.
