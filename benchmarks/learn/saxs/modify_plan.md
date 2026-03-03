# Modify Plan -- saxs

## Current State

- **Category:** scientific_instrumentation
- **Carrier:** X-ray
- **Routing:** No carrier routing for `("scientific_instrumentation", "X-ray")` -> falls to `_CATEGORY_ALGORITHMS["scientific_instrumentation"]`
- **Score key:** scientific_instrumentation
- **Algorithms assigned:**
  1. Deconv (Classical) -- Analytical baseline
  2. PnP-BM3D (PnP) -- Danielyan et al., 2012
  3. ResNet-Calib (Deep Learning) -- ResNet for calibration, 2022
  4. CalibFormer (Transformer) -- Transformer calibration, 2024

## Assessment

**Partially appropriate: Acceptable but not domain-optimal.**

Small-Angle X-ray Scattering (SAXS) measures the scattering intensity I(q) as a function of momentum transfer q to characterize nanoscale structure (particle size, shape, spacing). The inverse problem involves recovering structural parameters or real-space electron density distributions from 1D/2D scattering patterns.

- **Deconv**: Deconvolution is used in SAXS for slit-smearing correction and beam profile deconvolution. Reasonable classical baseline.
- **PnP-BM3D**: Generic denoising prior. SAXS data can be noisy at high q, so a denoising approach is applicable, though not domain-standard.
- **ResNet-Calib**: The "calibration" framing is somewhat generic. SAXS does require careful calibration (beam center, detector distance, normalization), so this is loosely relevant.
- **CalibFormer**: Same as above -- generic calibration transformer.

The domain-standard approaches for SAXS would include indirect Fourier transform (IFT / GNOM by Svergun, 1992), pair-distance distribution fitting, and model-based fitting (SasView). However, the current generic scientific_instrumentation pool is functional for the benchmark's inverse-problem framework.

## Plan

No code changes needed. The current algorithms are functional for the benchmark framework. A dedicated SAXS algorithm pool could improve domain authenticity, but is not required for correctness.
