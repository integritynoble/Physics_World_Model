# Modify Plan: polsar

## Current State
- **Category:** remote_sensing
- **Carrier:** RF
- **Score key:** remote_sensing
- **Algorithms:**
  1. Matched Filter (Classical) -- Standard SAR focusing
  2. SAR-BM3D (PnP) -- Parrilli et al., IEEE TGRS 2012
  3. SAR-DRN (Deep Learning) -- Zhang et al., RS 2018
  4. SAR-CAM (Transformer) -- Cross-attention SAR, 2024

## Assessment

Polarimetric SAR (PolSAR) is a SAR system that transmits/receives in multiple polarization channels (HH, HV, VH, VV) to characterize scattering properties. The category `remote_sensing` is correct, and the carrier routing `("remote_sensing", "RF")` keeps SAR algorithms, which is largely appropriate since PolSAR **is** SAR.

The algorithms are partially appropriate:
- **Matched Filter** -- SAR focusing is the first step in PolSAR processing. Appropriate.
- **SAR-BM3D** -- speckle filtering applicable to PolSAR channels. Appropriate.
- **SAR-DRN** -- deep SAR denoising. Applicable to PolSAR. Acceptable.
- **SAR-CAM** -- SAR cross-attention model. Applicable. Acceptable.

However, PolSAR has domain-specific algorithms for polarimetric decomposition and classification:
- Pauli/Freeman-Durden decomposition (Classical)
- Refined Lee filter (Lee et al., IEEE TGRS 2006) -- polarimetric speckle filter
- PolSAR-CNN (Zhang et al., IEEE TGRS 2017) -- polarimetric classification
- Wishart classifier (Lee et al., IEEE TGRS 1999)

The current SAR algorithms are a reasonable approximation since they apply to each PolSAR channel's image formation and denoising. The mismatch is mild.

## Required Changes

No code changes needed. The SAR algorithms are applicable to PolSAR image formation and denoising, which is the primary reconstruction task. Polarimetric decomposition is a downstream analysis step beyond the scope of a reconstruction benchmark.
