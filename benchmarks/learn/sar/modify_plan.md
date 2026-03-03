# Modify Plan -- sar

## Current State

- **Category:** remote_sensing
- **Carrier:** RF
- **Routing:** No carrier routing for `("remote_sensing", "RF")` -> falls to `_CATEGORY_ALGORITHMS["remote_sensing"]`
- **Score key:** remote_sensing
- **Algorithms assigned:**
  1. Matched Filter (Classical) -- Standard SAR focusing
  2. SAR-BM3D (PnP) -- Parrilli et al., IEEE TGRS 2012
  3. SAR-DRN (Deep Learning) -- Zhang et al., RS 2018
  4. SAR-CAM (Transformer) -- Cross-attention SAR, 2024

## Assessment

**Appropriate: YES.**

Synthetic Aperture Radar (SAR) forms images by coherently processing radar returns from a moving platform. The remote_sensing algorithm pool was explicitly designed for SAR:

- **Matched Filter**: Standard range-Doppler focusing / range compression -- the canonical SAR image formation algorithm.
- **SAR-BM3D**: Block-matching despeckling adapted for SAR multiplicative noise (Parrilli et al., IEEE TGRS 2012). A well-known SAR denoising method.
- **SAR-DRN**: Deep residual network for SAR image enhancement (Zhang et al., Remote Sensing 2018).
- **SAR-CAM**: Cross-attention mechanism for SAR image reconstruction, representing the latest transformer-based approaches.

All four algorithms are SAR-domain-specific and well-cited. This is one of the best-matched modality-algorithm assignments in the catalog.

## Plan

No code changes needed.
