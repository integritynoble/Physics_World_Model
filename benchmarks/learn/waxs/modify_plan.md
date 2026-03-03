# Modify Plan: WAXS (Wide-Angle X-ray Scattering)

**Created:** 2026-03-03
**Status:** Algorithms are acceptable but generic

## Assessment

WAXS falls under `scientific_instrumentation` category with carrier `X-ray`. It receives:

- Deconv (Classical) -- analytical deconvolution baseline
- PnP-BM3D (PnP) -- plug-and-play with BM3D denoiser (Danielyan et al., 2012)
- ResNet-Calib (Deep Learning) -- ResNet for calibration
- CalibFormer (Transformer) -- transformer for calibration

### Analysis

WAXS measures X-ray scattering at wide angles to determine crystal structure, phase composition, and texture. The "reconstruction" in WAXS typically involves:

- Background subtraction and peak fitting (classical)
- Rietveld refinement for crystal structure determination
- Pair distribution function (PDF) analysis
- Denoising of 2D detector images for low-count measurements

The generic `scientific_instrumentation` pool (Deconv, PnP-BM3D, ResNet-Calib, CalibFormer) treats this as a signal recovery / calibration problem, which is a reasonable abstraction. Deconvolution and denoising of scattering patterns is a valid task. The algorithms are not WAXS-specific but serve as generic instrumentation baselines.

Score key `scientific_instrumentation` is correct for this generic pool.

## Deferred Items

1. **Specialization**: A WAXS-specific pool could include Rietveld refinement or PDF-based methods, but these are structure determination methods rather than image reconstruction algorithms. Low priority.

No code changes required.
