# Modify Plan: ebsd (Electron Backscatter Diffraction)

## Current State

- **Category:** electron_microscopy
- **Carrier:** Electron
- **Score key:** em_generic
- **Algorithms served (EM generic pool):**
  1. Wiener Filter (Classical) -- Analytical baseline
  2. BM3D (PnP) -- Dabov et al., IEEE TIP 2007
  3. Noise2Void (Deep Learning) -- Krull et al., CVPR 2019
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

The generic EM denoising pool is a **rough but acceptable** fit. EBSD collects
Kikuchi diffraction patterns from which crystallographic orientation maps are
reconstructed. The primary reconstruction task is pattern indexing (Hough
transform-based or dictionary indexing), which is quite different from
image denoising.

However, within the PWM benchmark framework, the evaluation focuses on
image-domain quality metrics (PSNR/SSIM) of the reconstructed orientation
maps, where denoising and spatial regularization algorithms are applicable.
The generic EM pool addresses the low-SNR nature of EBSD patterns:

- Wiener Filter provides a baseline spectral denoising.
- BM3D exploits patch self-similarity in noisy EBSD patterns.
- Noise2Void is directly applicable to EM data with Poisson-dominated noise.
- SwinIR provides strong restoration performance.

Domain-specific algorithms (dictionary indexing with denoised patterns, or
DI-based orientation mapping like EDAX OIM, EMsoft) are more appropriate for
the indexing step, but the denoising pool is reasonable for the image
restoration aspect evaluated by the benchmark.

## Verdict

No code changes needed. The generic EM denoising pool is acceptable for the
image restoration evaluation, even though EBSD-specific indexing algorithms
would be more domain-authentic.
