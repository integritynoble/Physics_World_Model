# Modify Plan: edx_mapping (STEM-EDX Elemental Mapping)

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

Reasonable match. STEM-EDX elemental mapping produces spectral images where
each pixel contains an X-ray energy spectrum. The primary reconstruction tasks
are: (1) spectral denoising (EDX signals are photon-starved), (2) peak
deconvolution for overlapping elemental lines, and (3) quantification
(Cliff-Lorimer or zeta-factor methods).

The generic EM denoising pool addresses task (1) well:

- Wiener Filter provides spectral denoising baseline.
- BM3D exploits spatial self-similarity in elemental maps (which often have
  repeated microstructural features).
- Noise2Void is directly applicable to low-dose STEM-EDX data.
- SwinIR provides strong spatial restoration.

The pool does not address peak deconvolution or quantification, but these are
handled by the forward model rather than the reconstruction algorithm pool.
For the image-quality-focused benchmark evaluation, denoising algorithms are
the correct category.

## Verdict

No code changes needed. The generic EM denoising pool is appropriate for
STEM-EDX elemental map restoration, which is fundamentally a low-SNR image
denoising problem.
