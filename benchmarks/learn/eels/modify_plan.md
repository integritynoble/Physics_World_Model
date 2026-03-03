# Modify Plan: eels (Electron Energy Loss Spectroscopy)

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

Reasonable match. EELS produces spectrum images where each pixel contains an
energy-loss spectrum. The primary reconstruction tasks are: (1) denoising of
low-dose spectrum images, (2) deconvolution of plural scattering (Fourier-ratio
or Fourier-log methods), and (3) background subtraction (power-law fitting).

The generic EM denoising pool addresses the spatial denoising aspect:

- Wiener Filter is a standard spectral denoising baseline directly applicable
  to EELS spectrum images.
- BM3D exploits spatial self-similarity in EELS maps (useful for crystalline
  materials with repeated unit cells).
- Noise2Void handles the photon-limited noise regime common in EELS without
  requiring clean reference data.
- SwinIR provides strong image restoration for the 2D elemental/bonding maps
  extracted from EELS data cubes.

The Fourier-ratio deconvolution step is specific to EELS and not captured by
the generic pool, but this is handled by the forward model's default solver
(fourier_ratio) rather than the algorithm catalog.

## Verdict

No code changes needed. The generic EM denoising pool is appropriate for
EELS spectrum image restoration, which is fundamentally a low-SNR
denoising/deconvolution problem in the spatial domain.
