# Modify Plan: TEM (Transmission Electron Microscopy)

**Created:** 2026-03-03
**Status:** No code changes needed

## Assessment

TEM falls under `electron_microscopy` category with carrier `Electron`. Since TEM is not in the `_CRYO_EM_VARIANTS` set, it correctly routes to the `_EM_GENERIC_POOL`:

- Wiener Filter (Classical) -- standard EM denoising baseline
- BM3D (PnP) -- widely used for EM image denoising
- Noise2Void (Deep Learning) -- self-supervised denoiser popular in EM (Krull et al., CVPR 2019)
- SwinIR (Transformer) -- general-purpose restoration transformer

These are appropriate for TEM image restoration. TEM does not use cryo-EM single-particle reconstruction (RELION/cryoSPARC), so the generic EM denoising pool is the correct choice. Score key `em_generic` is also correct.

No algorithm or citation changes required.
