# Modify Plan: SEM (Scanning Electron Microscopy)

**Created:** 2026-03-03
**Status:** PASS -- no code changes needed

## Current State

- **Category:** electron_microscopy
- **Carrier:** Electron
- **Score key:** em_generic (routed via special EM routing for non-cryo variants)
- **Algorithms served (4):**
  1. Wiener Filter (Classical) -- Wiener, 1949
  2. BM3D (PnP) -- Dabov et al., IEEE TIP 2007
  3. Noise2Void (Deep Learning) -- Krull et al., CVPR 2019
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

**Correct.** SEM was previously getting cryo-EM particle reconstruction algorithms
(RELION, cryoSPARC) via the generic "electron_microscopy" category. These are wrong
for SEM, which needs image denoising/restoration, not single-particle reconstruction.

Fixed by special EM routing: non-cryo EM variants -> em_generic pool.

The current algorithms are all domain-appropriate:
- Wiener Filter provides a classical deconvolution baseline
- BM3D is the gold standard for patch-based image denoising
- Noise2Void enables self-supervised denoising without clean targets
- SwinIR provides modern transformer-based image restoration

## Verdict

**PASS -- no code changes needed.** The special EM routing correctly sends SEM
to the em_generic pool with denoising/restoration algorithms.

## Recommended Changes

None required. The fix has already been implemented and verified.
