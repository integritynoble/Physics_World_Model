# Modify Plan: nsom (Near-field Scanning Optical Microscopy)

## Current State

- **Category:** scanning_probe
- **Carrier:** Photon
- **Score key:** scanning_probe (direct category match)
- **Algorithms served (4):**
  1. BTR (Classical) -- Villarrubia, JRNIST 1997
  2. Reg-Deconv (PnP) -- Dongmo et al., 2000
  3. DeepSPM (Deep Learning) -- Alldritt et al., Commun. Phys. 2020
  4. E2E-BTR (Deep Learning) -- Kossler et al., Sci. Rep. 2022

## Assessment

**Acceptable.** NSOM (Near-field Scanning Optical Microscopy) is a scanning probe
technique that shares the same fundamental reconstruction challenges as AFM and STM:
tip-sample deconvolution, scanner nonlinearity correction, and drift compensation.

The scanning_probe algorithm pool is reasonably appropriate:
- **BTR** (Blind Tip Reconstruction) is directly applicable -- NSOM probes have
  aperture/tip convolution effects analogous to AFM tip convolution.
- **Reg-Deconv** (Regularized Deconvolution) applies to deconvolving the near-field
  probe function from the measured signal.
- **DeepSPM** was developed for STM but the approach (learned scanning probe
  correction) generalizes to NSOM.
- **E2E-BTR** (End-to-End Blind Tip Reconstruction) is similarly applicable.

More NSOM-specific algorithms would include near-field deconvolution methods
(e.g., Taubner et al., 2004) and nano-FTIR spectral deconvolution, but the
scanning_probe pool captures the key reconstruction paradigms.

## Verdict

No code changes needed.
