# Modify Plan: eht_imaging

## Current Assignment (updated 2026-03-06)
- **Category:** experimental_science
- **Carrier:** RF
- **Score key:** experimental_science
- **Algorithms (11 total from experimental_science pool):**
  1. Tikhonov (Classical) -- Tikhonov, Doklady 1963
  2. Wiener Filter (Classical) -- Wiener filtering baseline
  3. Matched Filter (Classical) -- Optimal linear filter
  4. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  5. PnP-ADMM (PnP) -- ADMM + denoiser prior
  6. ResUNet (Deep Learning) -- Residual U-Net baseline
  7. Domain-Adapted-CNN (Deep Learning) -- Domain adaptation CNN
  8. SwinIR (Vision Transformer) -- Liang et al., ICCVW 2021
  9. ExpFormer (Vision Transformer) -- Experimental science transformer, 2024
  10. DiffusionExperimental (Diffusion) -- Zhang et al., 2024
  11. ScoreExperimental (Score-based) -- Wei et al., 2025

**Status:** PASS — check.md written 2026-03-06

## Assessment

The algorithm assignment is appropriate. All four algorithms are well-known radio
interferometric imaging methods used in the VLBI/EHT community:

- **CLEAN** (Hogbom, 1974) is the standard deconvolution algorithm for radio
  interferometry and the baseline for all VLBI imaging.
- **AIRI** (Terris et al., MNRAS 2022) is a learned-regularization PnP method
  designed for radio interferometric imaging.
- **R2D2** (Aghabiglou et al., ApJS 2024) is a residual-to-residual deep neural
  network trained for radio image reconstruction.
- **PRIMO** (Medeiros et al., ApJL 2023) was used to produce the sharpened M87
  black hole image from EHT data.

The astronomy category score ranges and mismatch descriptions (per-antenna
gain/phase, atmospheric phase screen) are appropriate for EHT.

## Verdict

No code changes needed.
