# Modify Plan: eht_imaging

## Current Assignment
- **Category:** astronomy
- **Carrier:** RF
- **Score key:** astronomy
- **Algorithms:** CLEAN (Classical), AIRI (PnP), R2D2 (Deep Learning), PRIMO (Deep Learning)

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
