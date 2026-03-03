# Modify Plan: eddy_current (Eddy Current Imaging)

## Current State

- **Category:** industrial_inspection
- **Carrier:** EM
- **Score key:** industrial_inspection
- **Algorithms served:**
  1. TSR (Classical) -- Shepard et al., 2003
  2. PnP-ADMM (PnP) -- ADMM + denoiser prior
  3. DefectNet (Deep Learning) -- U-Net for NDT, 2021
  4. LSTM-NDT (Recurrent) -- Fang et al., 2022

## Assessment

Acceptable match. Eddy current imaging is a non-destructive testing (NDT)
modality, and the industrial inspection pool contains generic NDT algorithms.

- TSR (Thermographic Signal Reconstruction) is thermography-specific, not
  eddy-current-specific. The eddy current analog would be impedance plane
  analysis or multi-frequency inversion. However, TSR is used here as the
  "classical NDT baseline" representative.
- PnP-ADMM is a generic regularized inversion framework applicable to any
  linear inverse problem including eddy current inversion.
- DefectNet (U-Net for NDT) is applicable to defect detection/characterization
  in eddy current C-scan images.
- LSTM-NDT for temporal NDT signal processing is relevant since eddy current
  data often has temporal/multi-frequency dimensions.

The pool is designed for the industrial inspection category as a whole (covering
thermography, eddy current, ultrasonics, radiography), so some algorithms
(like TSR) are not eddy-current-specific. The overall coverage is reasonable.

## Verdict

No code changes needed. The industrial inspection pool provides a reasonable
set of algorithms spanning classical to deep learning for NDT applications.
The TSR classical baseline is thermography-specific rather than eddy-current-
specific, but this is acceptable within the generic NDT pool design.
