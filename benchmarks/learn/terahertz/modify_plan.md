# Modify Plan: Terahertz Imaging (THz)

**Created:** 2026-03-03
**Status:** No code changes needed

## Assessment

Terahertz imaging falls under `industrial_inspection` category with carrier `THz`. It receives:

- TSR (Classical) -- Thermographic Signal Reconstruction (Shepard et al., 2003)
- PnP-ADMM (PnP) -- generic plug-and-play prior
- DefectNet (Deep Learning) -- U-Net for NDT
- LSTM-NDT (Recurrent) -- Fang et al., 2022

TSR is strictly a thermographic technique (time-domain polynomial fitting of IR decay), not a THz method. THz imaging has its own reconstruction approaches (e.g., THz time-domain spectral analysis, THz pulsed imaging deconvolution). However, in the NDT context the industrial_inspection pool is used as a generic NDT baseline across thermography, ultrasonic, eddy current, and THz modalities. Since no THz-specific reconstruction benchmark exists, the generic NDT pool is an acceptable placeholder.

## Deferred Items

1. **TSR specificity**: TSR is thermography-specific. A more appropriate classical baseline for THz might be time-domain deconvolution or matched filtering. Low priority since this is a shared NDT pool.

No code changes required at this time.
