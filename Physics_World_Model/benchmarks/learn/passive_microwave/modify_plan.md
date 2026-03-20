# Modify Plan: passive_microwave (Passive Microwave Radiometry)

**Created:** 2026-03-03
**Status:** PASS -- no code changes needed

## Current State

- **Category:** remote_sensing
- **Carrier:** RF
- **Score key:** remote_sensing (with variant override for passive microwave)
- **Algorithms served (4):**
  1. Backus-Gilbert (Classical) -- Backus & Gilbert, 1970
  2. Tikhonov-SMOS (Classical/PnP) -- Tikhonov regularized inversion for SMOS
  3. RadioNet (Deep Learning) -- CNN-based brightness temperature retrieval
  4. MWR-Former (Transformer) -- Transformer for microwave radiometry

## Assessment

**Correct.** The passive_microwave override provides radiometry-appropriate algorithms.
The previous assignment used SAR algorithms (Matched Filter, SAR-BM3D, SAR-DRN,
SAR-CAM), which were entirely wrong:

- SAR is an active coherent imaging system (radar echoes + range-Doppler processing)
- Passive microwave is a thermal emission measurement (brightness temperature retrieval)
- The two modalities have completely different physics and algorithm requirements

The override correctly provides:
- Backus-Gilbert as the standard radiometric deconvolution baseline
- Tikhonov-SMOS for regularized aperture synthesis inversion
- RadioNet and MWR-Former for learned retrieval approaches

## Verdict

**PASS -- no code changes needed.** The variant override correctly provides
radiometry-specific algorithms for passive microwave.

## Recommended Changes

None required. The fix has already been implemented and verified.
