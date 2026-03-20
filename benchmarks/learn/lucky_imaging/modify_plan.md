# Modify Plan -- lucky_imaging

## Current State (Updated 2026-03-03)

- **Category:** astronomy
- **Carrier:** Photon
- **Score key:** astronomy
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["lucky_imaging"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. Shift-and-Add (Classical) -- Fried, JOSA 1966
  2. Drizzle (Classical) -- Fruchter & Hook, PASP 2002
  3. BDI (PnP) -- Law et al., ApJ 2006
  4. SpeckleNet (Deep Learning) -- Xin et al., ApJ 2022

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the radio interferometry pool (CLEAN, AIRI,
R2D2, PRIMO) with optical frame selection algorithms. Lucky imaging is an
optical technique (short-exposure frame selection through atmospheric
turbulence), fundamentally different from radio aperture synthesis. The new
algorithms address frame registration, stacking, and atmospheric PSF
deconvolution.

## Changes Applied

- Added `_VARIANT_OVERRIDES["lucky_imaging"]` with four optical astronomy algorithms
- Shift-and-Add: foundational frame registration and stacking
- Drizzle: sub-pixel super-resolved stacking
- BDI: brightest-pixel deconvolution imaging for lucky imaging
- SpeckleNet: deep learning atmospheric restoration

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
