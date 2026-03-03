# Modify Plan -- impedance_tomo

## Current State (Updated 2026-03-03)

- **Category:** experimental_science
- **Carrier:** Electric
- **Score key:** experimental_science
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["impedance_tomo"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. Gauss-Newton (Classical) -- Cheney et al., SIAM Rev. 1999
  2. TV-ADMM (PnP) -- Borsic et al., Physiol. Meas. 2010
  3. D-bar CNN (Deep Learning) -- Hamilton & Hauptmann, IEEE TMI 2018
  4. EIT-Former (Transformer) -- EIT reconstruction transformer, 2024

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the generic experimental_science pool (Tikhonov,
PnP-RED, ResUNet, SwinIR) with EIT-specific algorithms that address the
nonlinear, severely ill-posed nature of electrical impedance tomography. The
D-bar CNN is a particularly notable inclusion as a physics-informed hybrid
method unique to EIT.

## Changes Applied

- Added `_VARIANT_OVERRIDES["impedance_tomo"]` with four EIT-specific algorithms
- Gauss-Newton: standard iterative linearized EIT reconstruction
- TV-ADMM: total variation regularized conductivity reconstruction
- D-bar CNN: scattering-theory-based direct method + CNN post-processing
- EIT-Former: transformer-based EIT reconstruction

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
