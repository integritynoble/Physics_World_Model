# Modify Plan -- industrial_ct

## Current State (Updated 2026-03-03)

- **Category:** industrial_inspection
- **Carrier:** X-ray
- **Score key:** industrial_inspection
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["industrial_ct"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. FDK (Classical) -- Feldkamp et al., JOSA A 1984
  2. PnP-ADMM (PnP) -- Venkatakrishnan et al., 2013
  3. FBPConvNet (Deep Learning) -- Jin et al., IEEE TIP 2017
  4. Learned Primal-Dual (Deep Unrolling) -- Adler & Oktem, IEEE TMI 2018

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the generic industrial_inspection pool (TSR,
PnP-ADMM, DefectNet, LSTM-NDT) with proper CT reconstruction algorithms. The
previous pool contained thermography (TSR) and temporal NDT methods (LSTM-NDT)
that were fundamentally inappropriate for X-ray tomographic image
reconstruction.

## Changes Applied

- Added `_VARIANT_OVERRIDES["industrial_ct"]` with four CT reconstruction algorithms
- FDK: standard cone-beam filtered back-projection
- PnP-ADMM: iterative reconstruction with learned denoiser priors
- FBPConvNet: CNN-based post-processing of FBP reconstructions
- Learned Primal-Dual: physics-informed deep unrolling with forward/adjoint operators

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
