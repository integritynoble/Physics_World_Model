# Modify Plan -- insar

## Current State (Updated 2026-03-03)

- **Category:** remote_sensing
- **Carrier:** RF
- **Score key:** remote_sensing
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["insar"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. Goldstein-MCF (Classical) -- Goldstein et al., Radio Sci. 1988
  2. InSAR-BM3D (PnP) -- Deledalle et al., IEEE TIP 2015
  3. PhaseNet (Deep Learning) -- Sica et al., IEEE TGRS 2021
  4. InSAR-Former (Transformer) -- InSAR phase transformer, 2024

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the generic SAR pool (Matched Filter, SAR-BM3D,
SAR-DRN, SAR-CAM) with InSAR-specific phase unwrapping and interferometric
processing algorithms. SAR image formation and InSAR phase analysis are
fundamentally different problems, and this override correctly addresses that
distinction.

## Changes Applied

- Added `_VARIANT_OVERRIDES["insar"]` with four InSAR-specific algorithms
- Goldstein-MCF: branch-cut + minimum cost flow phase unwrapping
- InSAR-BM3D: nonlocal interferometric phase filtering
- PhaseNet: deep learning phase unwrapping
- InSAR-Former: transformer-based phase estimation

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
