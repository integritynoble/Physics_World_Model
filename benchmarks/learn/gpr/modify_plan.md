# Modify Plan: gpr

## Current State (Updated 2026-03-03)

- **Category:** remote_sensing
- **Carrier:** RF
- **Score key:** remote_sensing
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["gpr"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. Kirchhoff Migration (Classical) -- Stolt, Geophysics 1978
  2. RTM (Classical) -- Baysal et al., Geophysics 1983
  3. GPR-RCNN (Deep Learning) -- Pham & Lefevre, JECE 2020
  4. HyperDet (Deep Learning) -- GPR detection transformer, 2023

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the generic remote_sensing SAR pool (Matched
Filter, SAR-BM3D, SAR-DRN, SAR-CAM) with GPR-specific algorithms. GPR uses
near-field subsurface migration rather than far-field SAR focusing, making
this distinction critical.

## Changes Applied

- Added `_VARIANT_OVERRIDES["gpr"]` with four GPR-specific algorithms
- Kirchhoff Migration: standard GPR diffraction collapse
- RTM: wave-equation-based migration for complex media
- GPR-RCNN: region-based CNN for subsurface object detection
- HyperDet: transformer-based hyperbola detection from radargrams

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
