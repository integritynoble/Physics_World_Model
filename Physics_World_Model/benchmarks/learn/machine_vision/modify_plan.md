# Modify Plan -- machine_vision

## Current State (Updated 2026-03-03)

- **Category:** industrial_inspection
- **Carrier:** Photon
- **Score key:** industrial_inspection
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["machine_vision"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. Template Match (Classical) -- Brunelli, Template Matching, 2009
  2. PnP-ADMM (PnP) -- Venkatakrishnan et al., 2013
  3. PatchCore (Deep Learning) -- Roth et al., CVPR 2022
  4. UniAD (Transformer) -- You et al., NeurIPS 2022

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the thermal/NDT pool (TSR, PnP-ADMM, DefectNet,
LSTM-NDT) with optical anomaly detection algorithms. TSR (Thermographic Signal
Reconstruction) and LSTM-NDT are thermography/temporal NDT methods with no
applicability to visible-light optical inspection. The new set includes
state-of-the-art MVTec-AD methods (PatchCore, UniAD) alongside classical
template matching.

## Changes Applied

- Added `_VARIANT_OVERRIDES["machine_vision"]` with four AOI-appropriate algorithms
- Template Match: classical reference-based defect detection
- PnP-ADMM: general image enhancement pre-processing
- PatchCore: memory-bank anomaly detection (SOTA on MVTec-AD)
- UniAD: unified transformer for industrial anomaly detection

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
