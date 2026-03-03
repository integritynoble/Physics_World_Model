# Modify Plan: Time-of-Flight Depth Camera

## Current State (Updated 2026-03-03)

- **Category:** depth_imaging
- **Carrier:** Photon/IR
- **Score key:** depth_imaging
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["tof_camera"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. Phase Unwrap (Classical) -- Bamji et al., IEEE SSC 2015
  2. PnP-ToF (PnP) -- PnP with depth prior for ToF
  3. DeepToF (Deep Learning) -- Marco et al., ECCV 2018
  4. MPI-Former (Transformer) -- Multi-path interference correction, 2023

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the stereo depth estimation pool (SGM, PnP-ADMM,
PSMNet, RAFT-Stereo) with ToF-specific algorithms. ToF cameras measure depth
via phase-shift of modulated light, facing unique challenges (multi-path
interference, phase wrapping) that stereo matching methods do not address.
The previous pool contained binocular stereo methods fundamentally
inapplicable to phase-based depth sensing.

## Changes Applied

- Added `_VARIANT_OVERRIDES["tof_camera"]` with four ToF-specific algorithms
- Phase Unwrap: multi-frequency phase unwrapping for range ambiguity resolution
- PnP-ToF: plug-and-play with depth-specific priors for ToF refinement
- DeepToF: CNN for multi-path interference correction (ECCV 2018)
- MPI-Former: transformer-based MPI correction

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
