# Modify Plan: dexa

## Current State (After Fix)

- **Category:** medical
- **Carrier:** X-ray
- **Runner type:** `dual_energy` (variant override, was incorrectly `radon`)
- **Signal shape:** `[256, 256, 2]` (variant override, was `[128, 128, 64]`)
- **Score key:** medical (via `_VARIANT_SCORE_ALIASES`)
- **Algorithms served:**
  1. Dual-Energy Subtraction (Classical) — Lehmann et al., Med. Phys. 1981
  2. PnP-ADMM (PnP) — Venkatakrishnan et al., 2013
  3. Butterfly-Net (Deep Learning) — Li et al., SIAM J. Sci. Comput. 2020
  4. DECT-MULTRA (Deep Unrolling) — Gong et al., IEEE TMI 2020

## Problem (FIXED)

DEXA was using the `radon` runner (CT sinograms) when it should use dual-energy projection.

## Changes Implemented

### 1. `generate_challenge_datasets.py`
- Added `_VARIANT_TO_RUNNER` dict: `"dexa": "dual_energy"`
- Added `_forward_dual_energy()`: physics-accurate Beer-Lambert dual-energy projection model
- Added `_forward_projection()`: simple 2D X-ray projection (for mammography, fluoroscopy, radiography)
- Added `_make_dexa_phantom()`: anatomical bone + soft tissue phantom generator
- Updated `_get_runner_type()`: checks variant overrides first
- Updated `_apply_forward_model()`: dispatches `dual_energy` and `projection` runners
- Updated `_generate_fallback_phantom()`: dispatches to `_make_dexa_phantom()`

### 2. `_challenge_data.py`
- Added `_VARIANT_SIGNAL_SHAPE` dict: `"dexa": [256, 256, 2]`
- Updated `generate_challenge_config()`: checks variant shape overrides

### 3. Datasets regenerated
- All 3 tiers regenerated with correct dual-energy format
- Uploaded to GCS: `gs://pwm-benchmark-datasets/challenge-data/v1.0/dexa_challenge_{tier}.h5`
- Dev tier stripped of x_true

## Verdict

All code changes implemented and verified. No further changes needed.
