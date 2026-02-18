# Combined PhotonAgent + MismatchAgent Redesign — Implementation Report

**Date:** 2026-02-11
**Branch:** `master`
**Commit baseline:** `5a9576e` (all 26 modalities pass benchmarks)

---

## 1. Problem Statement

Two critical fidelity issues affected the PWM simulation pipeline:

### Problem A — PhotonAgent Ignored by Experiments

The PhotonAgent computes a physics-based photon count per modality, but **all experiments ignored it** and used the same hardcoded `[1e3, 1e4, 1e5]` photon levels for every modality (CASSI, SPC, CACTI). The same `_apply_photon_noise()` function was copy-pasted across **8+ files** with an identical Poisson+Gaussian model — even for modalities like MRI (thermal/Gaussian noise) or CT (Poisson-only). No modality-specific noise recipes existed.

### Problem B — MismatchAgent Sub-pixel Blindness

The MismatchAgent had no awareness of parameter physics types. **80+ call sites** used `np.roll(arr, int(round(dx)))` for spatial shifts — Python 3 banker's rounding makes `int(round(0.5))` = 0, so the **CASSI mild-severity mask shift (0.5 px) did literally nothing**. `np.roll` also wraps edges, which is physically wrong for sensor-based imaging.

---

## 2. Solution Architecture

A unified "physical fidelity layer" implemented in 9 phases:

```
Phase 1  subpixel.py         ─── pure addition, no breakage
Phase 2  noise/apply.py      ─── pure addition, no breakage
Phase 3  photon_db.yaml      ─── additive YAML + backward-compatible schema
Phase 4  PhotonAgent          ─── optional fields only
Phase 5  MismatchAgent        ─── additive schema + agent enhancements
Phase 6  mismatch_sweep.py   ─── primary sub-pixel bug fix
Phase 7  core operators       ─── targeted np.roll → subpixel fixes
Phase 8  experiment noise     ─── imports only, same behavior
Phase 9  tests                ─── validate everything
```

---

## 3. Phase-by-Phase Implementation

### Phase 1: Sub-pixel Shift Utilities

**New file:** `packages/pwm_core/pwm_core/mismatch/subpixel.py`

Three functions replacing `np.roll` throughout the codebase:

| Function | Purpose | Backing |
|---|---|---|
| `subpixel_shift_2d(arr, dx, dy)` | 2D shift with sub-pixel accuracy | `scipy.ndimage.shift` |
| `subpixel_shift_3d_spatial(arr, dx, dy)` | Per-frame shift for (H,W,T) stacks | Loops `subpixel_shift_2d` |
| `subpixel_warp_2d(arr, dx, dy, theta_deg)` | Shift + rotation via affine transform | `scipy.ndimage.affine_transform` |

Key design choice: `mode="constant"` (zero-fill at boundaries) is physically correct for sensor-based imaging, unlike `np.roll` which wraps pixels around edges.

### Phase 2: Shared Noise Application Module

**New file:** `packages/pwm_core/pwm_core/noise/apply.py`

Replaces 8 copy-pasted `_apply_photon_noise()` functions with one canonical version:

```python
apply_photon_noise(y, photon_level, rng, *, noise_model="poisson_gaussian",
                   read_sigma_fraction=0.01)
```

Three noise models:

| Model | Use case | Behaviour |
|---|---|---|
| `poisson_gaussian` | Default (fluorescence, CASSI, SPC, CACTI) | Poisson shot + Gaussian read noise |
| `poisson` | CT, ptychography | Shot noise only, no read component |
| `gaussian` | MRI | Thermal noise only, sigma = range/level |

Default parameters produce **bit-identical** output to the existing code — verified by seed-42 backward compatibility test.

Also provides `get_noise_recipe(modality_key, level_name, photon_db)` for registry-driven noise lookup.

### Phase 3: Photon Levels in photon_db.yaml

**Modified:** `packages/pwm_core/contrib/photon_db.yaml`

Added `noise_model` and `photon_levels` (bright / standard / low_light) to all 25 modalities:

| Modality | Bright | Standard | Low Light | Noise Model |
|---|---|---|---|---|
| cassi | 1e5 | 1e4 | 1e3 | poisson_gaussian |
| spc | 1e5 | 1e4 | 5e2 | poisson_gaussian |
| cacti | 1e5 | 1e4 | 1e3 | poisson_gaussian |
| ct | 5e6 | 1e6 | 2e5 | poisson |
| mri | SNR 30 dB | SNR 20 dB | SNR 10 dB | gaussian |
| ptychography | 1e8 | 1e6 | 1e4 | poisson |
| holography | 1e7 | 1e5 | 1e3 | poisson_gaussian |
| nerf | 1e9 | 1e7 | 1e5 | poisson_gaussian |
| ... | ... | ... | ... | ... |

**Modified:** `packages/pwm_core/pwm_core/agents/registry_schemas.py`
- Added `PhotonLevelYaml` schema class (n_photons, snr_proxy, scenario, read_sigma_fraction, sigma)
- Added `noise_model` and `photon_levels` optional fields to `PhotonModelYaml`

### Phase 4: PhotonAgent Enhancement

**Modified:** `packages/pwm_core/pwm_core/agents/photon_agent.py`

Added ~20 lines at end of `run()` to populate new fields from registry:

```python
recommended_levels: Optional[Dict[str, Dict[str, Any]]]  # from photon_db
noise_recipe: Optional[Dict[str, Any]]                    # noise_model + default_level
```

**Modified:** `packages/pwm_core/pwm_core/agents/contracts.py`
- Added `recommended_levels` and `noise_recipe` optional fields to `PhotonReport`

### Phase 5: MismatchAgent — Parameter Physics Classification

**Modified:** `packages/pwm_core/pwm_core/agents/mismatch_agent.py`

Added two static methods:

**`classify_param_physics(name, param_spec) -> str`** — 7 valid types:

| param_type | Apply method | Examples |
|---|---|---|
| `spatial_shift` | `subpixel_shift_2d` | mask_dx, mask_dy, detector_offset |
| `rotation` | `subpixel_warp_2d` | rotation, tilt_angle |
| `scale` | Scalar multiply | gain, dispersion_step |
| `blur` | `gaussian_filter` | psf_sigma, irf_width |
| `offset` | Scalar add | background, defocus |
| `timing` | Index permutation | temporal_jitter |
| `position` | Coordinate perturbation | probe_position_error_x/y |

**`validate_mismatch_effect(name, param_type, typical_error) -> dict`** — catches dead parameters:
- Warns if `spatial_shift` typical_error < 0.5 px (rounds to 0 with `np.roll`)
- Warns if `rotation` typical_error < 0.01 deg (below measurable threshold)

**Modified:** `packages/pwm_core/contrib/mismatch_db.yaml`
- Added `param_type` to every parameter across all 26 modalities (80+ parameters)
- Added CACTI `mask_rotation` parameter (range: [-3.0, 3.0] degrees)
- Updated CACTI severity_weights to include mask_rotation: 0.10

**Modified:** `packages/pwm_core/pwm_core/agents/contracts.py`
- Added `param_types` and `subpixel_warnings` optional fields to `MismatchReport`
- Added `param_type: Optional[str]` to `MismatchParamYaml` in registry_schemas.py

### Phase 6: Fix Mismatch Injection (mismatch_sweep.py)

**Modified:** `experiments/inversenet/mismatch_sweep.py`

| Function | Before (broken) | After (fixed) |
|---|---|---|
| `apply_cassi_mask_shift` | `np.roll(np.roll(mask, int(round(dy)), 0), int(round(dx)), 1)` | `subpixel_shift_2d(mask, float(dx), float(dy))` |
| `apply_cacti_mask_shift` | `np.roll(masks, shift, axis=0)` | `subpixel_shift_3d_spatial(masks, 0.0, float(shift))` |

Additional changes:
- Changed `CACTI_MASK_SHIFT_TABLE` values from `int` to `float`
- Added `CACTI_MASK_ROTATION_TABLE` (0.5° / 1.5° / 3.0°) and `apply_cacti_rotation`
- Registered CACTI rotation in `_TABLES` dict and `apply_mismatch` dispatcher

### Phase 7: Fix Core Operators

Four targeted files:

| File | Line | Before | After |
|---|---|---|---|
| `cassi_operator.py` | forward | `np.roll(np.roll(band, int(dy), 0), int(dx), 1)` | `subpixel_shift_2d(band, dx, dy)` |
| `cassi_operator.py` | adjoint | `np.roll(np.roll(back, -int(dy), 0), -int(dx), 1)` | `subpixel_shift_2d(back, -dx, -dy)` |
| `primitives.py` | SpectralDispersion forward | `np.roll(x[:,:,l], int(round(shift)), axis=1)` | `subpixel_shift_2d(x[:,:,l], shift, 0.0)` |
| `primitives.py` | SpectralDispersion adjoint | same pattern, reversed | same pattern, negated |
| `gen_cassi.py` | `_cassi_forward` | `np.roll(np.roll(band, int(round(dy)), 0), int(round(dx)), 1)` | `subpixel_shift_2d(band, dx, dy)` |

### Phase 8: Migrate Experiments to Shared Noise

Replaced local `_apply_photon_noise()` definitions with delegation to `pwm_core.noise.apply.apply_photon_noise`:

| File | Change |
|---|---|
| `experiments/inversenet/gen_cassi.py` | Body → delegate to shared function |
| `experiments/inversenet/gen_spc.py` | Body → delegate to shared function |
| `experiments/inversenet/gen_cacti.py` | Body → delegate to shared function |
| `experiments/pwm_flagship/spc_loop.py` | Body → delegate to shared function |
| `experiments/pwm_flagship/cacti_loop.py` | Body → delegate to shared function |
| `experiments/pwm_flagship/cassi_loop.py` | Import chain already fixed via gen_cassi |
| `experiments/pwmi_cassi/*` | Import chain already fixed via gen_cassi |

### Phase 9: Tests + Verification

**New file:** `packages/pwm_core/tests/test_subpixel.py` — 20 tests

| Test | What it verifies |
|---|---|
| `test_half_pixel_shift_nonzero` | 0.5 px shift produces measurable change |
| `test_zero_shift_identity` | 0.0 shift is exact identity (copy) |
| `test_integer_shift_matches_interior` | Integer shift matches np.roll in interior |
| `test_energy_nonincreasing` | mode=constant can only lose boundary energy |
| `test_3d_shift_all_frames` | Each (H,W,T) frame shifted independently |
| `test_pure_shift_changes_array` | Warp with theta=0 produces measurable shift |
| `test_rotation_changes_array` | Non-zero rotation changes the array |
| `test_zero_warp_identity` | Zero shift + zero rotation = identity |
| `test_cassi_mask_shift_mild_nonzero` | Regression: mild severity (0.5 px) changes mask |
| `test_explicit_param_type_returned` | classify_param_physics returns explicit type |
| `test_infer_from_name_dx` | Infers spatial_shift from name+unit |
| `test_infer_rotation` | Infers rotation from name+unit |
| `test_infer_scale` | Infers scale from name |
| `test_fallback_to_scale` | Unknown names default to scale |
| `test_warns_subthreshold_spatial` | Flags spatial shift < 0.5 px |
| `test_adequate_spatial_shift` | Above-threshold shift is effective |
| `test_warns_subthreshold_rotation` | Flags rotation < 0.01 deg |
| `test_scale_always_effective` | Scale never flagged |
| `test_report_includes_param_type` | MismatchReport new fields work |
| `test_report_optional_fields_none` | Optional fields can be None |

**New file:** `packages/pwm_core/tests/test_photon_noise.py` — 16 tests

| Test | What it verifies |
|---|---|
| `test_poisson_gaussian_default` | Bit-identical to old inline code (seed 42) |
| `test_poisson_only_no_read_noise` | Poisson-only produces noisy output, correct mean |
| `test_gaussian_only_mode` | Gaussian sigma matches signal_range / photon_level |
| `test_unknown_model_raises` | Invalid noise_model raises ValueError |
| `test_poisson_gaussian_adds_read_noise` | Default model changes signal |
| `test_has_modalities` | photon_db.yaml has 18+ modalities |
| `test_all_modalities_have_noise_model` | Every modality has valid noise_model |
| `test_all_modalities_have_photon_levels` | Every modality has bright/standard/low_light |
| `test_photon_levels_have_required_fields` | Levels have scenario + (n_photons or sigma) |
| `test_cassi_standard_recipe` | get_noise_recipe returns correct CASSI values |
| `test_missing_modality_raises` | Unknown modality raises KeyError |
| `test_missing_level_raises` | Unknown level raises KeyError |
| `test_ct_poisson_model` | CT returns poisson noise_model |
| `test_mri_gaussian_model` | MRI returns gaussian noise_model |
| `test_report_with_recommended_levels` | PhotonReport accepts new optional fields |
| `test_report_without_optional_fields` | Optional fields default to None |

---

## 4. Results

### Test Suite

| Test run | Result |
|---|---|
| `test_subpixel.py` | **20/20 passed** |
| `test_photon_noise.py` | **16/16 passed** |
| `test_registry_integrity.py` | **18/18 passed** |
| Full regression (excl. correction) | **620 passed, 2 skipped, 0 failures** |

Previous baseline: 586 tests. New total: **620 tests** (+34 net new).

### Key Bug Fix Verified

```
CASSI mask_shift(dx=0.5, dy=0.5):
  Before: diff = 0.0000  (BROKEN — 0.5 rounds to 0)
  After:  diff = 1553.25 (FIXED — sub-pixel interpolation)
```

### Files Modified

| File | Phase | Type |
|---|---|---|
| `pwm_core/mismatch/subpixel.py` | 1 | **NEW** |
| `pwm_core/noise/apply.py` | 2 | **NEW** |
| `pwm_core/noise/__init__.py` | 2 | MODIFY |
| `contrib/photon_db.yaml` | 3 | MODIFY |
| `pwm_core/agents/registry_schemas.py` | 3, 5 | MODIFY |
| `pwm_core/agents/contracts.py` | 4, 5 | MODIFY |
| `pwm_core/agents/photon_agent.py` | 4 | MODIFY |
| `contrib/mismatch_db.yaml` | 5 | MODIFY |
| `pwm_core/agents/mismatch_agent.py` | 5 | MODIFY |
| `experiments/inversenet/mismatch_sweep.py` | 6 | MODIFY |
| `pwm_core/physics/spectral/cassi_operator.py` | 7 | MODIFY |
| `pwm_core/graph/primitives.py` | 7 | MODIFY |
| `experiments/inversenet/gen_cassi.py` | 7, 8 | MODIFY |
| `experiments/inversenet/gen_spc.py` | 8 | MODIFY |
| `experiments/inversenet/gen_cacti.py` | 8 | MODIFY |
| `experiments/pwm_flagship/spc_loop.py` | 8 | MODIFY |
| `experiments/pwm_flagship/cacti_loop.py` | 8 | MODIFY |
| `tests/test_subpixel.py` | 9 | **NEW** |
| `tests/test_photon_noise.py` | 9 | **NEW** |

**Total: 19 files** (4 new, 15 modified)

---

## 5. Boundary Behaviour Change

The only intentional numerical change:

| | `np.roll` (old) | `scipy.ndimage.shift` (new) |
|---|---|---|
| Boundary | Wraps pixels around edges | Fills with zeros |
| Sub-pixel | Integer-only (`int(round(dx))`) | Bilinear interpolation |
| 0.5 px shift | No effect (rounds to 0) | Measurable change |

This is **physically correct** for sensor-based imaging where pixels beyond the detector edge are zero, not wrapped. Existing benchmark numbers (`benchmarks/_cassi_upwmi.py`) are unaffected — they already used `affine_transform`.

---

## 6. Backward Compatibility

- Default `noise_model="poisson_gaussian"` + `read_sigma_fraction=0.01` produces **bit-identical** output to the old code
- `PHOTON_LEVELS = [1e3, 1e4, 1e5]` constants remain as defaults in experiment files — no forced migration
- All new fields in `PhotonReport`, `MismatchReport`, `PhotonModelYaml`, and `MismatchParamYaml` are `Optional` with `None` defaults
- Registry schemas validate both old (without new fields) and new YAML formats
