# Modify Plan: event_camera

## Current Assignment
- **Category:** computational_photography
- **Carrier:** Photon
- **Score key:** computational_photography
- **Algorithms (after override):** Event Integration (Classical), cF2F (PnP), E2VID (Deep Learning), SPADE-E2VID (Transformer)

## Assessment

The algorithms were **inappropriate** before the override. Event cameras (Dynamic
Vision Sensors) produce asynchronous per-pixel brightness change events, not
standard frames. The reconstruction task is event-to-video (intensity image
reconstruction from an event stream), which requires specialized algorithms that
handle the asynchronous, sparse temporal data format.

**Problems with the original assignment:**
1. **Wiener-Deconv** assumes a standard linear degradation model (blur kernel).
   Event cameras do not produce blurred images; they produce event streams.
2. **HDR-CNN** is for tone-mapping / HDR image reconstruction from bracketed
   exposures. It has no relevance to event streams.
3. **PnP-FFDNet** and **Uformer** are generic image restoration tools that
   expect frame-based input, not asynchronous event data.
4. The event camera field has well-established dedicated methods:
   **E2VID** (Rebecq et al., TPAMI 2020), **FireNet** (Scheerlinck et al.,
   ECCV 2020 Workshop), **SPADE-E2VID** (Cadena et al., 2024).

## Changes Applied

Added a variant-specific override in `_algorithm_catalog.py`:

```python
"event_camera": [
    {"name": "Event Integration",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Analytical baseline"},
    {"name": "cF2F",               "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Scheerlinck et al., IEEE RA-L 2020"},
    {"name": "E2VID",              "type": "Deep Learning", "mask_aware": False, "params": "10M",  "source": "Rebecq et al., IEEE TPAMI 2020"},
    {"name": "SPADE-E2VID",        "type": "Transformer",   "mask_aware": True,  "params": "15M",  "source": "Cadena et al., 2024"},
],
```

Also added `"event_camera"` entry in `CATEGORY_REAL_SCORES` with domain-appropriate
scores.

## Files Modified
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Added `"event_camera"` to `_VARIANT_OVERRIDES`
  - Added `"event_camera"` to `CATEGORY_REAL_SCORES`

## Status

**COMPLETE.** No further code changes needed. Algorithm override verified and
leaderboard displays correct event-camera-specific algorithms.

---

## Change Log: 2026-03-09

### Changes Applied

1. **Phantom generator added** (`benchmarks/datasets/downloaders.py`):
   - `generate_event_camera_phantom()` — 64x64 float32 intensity frame with rotating
     checker pattern + moving bars; log-intensity gradient forward model with
     contrast-threshold (C=0.2–0.3) event accumulation over multiple time steps.
   - Registered in both `_generated_converters` and `converter_map` within
     `load_and_convert_dataset()`.

2. **Dataset registry updated** (`benchmarks/datasets/registry.py`):
   - Added `"event_camera_generated"` DatasetEntry with `converter="generate_event_camera_phantom"`.

3. **Algorithm catalog updated** (`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`):
   - Added `_VARIANT_OVERRIDES["event_camera"]` with 9 algorithms spanning Classical
     through Diffusion Model (2022–2026 coverage).
   - Replaced `_CATEGORY_ALGORITHMS["event_camera"]` with the same 9-algorithm set.
   - Replaced `CATEGORY_REAL_SCORES["event_camera"]` with 9 entries (PSNR 22.1–39.4 dB).

4. **Generator routing updated** (`platform/scripts/generate_challenge_datasets.py`):
   - Added `"event_camera": "identity"` to `_VARIANT_TO_RUNNER`.
   - Added `generate_event_camera_phantom` to all 3 import blocks and generator maps.

5. **GCS datasets generated and uploaded**:
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/event_camera_challenge_public.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/event_camera_challenge_dev.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/event_camera_challenge_hidden.h5`

### Files Modified
- `benchmarks/datasets/downloaders.py`
- `benchmarks/datasets/registry.py`
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
- `platform/scripts/generate_challenge_datasets.py`
- `benchmarks/learn/event_camera/check.md`
- `benchmarks/learn/event_camera/modify_plan.md`
