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
