# Modify Plan: event_camera

## Current Assignment
- **Category:** computational_photography
- **Carrier:** Photon
- **Score key:** computational_photography
- **Algorithms:** Wiener-Deconv (Classical), PnP-FFDNet (PnP), HDR-CNN (Deep Learning), Uformer (Transformer)

## Assessment

The algorithms are **inappropriate**. Event cameras (Dynamic Vision Sensors)
produce asynchronous per-pixel brightness change events, not standard frames.
The reconstruction task is event-to-video (intensity image reconstruction from
an event stream), which requires specialized algorithms that handle the
asynchronous, sparse temporal data format.

**Problems:**
1. **Wiener-Deconv** assumes a standard linear degradation model (blur kernel).
   Event cameras do not produce blurred images; they produce event streams.
2. **HDR-CNN** is for tone-mapping / HDR image reconstruction from bracketed
   exposures. It has no relevance to event streams.
3. **PnP-FFDNet** and **Uformer** are generic image restoration tools that
   expect frame-based input, not asynchronous event data.
4. The event camera field has well-established dedicated methods:
   **E2VID** (Rebecq et al., TPAMI 2020), **FireNet** (Scheerlinck et al.,
   ECCV 2020 Workshop), **SPADE-E2VID** (Cadena et al., 2024).

## Recommended Changes

Add a variant-specific override:

```python
"event_camera": [
    {"name": "Event Integration",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Direct event accumulation baseline"},
    {"name": "cF2F",               "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Scheerlinck et al., CVPR 2019"},
    {"name": "E2VID",              "type": "Deep Learning", "mask_aware": False, "params": "10M",  "source": "Rebecq et al., IEEE TPAMI 2020"},
    {"name": "SPADE-E2VID",        "type": "Transformer",   "mask_aware": True,  "params": "15M",  "source": "Cadena et al., 2024"},
],
```

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Add `"event_camera"` to `_VARIANT_OVERRIDES`
  - Add `"event_camera"` to `CATEGORY_REAL_SCORES`
