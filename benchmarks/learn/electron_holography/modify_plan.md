# Modify Plan: electron_holography

## Current Assignment
- **Category:** electron_microscopy
- **Carrier:** Electron
- **Score key:** em_generic (not in `_CRYO_EM_VARIANTS`, so routes to generic EM pool)
- **Algorithms:** Wiener Filter (Classical), BM3D (PnP), Noise2Void (Deep Learning), SwinIR (Transformer)

## Assessment

The algorithms are **marginally appropriate but not ideal**. Electron holography
is an interference-based phase retrieval technique where an electron biprism
creates interference fringes encoding the specimen phase shift. The
reconstruction task is:

1. Extract the sideband from the Fourier transform of the hologram
2. Inverse-FFT to recover amplitude and phase
3. Unwrap the phase

The current em_generic pool provides generic denoising algorithms (Wiener, BM3D,
Noise2Void, SwinIR). These are reasonable for a **denoising post-processing**
step applied to the recovered phase/amplitude maps, but they miss the core
reconstruction step (sideband extraction + phase unwrapping).

**Problems:**
1. **Wiener Filter** is defensible as a generic baseline but does not reflect the
   actual holographic reconstruction workflow.
2. The field uses dedicated algorithms: **Fourier sideband filtering** (classical),
   **transport-of-intensity equation (TIE)**, and DL-based phase unwrapping
   methods like **PhaseNet** or **HoloNet**.
3. The learning materials correctly identify `fourier_sideband` as the default
   solver, which is absent from the leaderboard.

## Recommended Changes

Add a variant-specific override for electron holography:

```python
"electron_holography": [
    {"name": "Sideband FFT",     "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Lichte, Ultramicroscopy 1986"},
    {"name": "PnP-BM3D",         "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Danielyan et al., 2012"},
    {"name": "HoloNet",          "type": "Deep Learning", "mask_aware": False, "params": "3M",   "source": "Wang et al., ACS Nano 2021"},
    {"name": "PhaseNet-EH",      "type": "Deep Learning", "mask_aware": True,  "params": "1.5M", "source": "Ren et al., Microscopy 2023"},
],
```

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Add `"electron_holography"` to `_VARIANT_OVERRIDES`
  - Add `"electron_holography"` to `CATEGORY_REAL_SCORES`
