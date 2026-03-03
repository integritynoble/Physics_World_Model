# Modify Plan -- radio_interferometry

## Current State

- **Category:** remote_sensing
- **Carrier:** RF
- **Routing:** No carrier routing for `("remote_sensing", "RF")` -> falls to `_CATEGORY_ALGORITHMS["remote_sensing"]`
- **Score key:** remote_sensing
- **Algorithms assigned:**
  1. Matched Filter (Classical) -- Standard SAR focusing
  2. SAR-BM3D (PnP) -- Parrilli et al., IEEE TGRS 2012
  3. SAR-DRN (Deep Learning) -- Zhang et al., RS 2018
  4. SAR-CAM (Transformer) -- Cross-attention SAR, 2024

## Assessment

**INAPPROPRIATE. Needs change.**

Radio interferometry (VLBI -- Very Long Baseline Interferometry) is an astronomical imaging technique that recovers sky images from sparse Fourier-plane (visibility) measurements taken by geographically distributed antenna pairs. This is NOT synthetic aperture radar (SAR). The two modalities differ fundamentally:

- **SAR** focuses radar echoes from a moving platform to form terrain images (range-Doppler processing).
- **VLBI** reconstructs astronomical source structure from correlations of signals received at widely separated antennas (Fourier inversion from sparse baselines).

The current SAR-specific algorithms (Matched Filter, SAR-BM3D, SAR-DRN, SAR-CAM) are entirely wrong for radio interferometry. The correct algorithms are the same as radio astronomy:

- **CLEAN** (Hogbom 1974) -- the workhorse of radio interferometric imaging
- **AIRI** (Terris et al., MNRAS 2022) -- PnP for radio interferometric imaging
- **R2D2** (Aghabiglou et al., ApJS 2024) -- deep learning for radio imaging
- **PRIMO** (Medeiros et al., ApJL 2023) -- principal-component interferometric modeling

These already exist in `_CATEGORY_ALGORITHMS["astronomy"]`.

## Plan

Add a `_VARIANT_OVERRIDES` entry for `radio_interferometry` in `_algorithm_catalog.py` that uses the astronomy algorithms:

```python
"radio_interferometry": [
    {"name": "CLEAN",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Hogbom, A&AS 1974"},
    {"name": "AIRI",   "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Terris et al., MNRAS 2022"},
    {"name": "R2D2",   "type": "Deep Learning", "mask_aware": False, "params": "10M",  "source": "Aghabiglou et al., ApJS 2024"},
    {"name": "PRIMO",  "type": "Deep Learning", "mask_aware": True,  "params": "2M",   "source": "Medeiros et al., ApJL 2023"},
],
```

This is preferred over carrier routing because `("remote_sensing", "RF")` should still map to SAR algorithms for actual SAR modalities (like `sar` itself). A variant-level override is the cleanest fix.

### Alternatively:

The modality catalog could be changed from `category: remote_sensing` to `category: astronomy` in the YAML config, but that is a larger change requiring catalog regeneration.
