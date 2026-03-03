# Modify Plan: electron_tomography

## Current Assignment
- **Category:** electron_microscopy
- **Carrier:** Electron
- **Score key:** electron_microscopy (in `_CRYO_EM_VARIANTS`)
- **Algorithms:** RELION (Classical), cryoSPARC (PnP), cryoDRGN (Deep Learning), CryoTransformer (Transformer)

## Assessment

The algorithms are **partially appropriate but could be improved**. Electron
tomography (ET) reconstructs 3D volumes from tilt-series of 2D projections.
While it shares the "electron microscopy" category with cryo-EM single-particle
analysis, the reconstruction approaches are different:

- **ET** uses tilt-series alignment + tomographic reconstruction (WBP, SIRT,
  ART, GENFIRE). The sample is a single object imaged at multiple tilt angles.
- **Single-particle cryo-EM** (RELION, cryoSPARC) averages thousands of
  identical particles in random orientations. These tools do NOT perform
  tilt-series reconstruction.

**Problems:**
1. **RELION** and **cryoSPARC** are single-particle tools, not tilt-series
   tomographic reconstruction tools. The standard classical ET tool is **WBP**
   (weighted back-projection) or **SIRT** from IMOD/Etomo.
2. **cryoDRGN** handles heterogeneity in single-particle data, not tilt-series.
3. The leaderboard (per check.md) shows WBP and CryoAI, which are more relevant,
   but the catalog still returns the wrong base pool.

**Mitigating factor:** cryo-ET is a real sub-field, and some cryo-EM tools
(cryoSPARC) have added tilt-series processing. IsoNet (a cryo-ET-specific
denoising tool) would be more appropriate.

## Recommended Changes

Add a variant-specific override:

```python
"electron_tomography": [
    {"name": "WBP",       "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Radermacher, Electron Tomography 2006"},
    {"name": "SIRT",      "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Gilbert, J. Theor. Biol. 1972"},
    {"name": "IsoNet",    "type": "Deep Learning", "mask_aware": False, "params": "8M",   "source": "Liu et al., Nat. Commun. 2022"},
    {"name": "CryoAI",    "type": "Deep Learning", "mask_aware": True,  "params": "12M",  "source": "Levy et al., 2022"},
],
```

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Add `"electron_tomography"` to `_VARIANT_OVERRIDES`
  - Remove `"electron_tomography"` from `_CRYO_EM_VARIANTS`
  - Add `"electron_tomography"` to `CATEGORY_REAL_SCORES`
