# Modify Plan: electron_diffraction

## Current Assignment
- **Category:** electron_microscopy
- **Carrier:** Electron
- **Score key:** electron_microscopy (in `_CRYO_EM_VARIANTS`)
- **Algorithms:** RELION (Classical), cryoSPARC (PnP), cryoDRGN (Deep Learning), CryoTransformer (Transformer)

## Assessment

The algorithms are **inappropriate**. The variant is in `_CRYO_EM_VARIANTS`,
so it receives cryo-EM single-particle reconstruction tools. However,
4D-STEM electron diffraction is a fundamentally different technique:

- **4D-STEM** records a 2D diffraction pattern at each scan position,
  producing a 4D dataset. The reconstruction task is ptychographic phase
  retrieval or strain mapping from convergent-beam electron diffraction
  (CBED) patterns.
- **RELION / cryoSPARC / cryoDRGN** are designed for single-particle
  cryo-EM (classifying and averaging particle images to reconstruct a 3D
  density map). They have no relevance to 4D-STEM ptychography.

**Problems:**
1. All four algorithms are cryo-EM single-particle tools, none address
   electron ptychography or diffraction analysis.
2. The learning materials correctly identify `ptychography_epie` as the
   default solver, which contradicts the leaderboard algorithms.

## Recommended Changes

Remove `electron_diffraction` from `_CRYO_EM_VARIANTS` and add a
variant-specific override:

```python
# In _CRYO_EM_VARIANTS, remove electron_diffraction:
_CRYO_EM_VARIANTS = {"cryo_em", "cryo_et", "electron_tomography"}

# Add to _VARIANT_OVERRIDES:
"electron_diffraction": [
    {"name": "ePIE",           "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Maiden & Rodenburg, Ultramicroscopy 2009"},
    {"name": "WDD",            "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Rodenburg et al., Ultramicroscopy 1993"},
    {"name": "PtychoNN",      "type": "Deep Learning", "mask_aware": False, "params": "2M",   "source": "Cherukara et al., Appl. Phys. Lett. 2020"},
    {"name": "AutoPhaseNN",   "type": "Deep Learning", "mask_aware": True,  "params": "5M",   "source": "Yao et al., NPJ Comput. Mater. 2022"},
],
```

Also add `"electron_diffraction"` to `CATEGORY_REAL_SCORES`.

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Remove `electron_diffraction` from `_CRYO_EM_VARIANTS`
  - Add `"electron_diffraction"` to `_VARIANT_OVERRIDES`
  - Add `"electron_diffraction"` to `CATEGORY_REAL_SCORES`
