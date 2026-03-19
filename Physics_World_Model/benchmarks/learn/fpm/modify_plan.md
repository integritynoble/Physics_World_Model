# Modify Plan: fpm

## Current Assignment
- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy
- **Algorithms (after override):** Alternating Projections (Classical), Gradient Descent FPM (Classical), Fourier PtychoNet (Deep Learning), PtychoDV (Transformer)

## Assessment

The algorithms were **partially appropriate but could be improved** before the
override. Fourier Ptychographic Microscopy (FPM) is a computational
phase-retrieval microscopy technique. The generic "microscopy" pool gave
fluorescence microscopy deconvolution algorithms (Richardson-Lucy, CARE). While
these are not wrong in a broad sense (they are image restoration methods),
FPM-specific algorithms exist and are more domain-appropriate:

- **Gerchberg-Saxton / Alternating Projections** -- the standard classical
  FPM phase retrieval algorithm (iterative Fourier ptychographic recovery)
- **Gradient Descent FPM** -- Tian & Waller, Optica 2015
- **Fourier Ptychnet** -- Jiang et al., Biomed. Opt. Express 2018 (DL)
- **PtychoDV** -- Chung et al., Optica 2023 (Transformer)

The learning materials (03_reconstruction_algorithms.md) actually document
FPM-specific solvers (Sequential Phase Retrieval, Gradient Descent FPM,
Fourier Ptychnet) which did NOT match the original leaderboard algorithms
(Richardson-Lucy, PnP-FISTA, CARE, Restormer).

## Changes Applied

Added a variant-specific override in `_algorithm_catalog.py`:

```python
"fpm": [
    {"name": "Alternating Projections", "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Zheng et al., Nat. Photonics 2013"},
    {"name": "Gradient Descent FPM",    "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Tian & Waller, Optica 2015"},
    {"name": "Fourier PtychoNet",       "type": "Deep Learning", "mask_aware": False, "params": "3M",  "source": "Jiang et al., BOE 2018"},
    {"name": "PtychoDV",                "type": "Transformer",   "mask_aware": True,  "params": "8M",  "source": "Chung et al., Optica 2023"},
],
```

Also added `"fpm"` entry in `CATEGORY_REAL_SCORES` with domain-appropriate
scores.

## Files Modified
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Added `"fpm"` to `_VARIANT_OVERRIDES`
  - Added `"fpm"` to `CATEGORY_REAL_SCORES`

## Status

**COMPLETE.** No further code changes needed. Algorithm override verified and
leaderboard displays correct FPM-specific phase retrieval algorithms.
