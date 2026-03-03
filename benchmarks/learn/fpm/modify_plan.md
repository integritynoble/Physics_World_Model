# Modify Plan: fpm

## Current State

- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy
- **Algorithms assigned:**
  1. Richardson-Lucy (Classical) -- Richardson 1972 / Lucy 1974
  2. PnP-FISTA (PnP) -- Bai et al., 2020
  3. CARE (Deep Learning) -- Weigert et al., Nat. Methods 2018
  4. Restormer (Transformer) -- Zamir et al., CVPR 2022

## Assessment

**Partially appropriate -- could be improved**

Fourier Ptychographic Microscopy (FPM) is a computational phase-retrieval
microscopy technique. The current "microscopy" pool gives generic fluorescence
microscopy deconvolution algorithms (Richardson-Lucy, CARE). While these are
not wrong in a broad sense (they are image restoration methods), FPM-specific
algorithms exist and would be more domain-appropriate:

- **Gerchberg-Saxton / Alternating Projections** -- the standard classical
  FPM phase retrieval algorithm (iterative Fourier ptychographic recovery)
- **Gradient Descent FPM** -- Tian et al., Biomed. Opt. Express 2014
- **Fourier Ptychnet** -- Jiang et al., Biomed. Opt. Express 2018 (DL)
- **prDeep** or **PhaseNet** from the coherent pool would also be more fitting

The learning materials (03_reconstruction_algorithms.md) actually document
FPM-specific solvers (Sequential Phase Retrieval, Gradient Descent FPM,
Fourier Ptychnet) which do NOT match the leaderboard algorithms (Richardson-Lucy,
PnP-FISTA, CARE, Restormer).

However, changing the algorithm pool requires either:
(a) adding an FPM-specific entry to `_VARIANT_OVERRIDES`, or
(b) routing microscopy+Photon to a different sub-category

Since microscopy+Photon covers many modalities (confocal, lightsheet, etc.),
option (a) is preferred.

## Code Changes Needed

**Option A (recommended): Add FPM-specific variant override**

In `_algorithm_catalog.py`, add to `_VARIANT_OVERRIDES`:

```python
"fpm": [
    {"name": "AP (Alternating Projections)", "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Zheng et al., Nat. Photonics 2013"},
    {"name": "Gradient Descent FPM",         "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Tian et al., Biomed. Opt. Express 2014"},
    {"name": "Fourier Ptychnet",             "type": "Deep Learning", "mask_aware": False, "params": "7M",   "source": "Jiang et al., Biomed. Opt. Express 2018"},
    {"name": "PtychoDV",                     "type": "Transformer",   "mask_aware": True,  "params": "12M",  "source": "Chung et al., Optica 2023"},
],
```

Also add "fpm" score entry to `CATEGORY_REAL_SCORES` if FPM-specific scores
are desired, or let it fall through to "microscopy" scores (acceptable).
