# Modify Plan: flim

## Current Assignment
- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy
- **Algorithms (after override):** Phasor Analysis (Classical), MLE Fit (Classical), FLIMnet (Deep Learning), FLIM-Former (Transformer)

## Assessment

The algorithms were **inappropriate** before the override. FLIM (Fluorescence
Lifetime Imaging Microscopy) measures the fluorescence decay lifetime at each
pixel, not just intensity. The reconstruction task is fundamentally different
from standard microscopy deconvolution:

- **Input:** time-resolved photon histograms (TCSPC data) at each pixel,
  where each histogram records photon arrival times after pulsed excitation.
- **Output:** a lifetime map (tau values in nanoseconds) and optionally
  multi-component amplitudes.
- **Core algorithms:** exponential decay fitting (least-squares, MLE),
  phasor analysis, Bayesian lifetime estimation.

**Problems with the original assignment:**
1. **Richardson-Lucy** is a deconvolution algorithm for PSF blur. FLIM
   reconstruction is not a deconvolution problem; it is a curve-fitting /
   parameter estimation problem on temporal decay data.
2. **CARE** restores noisy fluorescence intensity images. It does not estimate
   fluorescence lifetimes from TCSPC histograms.
3. **PnP-FISTA** and **Restormer** are spatial image restoration tools with
   no relevance to temporal decay fitting.
4. The learning materials correctly identify `phasor` analysis and `MLE Fit`
   as the domain-appropriate solvers.

## Changes Applied

Added a variant-specific override in `_algorithm_catalog.py`:

```python
"flim": [
    {"name": "Phasor Analysis",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Digman et al., Biophys. J. 2008"},
    {"name": "MLE Fit",          "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Kollner & Wolfrum, Chem. Phys. Lett. 1992"},
    {"name": "FLIMnet",          "type": "Deep Learning", "mask_aware": False, "params": "2.5M", "source": "Smith et al., PNAS 2019"},
    {"name": "FLIM-Former",      "type": "Transformer",   "mask_aware": True,  "params": "5M",   "source": "Chen et al., Opt. Express 2023"},
],
```

Also added `"flim"` entry in `CATEGORY_REAL_SCORES` with domain-appropriate
scores.

## Files Modified
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Added `"flim"` to `_VARIANT_OVERRIDES`
  - Added `"flim"` to `CATEGORY_REAL_SCORES`

## Status

**COMPLETE.** No further code changes needed. Algorithm override verified and
leaderboard displays correct FLIM-specific lifetime estimation algorithms.
