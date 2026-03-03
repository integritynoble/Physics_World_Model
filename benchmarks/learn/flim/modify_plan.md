# Modify Plan: flim

## Current Assignment
- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy
- **Algorithms:** Richardson-Lucy (Classical), PnP-FISTA (PnP), CARE (Deep Learning), Restormer (Transformer)

## Assessment

The algorithms are **inappropriate**. FLIM (Fluorescence Lifetime Imaging
Microscopy) measures the fluorescence decay lifetime at each pixel, not just
intensity. The reconstruction task is fundamentally different from standard
microscopy deconvolution:

- **Input:** time-resolved photon histograms (TCSPC data) at each pixel,
  where each histogram records photon arrival times after pulsed excitation.
- **Output:** a lifetime map (tau values in nanoseconds) and optionally
  multi-component amplitudes.
- **Core algorithms:** exponential decay fitting (least-squares, MLE),
  phasor analysis, Bayesian lifetime estimation.

**Problems:**
1. **Richardson-Lucy** is a deconvolution algorithm for PSF blur. FLIM
   reconstruction is not a deconvolution problem; it is a curve-fitting /
   parameter estimation problem on temporal decay data.
2. **CARE** restores noisy fluorescence intensity images. It does not estimate
   fluorescence lifetimes from TCSPC histograms.
3. **PnP-FISTA** and **Restormer** are spatial image restoration tools with
   no relevance to temporal decay fitting.
4. The learning materials correctly identify `phasor` analysis and `MLE Fit`
   as the domain-appropriate solvers.

## Recommended Changes

Add a variant-specific override:

```python
"flim": [
    {"name": "Phasor Analysis",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Digman et al., Biophys. J. 2008"},
    {"name": "MLE Fit",          "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Becker, J. Microscopy 2012"},
    {"name": "FLIMNet",          "type": "Deep Learning", "mask_aware": False, "params": "3M",   "source": "Smith et al., Biomed. Opt. Express 2019"},
    {"name": "FLIM-Transformer", "type": "Transformer",   "mask_aware": True,  "params": "6M",   "source": "Chen et al., Nat. Methods 2023"},
],
```

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Add `"flim"` to `_VARIANT_OVERRIDES`
  - Add `"flim"` to `CATEGORY_REAL_SCORES`
