# Modify Plan: spectral_ct

## Current State
- **Category:** medical
- **Carrier:** X-ray
- **Score key:** medical
- **Algorithms:**
  1. FBP (Classical) -- Analytical baseline
  2. PnP-ADMM (PnP) -- Venkatakrishnan et al., 2013
  3. FBPConvNet (Deep Learning) -- Jin et al., IEEE TIP 2017
  4. Learned Primal-Dual (Deep Unrolling) -- Adler & Oktem, IEEE TMI 2018

## Assessment

**Problem:** Spectral CT (photon-counting CT) gets the generic medical/CT pool. While these are valid CT reconstruction algorithms, spectral CT has a unique additional challenge: material decomposition across energy bins. The reconstruction problem is not just image reconstruction but also energy-dependent material separation.

- **FBP** can be applied per energy bin but ignores cross-channel information.
- **PnP-ADMM** is a generic method, applicable but not spectral-specific.
- **FBPConvNet** was designed for conventional CT, not multi-energy.
- **Learned Primal-Dual** was designed for conventional CT, not multi-energy.

Appropriate spectral CT algorithms include:
1. **FBP + Material Decomposition** (Classical) -- Alvarez & Macovski, Phys. Med. Biol. 1976; sinogram-domain decomposition
2. **One-Step Spectral CT** (Iterative) -- Long & Fessler, IEEE TMI 2014; joint reconstruction + decomposition
3. **Butterfly-Net** (Deep Learning) -- Fan et al., SIAM JSC 2019; multi-scale spectral CT
4. **DECT-MULTRA** (Dictionary) -- Zeng et al., IEEE TMI 2021; multi-energy learned transform

However, the existing algorithms are still applicable as a baseline (FBP and iterative methods work on each energy channel), and the mismatch is more about missing spectral-specific methods than having wrong methods. This is a "could be better" situation rather than a "fundamentally wrong" situation.

## Required Changes

Optionally add `spectral_ct` to `_VARIANT_OVERRIDES` to include material-decomposition-aware algorithms. This is a low-priority enhancement.

```python
"spectral_ct": [
    {"name": "FBP",                 "type": "Classical",      "mask_aware": True,  "params": "0",    "source": "Analytical baseline (per-bin)"},
    {"name": "One-Step Spectral",   "type": "Iterative",      "mask_aware": True,  "params": "0",    "source": "Long & Fessler, IEEE TMI 2014"},
    {"name": "PnP-ADMM",            "type": "PnP",            "mask_aware": True,  "params": "0",    "source": "Venkatakrishnan et al., 2013"},
    {"name": "Butterfly-Net",       "type": "Deep Learning",  "mask_aware": False, "params": "5M",   "source": "Fan et al., SIAM JSC 2019"},
],
```

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`: Optionally add variant override for `spectral_ct`
