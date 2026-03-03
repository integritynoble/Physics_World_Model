# Modify Plan -- light_field

## Current State

- **Category:** computational
- **Carrier:** Photon
- **Score key:** computational
- **Algorithms:**
  1. Tikhonov (Classical) -- Analytical baseline
  2. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  3. Deep Image Prior (Deep Learning) -- Ulyanov et al., CVPR 2018
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

**Problem:** Light field imaging has a well-established reconstruction literature with domain-specific algorithms. The generic `computational` pool (Tikhonov, PnP-RED, DIP, SwinIR) does not reflect the actual methods used in light field reconstruction. The key tasks in light field processing are:
- Angular/spatial super-resolution
- Depth estimation from the 4D light field
- View synthesis / novel view generation
- Refocusing and all-in-focus image generation

**Appropriate domain algorithms would include:**
- Classical: Shift-and-add refocusing or depth-from-focus (Ng et al., Stanford Tech Report 2005)
- PnP: PnP with disparity-guided regularization
- Deep Learning: LFNet (Wang et al., IEEE TPAMI 2020) or LFSSR (light field spatial super-resolution)
- Transformer: DistgSSR (Wang et al., CVPR 2022) or LFT (Light Field Transformer)

This is the same issue as `integral` -- both are plenoptic/light field modalities that need light field-specific algorithms rather than generic inverse problem solvers.

## Recommendation

**Code changes needed** -- add a `_VARIANT_OVERRIDES` entry for `light_field` in `_algorithm_catalog.py` with light-field-specific algorithms.

### Proposed change in `_algorithm_catalog.py`:

```python
"light_field": [
    {"name": "Shift-and-Sum", "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Ng et al., Stanford Tech Report 2005"},
    {"name": "PnP-LF",        "type": "PnP",           "mask_aware": True,  "params": "0",   "source": "PnP-ADMM with angular prior"},
    {"name": "LFNet",         "type": "Deep Learning", "mask_aware": False, "params": "5.8M","source": "Wang et al., IEEE TPAMI 2020"},
    {"name": "DistgSSR",      "type": "Transformer",   "mask_aware": True,  "params": "12M", "source": "Wang et al., CVPR 2022"},
],
```

Add corresponding `CATEGORY_REAL_SCORES["light_field"]`.

### Files to modify:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
