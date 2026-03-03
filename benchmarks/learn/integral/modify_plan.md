# Modify Plan -- integral

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

**Acceptable but generic.** Integral photography (light field captured through a microlens array) falls under computational imaging. The generic `computational` pool provides reasonable general-purpose reconstruction algorithms. However, integral imaging has dedicated methods:

- Classical: Shift-and-add / MFBP (Multi-Focus Back-Projection) for depth reconstruction
- PnP: PnP with depth-aware regularization
- Deep Learning: Light field-specific networks (e.g., LFAttNet for depth estimation, LFSSR for super-resolution)
- Transformer: DistgSSR (Wang et al., CVPR 2022) or similar light field super-resolution transformers

The generic Tikhonov/PnP-RED/DIP/SwinIR set is not inappropriate for an image reconstruction benchmark, but it lacks domain specificity. Since integral photography and light_field share similar physics (both are plenoptic/light field systems), they should ideally have light-field-specific algorithms.

## Recommendation

**Code changes needed** -- add a `_VARIANT_OVERRIDES` entry for `integral` with light-field-specific reconstruction algorithms.

### Proposed change in `_algorithm_catalog.py`:

```python
"integral": [
    {"name": "Shift-and-Add", "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Ng et al., Stanford Tech Report 2005"},
    {"name": "PnP-LF",        "type": "PnP",           "mask_aware": True,  "params": "0",   "source": "PnP-ADMM with LF prior"},
    {"name": "LFAttNet",      "type": "Deep Learning", "mask_aware": False, "params": "4.5M","source": "Tsai et al., IEEE TIP 2020"},
    {"name": "DistgSSR",      "type": "Transformer",   "mask_aware": True,  "params": "12M", "source": "Wang et al., CVPR 2022"},
],
```

Add corresponding `CATEGORY_REAL_SCORES["integral"]`.

### Files to modify:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
