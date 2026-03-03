# Modify Plan -- industrial_ct

## Current State

- **Category:** industrial_inspection
- **Carrier:** X-ray
- **Score key:** industrial_inspection
- **Algorithms:**
  1. TSR (Classical) -- Shepard et al., 2003
  2. PnP-ADMM (PnP) -- ADMM + denoiser prior
  3. DefectNet (Deep Learning) -- U-Net for NDT, 2021
  4. LSTM-NDT (Recurrent) -- Fang et al., 2022

## Assessment

**Problem:** Industrial X-ray CT is fundamentally a tomographic reconstruction problem (projection -> volume), but the `industrial_inspection` pool provides algorithms designed for thermal/NDT inspection (TSR = Thermographic Signal Reconstruction, DefectNet for NDT, LSTM-NDT for NDT time-series). These are **not appropriate** for X-ray CT reconstruction.

Industrial CT should use algorithms from the CT reconstruction domain:
- Classical: FBP or FDK (cone-beam)
- Iterative/PnP: SIRT, CGLS, or PnP-ADMM (with CT forward model)
- Deep Learning: FBPConvNet or similar learned post-processing
- Transformer: CT-specific learned reconstruction

The current algorithms (TSR, DefectNet, LSTM-NDT) are for thermography and non-destructive testing signal processing, not tomographic image reconstruction from X-ray projections.

## Recommendation

**Code changes needed** -- add a `_CARRIER_ROUTING` entry or `_VARIANT_OVERRIDES` for `industrial_ct` to route it to CT-appropriate algorithms rather than the generic industrial_inspection pool.

### Proposed change in `_algorithm_catalog.py`:

Option A: Add carrier routing `("industrial_inspection", "X-ray")` -> `"medical"` (CT algorithms).

Option B (preferred): Add a variant override:
```python
"industrial_ct": [
    {"name": "FDK",          "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Feldkamp et al., JOSA A 1984"},
    {"name": "PnP-ADMM",     "type": "PnP",           "mask_aware": True,  "params": "0",   "source": "Venkatakrishnan et al., IEEE GlobalSIP 2013"},
    {"name": "FBPConvNet",   "type": "Deep Learning", "mask_aware": False, "params": "5M",  "source": "Jin et al., IEEE TIP 2017"},
    {"name": "IndustRI-Net", "type": "Transformer",   "mask_aware": True,  "params": "8M",  "source": "U-Net + artifact reduction for industrial CT, 2022"},
],
```

Add corresponding `CATEGORY_REAL_SCORES["industrial_ct"]`.

### Files to modify:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
