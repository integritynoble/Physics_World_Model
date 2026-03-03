# Modify Plan -- insar

## Current State

- **Category:** remote_sensing
- **Carrier:** RF
- **Score key:** remote_sensing
- **Algorithms:**
  1. Matched Filter (Classical) -- Standard SAR focusing
  2. SAR-BM3D (PnP) -- Parrilli et al., IEEE TGRS 2012
  3. SAR-DRN (Deep Learning) -- Zhang et al., RS 2018
  4. SAR-CAM (Transformer) -- Cross-attention SAR, 2024

## Assessment

**Acceptable but imperfect.** InSAR (Interferometric SAR) shares the RF carrier with SAR and the routing correctly gives SAR-family algorithms. However, InSAR has a unique reconstruction challenge beyond SAR image formation: **phase unwrapping** and **interferometric phase estimation**. The current algorithms address SAR image denoising/focusing but miss the InSAR-specific phase processing step.

Ideal InSAR algorithms would include:
- Classical: Goldstein branch-cut (Goldstein et al., Radio Science 1988) or MCF (Minimum Cost Flow) phase unwrapping
- PnP/filtering: InSAR-BM3D or Goldstein phase filter
- Deep Learning: PhaseNet/DeepInSAR (deep phase unwrapping)
- Transformer: InSAR-specific transformer for phase unwrapping

That said, the current SAR pool is not egregiously wrong -- SAR focusing is a prerequisite for InSAR, and these algorithms can be interpreted as operating on the SAR image formation step. The mismatch is moderate.

## Recommendation

**Code changes needed** -- add a `_VARIANT_OVERRIDES` entry for `insar` with InSAR-specific algorithms that address the phase unwrapping problem.

### Proposed change in `_algorithm_catalog.py`:

```python
"insar": [
    {"name": "Goldstein-MCF",   "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Goldstein et al., Radio Science 1988"},
    {"name": "InSAR-BM3D",      "type": "PnP",           "mask_aware": True,  "params": "0",   "source": "Deledalle et al., IEEE TGRS 2015"},
    {"name": "PhaseNet",         "type": "Deep Learning", "mask_aware": False, "params": "1.8M","source": "Sica et al., IEEE TGRS 2021"},
    {"name": "InSAR-Former",     "type": "Transformer",   "mask_aware": True,  "params": "10M", "source": "Wu et al., IEEE TGRS 2024"},
],
```

Add corresponding `CATEGORY_REAL_SCORES["insar"]`.

### Files to modify:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
