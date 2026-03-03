# Modify Plan -- lensless

## Current State

- **Category:** computational_photography
- **Carrier:** Photon
- **Score key:** computational_photography
- **Algorithms:**
  1. Wiener-Deconv (Classical) -- Analytical baseline
  2. PnP-FFDNet (PnP) -- Zhang et al., 2017
  3. HDR-CNN (Deep Learning) -- Eilertsen et al., ACM TOG 2017
  4. Uformer (Transformer) -- Wang et al., CVPR 2022

## Assessment

**Problem:** Lensless imaging (diffuser camera) reconstruction is a deconvolution/inverse problem that recovers a scene from a coded/diffused measurement. While Wiener deconvolution and PnP methods are reasonable, **HDR-CNN is inappropriate** -- it is designed for HDR tone mapping / LDR-to-HDR conversion, which is a completely different problem from lensless image reconstruction.

**Appropriate domain algorithms would include:**
- Classical: Wiener deconvolution or ADMM (Antipa et al., Optica 2018) -- fine as-is
- PnP: PnP-ADMM with learned denoisers (appropriate for this problem)
- Deep Learning: FlatNet (Khan et al., IEEE TPAMI 2020) or LenslessNet -- physics-informed networks for lensless reconstruction
- Transformer: A general restoration transformer (Uformer is acceptable) or lensless-specific architecture

The main issue is HDR-CNN, which has no relevance to lensless imaging reconstruction.

## Recommendation

**Code changes needed** -- add a `_VARIANT_OVERRIDES` entry for `lensless` in `_algorithm_catalog.py` to replace HDR-CNN with a lensless-specific deep learning method.

### Proposed change in `_algorithm_catalog.py`:

```python
"lensless": [
    {"name": "Wiener-ADMM",  "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Antipa et al., Optica 2018"},
    {"name": "PnP-ADMM",     "type": "PnP",           "mask_aware": True,  "params": "0",   "source": "Monakhova et al., Opt. Express 2019"},
    {"name": "FlatNet",      "type": "Deep Learning", "mask_aware": False, "params": "4.2M","source": "Khan et al., IEEE TPAMI 2020"},
    {"name": "Uformer",      "type": "Transformer",   "mask_aware": True,  "params": "20M", "source": "Wang et al., CVPR 2022"},
],
```

Add corresponding `CATEGORY_REAL_SCORES["lensless"]`.

### Files to modify:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
