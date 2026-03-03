# Modify Plan: endoscopy

## Current Assignment
- **Category:** medical
- **Carrier:** Photon
- **Score key:** clinical_optics (routed via carrier)
- **Algorithms:** FFT-OCT (Classical), BM4D (PnP), Speckle-DenoiseNet (Deep Learning), OCTA-Net (Transformer)

## Assessment

The algorithms are **inappropriate**. Carrier-based routing sends endoscopy
to the `clinical_optics` pool, which contains OCT and retinal imaging algorithms.
Fiber bundle endoscopy has a completely different imaging physics:

- The image is transmitted through a coherent fiber bundle, causing a honeycomb
  pattern artifact (inter-core spacing) and each core has its own PSF.
- The reconstruction task is **fiber bundle deconvolution** (removing the
  honeycomb pattern and per-core PSF blur), not OCT processing.

**Problems:**
1. **FFT-OCT** is an OCT-specific spectral domain processing step. It has no
   relevance to fiber bundle endoscopy.
2. **OCTA-Net** is for retinal vasculature segmentation from OCT angiography.
3. **Speckle-DenoiseNet** is for OCT speckle, not fiber bundle artifacts.
4. **BM4D** is a volumetric denoiser; marginally applicable for general
   denoising but misses the core fiber-bundle-specific reconstruction.
5. The learning materials correctly identify `tv_fista` as the default solver.

## Recommended Changes

Add a variant-specific override:

```python
"endoscopy": [
    {"name": "Interpolation",    "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Nearest-neighbor/Voronoi baseline"},
    {"name": "PnP-BM3D",         "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Fiber deconv + BM3D prior"},
    {"name": "FiberNet",         "type": "Deep Learning", "mask_aware": False, "params": "4M",   "source": "Ravì et al., MICCAI 2018"},
    {"name": "EndoL2H",          "type": "Deep Learning", "mask_aware": True,  "params": "11M",  "source": "Luo et al., IEEE TMI 2023"},
],
```

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Add `"endoscopy"` to `_VARIANT_OVERRIDES`
  - Add `"endoscopy"` to `CATEGORY_REAL_SCORES`
