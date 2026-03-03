# Modify Plan: gpr

## Current State

- **Category:** remote_sensing
- **Carrier:** RF
- **Score key:** remote_sensing
- **Algorithms assigned:**
  1. Matched Filter (Classical) -- Standard SAR focusing
  2. SAR-BM3D (PnP) -- Parrilli et al., IEEE TGRS 2012
  3. SAR-DRN (Deep Learning) -- Zhang et al., RS 2018
  4. SAR-CAM (Transformer) -- Cross-attention SAR, 2024

## Assessment

**Partially appropriate -- SAR algorithms applied to GPR is questionable**

Ground-Penetrating Radar and SAR are both RF-based imaging modalities, but
their physics and reconstruction approaches differ significantly:

- **GPR** processes subsurface reflections using migration algorithms (Kirchhoff
  migration, reverse-time migration, f-k migration). The inverse problem is
  recovering subsurface structure from time-domain radar traces (B-scans).
- **SAR** processes synthetic aperture data using range-Doppler focusing. SAR
  operates in the far field; GPR operates in the near field.

The current algorithms are SAR-specific:
- **Matched Filter / SAR focusing**: SAR terminology, not GPR.
- **SAR-BM3D, SAR-DRN, SAR-CAM**: All explicitly SAR-named methods.

Better GPR-specific algorithms:
- **Kirchhoff Migration** (classical, standard GPR)
- **Reverse-Time Migration (RTM)** (wave-equation based)
- **GPR-Net** or similar subsurface imaging CNNs
- **GPR-Transformer** (attention-based hyperbola detection)

## Code Changes Needed

**Add GPR-specific variant override in `_algorithm_catalog.py`:**

```python
"gpr": [
    {"name": "Kirchhoff Migration", "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Schneider, Geophysics 1978"},
    {"name": "RTM",                 "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Fisher et al., Geophysics 1992"},
    {"name": "GPR-RCNN",            "type": "Deep Learning", "mask_aware": False, "params": "8M",   "source": "Pham & Lefeuvre, NDT&E Int. 2022"},
    {"name": "HyperDet",            "type": "Transformer",   "mask_aware": True,  "params": "12M",  "source": "Lei et al., IEEE TGRS 2023"},
],
```
