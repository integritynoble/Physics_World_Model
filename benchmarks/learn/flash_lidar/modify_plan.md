# Modify Plan: flash_lidar

## Current Assignment
- **Category:** depth_imaging
- **Carrier:** Photon
- **Score key:** depth_imaging
- **Algorithms:** SGM (Classical), PnP-ADMM (PnP), PSMNet (Deep Learning), RAFT-Stereo (Transformer)

## Assessment

The algorithms are **inappropriate**. The depth_imaging category pool contains
stereo vision / passive depth estimation algorithms. Flash LiDAR is an active
time-of-flight (ToF) depth sensor that floods the scene with a laser pulse and
measures per-pixel return times on a SPAD (single-photon avalanche diode) array.

**Problems:**
1. **SGM** (Semi-Global Matching) is a stereo disparity algorithm. Flash LiDAR
   has no stereo baseline; depth comes from photon arrival times.
2. **PSMNet** and **RAFT-Stereo** are stereo matching networks that require
   left/right image pairs. Flash LiDAR produces a single-sensor time-resolved
   histogram per pixel.
3. **PnP-ADMM** is generic enough but the context (stereo depth) is wrong.
4. The actual flash LiDAR reconstruction task is: given a noisy photon timing
   histogram per pixel (often with only a few photon counts), estimate the
   depth map. This requires histogram peak detection, Poisson denoising, and
   handling of background/pile-up effects.

## Recommended Changes

Add a variant-specific override:

```python
"flash_lidar": [
    {"name": "Log-Matched Filter",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Coates, J. Phys. D 1968"},
    {"name": "PnP-SPIRAL",          "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Harmany et al., IEEE TIP 2012"},
    {"name": "Deep-SPAD",           "type": "Deep Learning", "mask_aware": False, "params": "5M",   "source": "Lindell et al., SIGGRAPH 2018"},
    {"name": "SPADNet",             "type": "Deep Learning", "mask_aware": True,  "params": "8M",   "source": "Ruget et al., Opt. Express 2021"},
],
```

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Add `"flash_lidar"` to `_VARIANT_OVERRIDES`
  - Add `"flash_lidar"` to `CATEGORY_REAL_SCORES`
