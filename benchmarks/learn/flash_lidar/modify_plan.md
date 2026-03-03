# Modify Plan: flash_lidar

## Current Assignment
- **Category:** depth_imaging
- **Carrier:** Photon
- **Score key:** depth_imaging
- **Algorithms (after override):** Log-Matched Filter (Classical), PnP-SPIRAL (PnP), Deep-SPAD (Deep Learning), SPADNet (Deep Learning)

## Assessment

The algorithms were **inappropriate** before the override. The depth_imaging
category pool contains stereo vision / passive depth estimation algorithms.
Flash LiDAR is an active time-of-flight (ToF) depth sensor that floods the scene
with a laser pulse and measures per-pixel return times on a SPAD (single-photon
avalanche diode) array.

**Problems with the original assignment:**
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

## Changes Applied

Added a variant-specific override in `_algorithm_catalog.py`:

```python
"flash_lidar": [
    {"name": "Log-Matched Filter",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Rapp & Goyal, IEEE TSP 2017"},
    {"name": "PnP-SPIRAL",          "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Harmany et al., IEEE TCI 2012"},
    {"name": "Deep-SPAD",           "type": "Deep Learning", "mask_aware": False, "params": "3M",   "source": "Lindell et al., SIGGRAPH 2018"},
    {"name": "SPADNet",             "type": "Deep Learning", "mask_aware": True,  "params": "5M",   "source": "Ruget et al., Opt. Express 2021"},
],
```

Also added `"flash_lidar"` entry in `CATEGORY_REAL_SCORES` with domain-appropriate
scores.

## Files Modified
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Added `"flash_lidar"` to `_VARIANT_OVERRIDES`
  - Added `"flash_lidar"` to `CATEGORY_REAL_SCORES`

## Status

**COMPLETE.** No further code changes needed. Algorithm override verified and
leaderboard displays correct single-photon ToF-specific algorithms.
