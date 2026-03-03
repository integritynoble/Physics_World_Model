# Modify Plan: structured_light

## Current State
- **Category:** depth_imaging
- **Carrier:** Photon
- **Score key:** depth_imaging
- **Algorithms:**
  1. SGM (Classical) -- Hirschmuller, TPAMI 2007
  2. PnP-ADMM (PnP) -- ADMM + denoiser prior
  3. PSMNet (Deep Learning) -- Chang & Chen, CVPR 2018
  4. RAFT-Stereo (Transformer) -- Lipson et al., 3DV 2021

## Assessment

**Problem:** The depth_imaging pool contains stereo matching algorithms, which are not the right approach for structured light depth cameras. Structured light projects known patterns (stripes, dots, Gray codes) and decodes depth from pattern deformation -- this is fundamentally different from stereo disparity estimation.

- **SGM (Semi-Global Matching)** is a stereo matching algorithm that finds correspondences between left/right images. Structured light does not use stereo matching -- it decodes projected patterns.
- **PSMNet** is a deep stereo matching network. Same issue.
- **RAFT-Stereo** is a stereo optical flow network. Same issue.
- **PnP-ADMM** is generic enough to apply but does not capture structured light decoding.

Note: Some structured light systems (like Intel RealSense) do use stereo matching as part of their pipeline, but the core structured light approach (Gray code, phase shifting, dot pattern) is pattern decoding, not stereo.

Appropriate structured light algorithms include:
1. **Phase Shifting** (Classical) -- Srinivasan et al., 1984; multi-frame phase unwrapping
2. **Gray Code** (Classical) -- Inokuchi et al., 1984; binary pattern decoding
3. **FPP-Net** (Deep Learning) -- Feng et al., Opt. Express 2019; fringe projection profilometry with DL
4. **DeepSL** (Deep Learning) -- Riegler et al., ECCV 2020; deep structured light

## Required Changes

Add `structured_light` to `_VARIANT_OVERRIDES` in `_algorithm_catalog.py`:

```python
"structured_light": [
    {"name": "Phase Shifting",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Srinivasan et al., Appl. Opt. 1984"},
    {"name": "Gray Code",       "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Inokuchi et al., 1984"},
    {"name": "PnP-ADMM",        "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "ADMM + denoiser prior"},
    {"name": "FPP-Net",         "type": "Deep Learning", "mask_aware": False, "params": "4M",   "source": "Feng et al., Opt. Express 2019"},
],
```

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`: Add variant override for `structured_light`
