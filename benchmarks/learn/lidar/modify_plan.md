# Modify Plan -- lidar

## Current State (Updated 2026-03-03)

- **Category:** depth_imaging
- **Carrier:** Photon
- **Score key:** depth_imaging
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["lidar"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. Bilateral Filter (Classical) -- Tomasi & Manduchi, ICCV 1998
  2. PnP-ADMM (PnP) -- Venkatakrishnan et al., 2013
  3. RandLA-Net (Deep Learning) -- Hu et al., CVPR 2020
  4. Point Transformer (Transformer) -- Zhao et al., ICCV 2021

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the stereo depth estimation pool (SGM, PnP-ADMM,
PSMNet, RAFT-Stereo) with LiDAR-appropriate algorithms. SGM, PSMNet, and
RAFT-Stereo are binocular stereo matching methods fundamentally inapplicable
to LiDAR range measurement data. The new set includes point cloud processing
methods (RandLA-Net, Point Transformer) alongside general-purpose depth
processing (Bilateral Filter, PnP-ADMM).

## Changes Applied

- Added `_VARIANT_OVERRIDES["lidar"]` with four LiDAR-appropriate algorithms
- Bilateral Filter: edge-preserving depth map smoothing
- PnP-ADMM: plug-and-play priors for depth completion
- RandLA-Net: efficient random sampling network for large-scale point clouds
- Point Transformer: self-attention architecture for 3D point cloud processing

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
