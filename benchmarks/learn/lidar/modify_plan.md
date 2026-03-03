# Modify Plan -- lidar

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

**Problem:** The `depth_imaging` pool contains algorithms designed for **stereo depth estimation** (SGM = Semi-Global Matching, PSMNet = Pyramid Stereo Matching Network, RAFT-Stereo = stereo optical flow). These are fundamentally different from LiDAR point cloud reconstruction/processing.

LiDAR produces direct range measurements via time-of-flight (ToF), not stereo disparity. The reconstruction tasks for LiDAR include:
- Point cloud densification/completion (upsampling sparse LiDAR returns)
- Depth completion (filling in missing measurements)
- Denoising (removing noise from ToF measurements)
- Surface reconstruction from point clouds

**Appropriate algorithms:**
- Classical: Bilateral filtering / moving least squares for point cloud smoothing
- Optimization: PnP with depth map priors
- Deep Learning: PointNet++ (Qi et al., NeurIPS 2017) or sparse-to-dense depth completion networks
- Transformer: PoinTr (Yu et al., ICCV 2021) or depth completion transformers

SGM and PSMNet are for stereo vision and have no applicability to LiDAR.

## Recommendation

**Code changes needed** -- add a `_VARIANT_OVERRIDES` entry for `lidar` in `_algorithm_catalog.py` with LiDAR-specific algorithms.

### Proposed change in `_algorithm_catalog.py`:

```python
"lidar": [
    {"name": "MLS",          "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Levin, SIGGRAPH 2004"},
    {"name": "PnP-DepthPrior","type": "PnP",          "mask_aware": True,  "params": "0",   "source": "PnP-ADMM with depth prior"},
    {"name": "SparseConvNet", "type": "Deep Learning", "mask_aware": False, "params": "5M",  "source": "Uhrig et al., 3DV 2017"},
    {"name": "PoinTr",        "type": "Transformer",   "mask_aware": True,  "params": "12M", "source": "Yu et al., ICCV 2021"},
],
```

Add corresponding `CATEGORY_REAL_SCORES["lidar"]`.

### Files to modify:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
