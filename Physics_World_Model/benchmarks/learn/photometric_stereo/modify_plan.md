# Modify Plan: photometric_stereo

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

Photometric stereo reconstructs surface normals from multiple images captured under varying illumination directions. The category `depth_imaging` is reasonable (it produces depth/surface information). However, the algorithms are **entirely stereo matching / stereo depth estimation** methods:

- **SGM** (Semi-Global Matching) -- a stereo correspondence algorithm for disparity estimation from binocular images. Not applicable to photometric stereo.
- **PSMNet** -- a deep stereo matching network. Not applicable.
- **RAFT-Stereo** -- a stereo depth estimation network. Not applicable.
- **PnP-ADMM** -- generic enough to be applicable, but cited as stereo-related.

Photometric stereo requires **fundamentally different** algorithms that solve the surface normal estimation problem from illumination variation, not stereo correspondence:

**Appropriate photometric stereo algorithms:**
- Least-Squares Normal Estimation (Classical) -- Woodham, Opt. Eng. 1980
- Robust PCA / Sparse + Low-Rank (PnP) -- Wu et al., ECCV 2010
- CNN-PS (Deep Learning) -- Ikehata, ECCV 2018
- PS-Transformer (Transformer) -- Li et al., ECCV 2022 (UniPS / SDM-UniPS)

This is a **severe mismatch** -- stereo matching algorithms cannot solve photometric stereo problems.

## Required Changes

Add a variant override in `_algorithm_catalog.py` for `photometric_stereo` with surface-normal-estimation algorithms.

### Files to modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py` -- add `_VARIANT_OVERRIDES["photometric_stereo"]` with photometric stereo algorithms (LS Normal Estimation, Robust PCA, CNN-PS, PS-Transformer/UniPS)
