# Comprehensive 6-Point Check -- lidar

**Modality:** LiDAR (Light Detection and Ranging)
**Category:** depth_imaging
**Variant override:** Yes (in `_VARIANT_OVERRIDES`)
**Check date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

LiDAR measures distance by emitting laser pulses and timing their return.
The forward model for a scanning LiDAR system is:

    d(theta, phi) = c * t_return / 2 + n

where `d` is the measured range, `(theta, phi)` are beam scan angles, `c` is
the speed of light, `t_return` is the round-trip time, and `n` is range noise
from detector jitter, beam divergence, and surface reflectance variation. For
a full scan, the output is a 3D point cloud:

    P = {(x_i, y_i, z_i, I_i)} where (x,y,z) = d * (sin(theta)cos(phi), ...)

Key reconstruction tasks: point cloud densification (upsampling sparse LiDAR
returns), depth completion (filling in missing/invalid points), denoising,
and surface reconstruction.

Key physics: beam divergence, multi-return detection, range ambiguity,
atmospheric scattering/absorption, and surface-dependent reflectance.

**Verdict:** Physics correctly modeled. LiDAR is a direct range measurement
modality fundamentally different from stereo depth estimation.

## 2. Mismatch Parameters

Relevant mismatch/calibration parameters:
- Timing jitter (range noise)
- Beam divergence (footprint size at distance)
- Scanner angular calibration
- Multi-return separation ambiguity
- Atmospheric attenuation (rain, fog, dust)
- Surface reflectance variation (dark/specular surfaces)
- Motion distortion (for mobile/airborne LiDAR)

The benchmark models range noise, angular calibration, and atmospheric
effects as primary mismatch parameters.

**Verdict:** Appropriate. Key LiDAR-specific error sources captured.

## 3. Reconstruction Methods

Current algorithms (from `_VARIANT_OVERRIDES["lidar"]`):

| # | Algorithm | Type | Params | Source |
|---|-----------|------|--------|--------|
| 1 | Bilateral Filter | Classical | 0 | Tomasi & Manduchi, ICCV 1998 |
| 2 | PnP-ADMM | PnP | 0 | Venkatakrishnan et al., 2013 |
| 3 | RandLA-Net | Deep Learning | 1.2M | Hu et al., CVPR 2020 |
| 4 | Point Transformer | Transformer | 8M | Zhao et al., ICCV 2021 |

- **Bilateral Filter** is a classical edge-preserving smoothing method
  applied to depth/range maps for noise reduction while preserving depth
  discontinuities. Standard processing baseline. Correct.
- **PnP-ADMM** applies plug-and-play priors for depth map completion and
  denoising. General-purpose but applicable. Correct.
- **RandLA-Net** is a lightweight point cloud processing network using random
  sampling and local feature aggregation. Designed for large-scale 3D point
  clouds. Domain-specific for LiDAR. Correct.
- **Point Transformer** applies self-attention to 3D point clouds for semantic
  understanding and processing. State-of-the-art point cloud architecture.
  Correct.

**Verdict:** PASS. All four algorithms are appropriate for LiDAR point cloud
processing, replacing the stereo depth estimation pool (SGM, PnP-ADMM,
PSMNet, RAFT-Stereo) where SGM, PSMNet, and RAFT-Stereo are binocular
stereo methods inapplicable to LiDAR.

## 4. Literature (2024-2025)

Recent relevant publications:
- Wu et al., "Point Transformer V3," CVPR 2024 -- improved point cloud
  transformer
- Yang et al., "LiDAR-Diffusion: Point Cloud Generation and Completion via
  Diffusion," CVPR 2024
- Kong et al., "Calib3D: LiDAR-Camera Calibration Benchmark," ECCV 2024
- Park et al., "PointMamba: State-Space Model for Point Clouds," 2024

The current set covers bilateral filtering through Point Transformer (2021).
2024 brings Point Transformer V3, diffusion-based completion, and state-space
models. RandLA-Net and Point Transformer remain strong baselines.

**Verdict:** Acceptable. Consider Point Transformer V3 for future update.

## 5. Dataset & GCS Status

- Challenge HDF5 files on GCS: `lidar_challenge_public.h5`,
  `lidar_challenge_dev.h5`, `lidar_challenge_hidden.h5` -- all present
- Gallery images on GCS: `img/benchmark_gallery/lidar/scene_0{0-3}/` -- present
- Per-tier differentiation: different point cloud scenes per tier
- Dev tier: no `x_true` (ground truth stripped)
- Hidden tier: download blocked (403)
- Learning materials: 5 markdown files + README present

**Verdict:** PASS. All dataset and GCS assets verified.

## 6. Assessment

| Criterion | Status |
|-----------|--------|
| Physics accuracy | PASS |
| Algorithm correctness | PASS |
| Algorithm domain-specificity | PASS -- 2/4 point-cloud-specific, 2 general but applicable |
| Literature coverage | PASS (through 2021; baselines still competitive) |
| Dataset completeness | PASS |
| Overall | **PASS** |

No code changes required. The variant override correctly replaces stereo
depth estimation methods with LiDAR point cloud processing algorithms.
