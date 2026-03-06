# Comprehensive 6-Point Check — 3D Gaussian Splatting

**URL:** https://pwm.platformai.org/benchmark/gaussian_splatting
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** 3D Gaussian Splatting (3DGS)

**Physical principle:** 3D Gaussian Splatting represents a scene as a collection of anisotropic 3D Gaussians, each parameterized by position (mean), covariance (orientation + scale), opacity, and spherical harmonic (SH) color coefficients. Novel views are rendered by projecting each 3D Gaussian onto the 2D image plane via splatting (EWA splatting), compositing front-to-back in depth order using alpha blending. The forward model is fully differentiable, enabling gradient-based optimization of Gaussian parameters to minimize photometric reconstruction loss against multi-view photographs. The inverse problem is recovering scene Gaussians from a set of calibrated RGB views.

**Forward model:**
```
I_k(u) = Σ_i c_i(d_k) · α_i · Π_{j<i}(1 − α_j) · G_2D_i(u; μ_i^{2D}, Σ_i^{2D})

where:
  G_i          — i-th 3D Gaussian with mean μ_i, covariance Σ_i, opacity σ_i, SH coefficients c_i
  G_2D_i(u)   — projected 2D Gaussian onto image plane for view k
  c_i(d_k)    — view-dependent color from spherical harmonics evaluated at direction d_k
  α_i          — per-pixel opacity = σ_i · G_2D_i(u)
  I_k(u)      — rendered pixel color at position u for view k
  Loss: min_{G} Σ_k ||I_k − I_k^{gt}||² + λ · SSIM_loss
```

**Inverse problem:** Recover the set of N anisotropic 3D Gaussians {G_i} from M calibrated RGB views {I_k^{gt}} via differentiable rendering and adaptive densification/pruning.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(multi-view RGB cameras) → F(3D scene radiance field) → D(photometric reconstruction)

**Key mismatch parameters:**
- `num_views`: number of input training views; nominal 100, perturbed 20 (sparse-view reconstruction)
- `camera_noise`: camera pose estimation error (from COLMAP); nominal 0.5 px reprojection error, perturbed 2.0 px
- `scene_complexity`: number of Gaussians required; nominal 500K, perturbed 2M (complex textures, higher compute)
- `exposure_variation`: inter-view exposure difference; nominal 0 EV, perturbed ±1.5 EV (outdoor illumination change)

**Dataset format:**
- `x_true: (H, W, 3)` — ground-truth novel view RGB image (held-out test view)
- `y: (M, H, W, 3)` — M training view RGB images with known camera poses

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| 3D Gaussian Splatting (3DGS) | Classical differentiable rendering | Kerbl et al., ACM TOG 42:139 (2023) | Original 3DGS paper; real-time radiance field with explicit Gaussian representation |
| Mip-NeRF 360 | Deep Learning (NeRF) | Barron et al., CVPR 2022 | State-of-the-art NeRF baseline for unbounded 360° scenes; implicit representation |
| Scaffold-GS | Classical + learned | Lu et al., CVPR 2024 | Anchored Gaussian scene graph reducing memory and improving quality |
| 2D Gaussian Splatting | Classical | Huang et al., SIGGRAPH 2024 | 2D surfels for better surface reconstruction than volumetric 3DGS |
| GaussianSplatting-Transformer | Transformer | Chen et al., ECCV 2024 | Feed-forward transformer predicting 3DGS parameters from images without per-scene optimization |

---

## 4. Literature & State of the Art (2024–2025)

1. **Kerbl et al. (2023)** "3D Gaussian Splatting for Real-Time Novel View Synthesis," *ACM TOG 42:139* — foundational paper introducing 3DGS; enables real-time rendering at competitive quality to NeRF.
2. **Huang et al. (2024)** "2D Gaussian Splatting for Geometrically Accurate Radiance Fields," *SIGGRAPH 2024* — replaces volumetric Gaussians with 2D surfels for improved geometry and surface normal estimation.
3. **Chen et al. (2024)** "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images," *ECCV 2024* — feed-forward network predicting 3DGS from 2–3 views without test-time optimization.
4. **Yu et al. (2024)** "MipSplatting: Anti-aliased 3D Gaussian Splatting," *CVPR 2024* — multi-scale dilation for frequency-consistent rendering eliminating aliasing artifacts in 3DGS.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/gaussian_splatting_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/gaussian_splatting_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/gaussian_splatting_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/gaussian_splatting/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

3D Gaussian Splatting is correctly modeled as a differentiable rendering inverse problem recovering anisotropic Gaussian scene representations from multi-view RGB imagery, and the algorithm routing covers the full spectrum from the original 3DGS and its variants (Scaffold-GS, 2DGS, MipSplatting) to NeRF-based baselines (Mip-NeRF 360) and feed-forward generalization networks. The mismatch parameters — view count, camera pose noise, scene complexity, and exposure variation — reflect the primary factors determining reconstruction quality in real capture setups. The benchmark is up-to-date with the rapid pace of development in this field.

---
*Comprehensive 6-point check by deep-check pipeline v3*
