# Comprehensive 6-Point Check — Photometric Stereo

**URL:** https://pwm.platformai.org/benchmark/photometric_stereo
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Photometric Stereo

**Physical principle:** Photometric stereo recovers the 3D surface normal field of an object by photographing it under multiple known illumination directions while keeping the camera fixed. Under the Lambertian reflectance model, each pixel's intensity is proportional to the dot product of the surface normal and the incident light direction, scaled by the surface albedo. By collecting images under at least three non-coplanar illumination directions, the system of linear equations can be solved per-pixel to recover both the surface normal and the albedo, enabling subsequent surface height integration.

**Forward model:**
```
I_k(x, y) = ρ(x, y) · [n(x, y) · l_k] + n_noise

where:
  I_k(x, y)  — image intensity at pixel (x,y) under illumination k
  ρ(x, y)    — surface albedo (diffuse reflectance coefficient)
  n(x, y)    — unit surface normal vector at (x,y)
  l_k        — unit vector toward light source k (known)
  n_noise    — additive Gaussian sensor noise

Matrix form: I = L · N_ρ, where L is K×3 light matrix, N_ρ encodes ρ·n per column
```

**Inverse problem:** Given K≥3 intensity images under known illumination directions L, recover the surface normal field n(x,y) and albedo ρ(x,y) per pixel; surface height z(x,y) is then obtained by integrating the normals using Poisson-equation-based integration.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(directional LED/lamp, K directions) → F(Lambertian/non-Lambertian reflectance) → D(camera, fixed viewpoint)

**Key mismatch parameters:**
- `light_direction_error`: calibration error in l_k; nominal 0°, perturbed ±3° angular deviation
- `non_lambertian_component`: specular lobe magnitude; nominal 0 (pure Lambertian), perturbed to ρ_s=0.3 Blinn-Phong
- `inter_reflections`: global illumination effect; nominal absent, perturbed with concave surface bounces
- `cast_shadows`: hard shadow fraction; nominal 0%, perturbed to 15% of pixels shadowed per image

**Dataset format:**
- `x_true: (H, W, 3)` — per-pixel surface normal field as unit vectors (nx, ny, nz); or optionally (H, W) height map z
- `y: (K, H, W)` — K intensity images under K different illumination directions

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Classic Least-Squares PS | Classical | Woodham, Optical Engineering 19, 139–144 (1980) | Foundational Lambertian PS via per-pixel pseudo-inverse; analytic and interpretable |
| Robust PCA PS (RPCA) | Classical | Wu et al., IEEE CVPR, pp. 1482–1489 (2010) | Handles outliers (shadows/specularities) via sparse+low-rank decomposition |
| PS-FCN | Deep Learning | Chen et al., ECCV, pp. 3–19 (2018) | Fully convolutional network; processes arbitrary number of input images |
| UniPS (Universal Photometric Stereo) | Deep Learning | Ikehata, NeurIPS 35 (2022) | Transformer-based PS that handles uncalibrated, non-Lambertian surfaces |
| SDM-UniPS | Diffusion | Ikehata et al., CVPR (2023) | Score distillation matching for single-image and multi-image PS via diffusion priors |

---

## 4. Literature & State of the Art (2024–2025)

1. **Ikehata (2024)** "Scalable, Detailed and Mask-Free Universal Photometric Stereo," *CVPR 2024* — ViT-based PS model trained on 1M+ synthetic images; achieves state-of-the-art on DiLiGenT benchmark.
2. **Liu et al. (2024)** "Diffusion-based photometric stereo with geometry-aware priors," *ECCV 2024* — diffusion model conditioned on depth cues recovers normals under extreme non-Lambertian effects.
3. **Quéau et al. (2024)** "Coupled shape-from-shading and photometric stereo for specular object recovery," *International Journal of Computer Vision* — joint optimization of geometry and BRDF under photometric stereo constraints.
4. **He et al. (2025)** "Foundation model for photometric stereo using internet-scale image pretraining," *arXiv* — large-scale vision model fine-tuned with PS losses; zero-shot generalization to new materials.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/photometric_stereo_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/photometric_stereo_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/photometric_stereo_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/photometric_stereo/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Photometric stereo is a well-established shape-from-shading inverse problem with a clean linear forward model under Lambertian assumption. Algorithm routing correctly includes classic Woodham least-squares, RPCA-based robust methods, and modern deep learning approaches (PS-FCN, UniPS). The four mismatch parameters (light direction error, non-Lambertian reflectance, inter-reflections, cast shadows) faithfully represent the primary sources of model-data mismatch in real photometric stereo experiments.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 29.01 | 0.9583 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** LS Normal Est.
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 32.67 dB |
| SSIM (sample_00) | 0.96 |
| Runtime | 1.05 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Robust PCA
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 32.67 dB |
| SSIM (sample_00) | 0.96 |
| Runtime | 1.03 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** LS Normal Est.
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 32.67 dB |
| SSIM (sample_00) | 0.96 |
| Runtime | 0.51 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Robust PCA
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 32.67 dB |
| SSIM (sample_00) | 0.96 |
| Runtime | 0.58 s/sample |

**Result: PASS**
