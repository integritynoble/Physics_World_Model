# Comprehensive 6-Point Check — Integral Imaging (Light Field Photography)

**URL:** https://pwm.platformai.org/benchmark/integral
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Integral Imaging (Integral Photography / Elemental Image Array)

**Physical principle:** Integral imaging is a glasses-free 3D display and capture technique that records a 2D array of micro-images (elemental images) through a lenslet array (microlens array, MLA) or pinhole array placed in front of a sensor. Each lenslet captures a slightly different perspective of the scene, encoding both spatial and angular information simultaneously. The inverse problem is to reconstruct a 3D scene or synthesize arbitrary viewpoints from the recorded elemental image array (EIA). Integral imaging is closely related to light field photography but historically predates it (Lippmann 1908) and is specifically used for autostereoscopic 3D displays and depth estimation.

**Forward model:**
```
EIA(u, v, i, j) = Σ_z I_3D(x, y, z) · h(u − x/z, v − y/z; i, j) + η

In matrix notation:
  y = A · x + η

where:
  EIA(u,v,i,j)  — elemental image at lenslet position (i,j), pixel (u,v) within that lenslet
  I_3D(x,y,z)   — 3D scene radiance at position (x,y,z)
  h(·)           — PSF of individual lenslet (u,v): sub-aperture view direction
  A              — forward sensing matrix (perspective projection through MLA)
  η              — sensor noise
  i,j ∈ {1…N_x, 1…N_y} — lenslet grid indices
```

**Inverse problem:** Reconstruct the 3D scene volume I_3D(x,y,z) or depth map D(x,y) from the 2D elemental image array EIA, or synthesize novel viewpoints from the captured 4D light field.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(incoherent scene illumination) → F(microlens array) → D(2D image sensor)

**Key mismatch parameters:**
- `lenslet_f_number`: f/# of individual lenslets; nominal f/2.8, perturbed f/5.6 (smaller aperture, less angular diversity)
- `lenslet_pitch`: spatial pitch of microlens array; nominal 1.0 mm, perturbed 0.5 mm (higher spatial-angular tradeoff)
- `depth_range`: scene depth range relative to focal plane; nominal 50–500 mm, perturbed 10–1000 mm (larger range, harder reconstruction)
- `sensor_noise_sigma`: Gaussian read noise; nominal σ=5 DN, perturbed σ=20 DN

**Dataset format:**
- `x_true: (H, W)` — ground-truth reconstructed 2D central view image (or 3D scene volume)
- `y: (H_total, W_total)` — flat 2D elemental image array (all lenslet sub-images tiled)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Computational reconstruction (back-projection) | Classical | Jang & Javidi, Opt. Lett. 26:1645 (2001) | Standard back-projection reconstruction through virtual lens array; analytic baseline |
| Depth-image-based rendering (DIBR) | Classical | Fehn, Proc. SPIE 5291:93 (2004) | Warp-based view synthesis using estimated depth for novel view generation |
| LFBM5D (light field denoising) | Classical | Alain & Smolic, IEEE ICIP 2017 | 5D collaborative filtering adapted for light field angular/spatial structure |
| EPINET (deep learning) | Deep Learning | Shin et al., CVPR 2018 | CNN exploiting epipolar plane image (EPI) structure for depth estimation |
| LFT (Light Field Transformer) | Transformer | Liang et al., ECCV 2022 | Transformer-based light field view synthesis with epipolar attention |

---

## 4. Literature & State of the Art (2024–2025)

1. **Wang et al. (2024)** "Efficient light field reconstruction via attention-guided feature aggregation," *IEEE Trans. Image Process.* — attention mechanism for adaptive angular feature aggregation in light field super-resolution.
2. **Jin et al. (2024)** "Diffusion-based light field synthesis from sparse angular views," *CVPR 2024* — score-based diffusion model for synthesizing dense 4D light fields from sparse captures.
3. **Shi et al. (2024)** "Stereo Meets Integral: Cross-Modal Self-Supervised Depth Learning from Integral Imaging," *IEEE Trans. Circuits Syst. Video Technol.* — self-supervised depth learning leveraging geometric constraints in integral image arrays.
4. **Liu et al. (2023)** "Disentangled Light Field Depth Estimation with Parallax-Aware Network," *Opt. Express* — disentangles texture and parallax cues for robust integral imaging depth estimation.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/integral_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/integral_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/integral_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/integral/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Integral imaging is correctly modeled as a perspective projection through a microlens array with 4D light field sampling, and the algorithm routing appropriately covers the canonical back-projection reconstruction, depth-image-based rendering, LFBM5D for denoising, and deep learning methods (EPINET for depth, LFT for view synthesis) that reflect current state of the art. The mismatch parameters — lenslet f/number, pitch, depth range, and sensor noise — capture the key optical tradeoffs in integral imaging system design. The benchmark is physically well-grounded for the elemental-image-array formulation of light field reconstruction.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 40.02 | 0.9990 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*
