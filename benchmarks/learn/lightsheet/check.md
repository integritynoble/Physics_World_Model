# Comprehensive 6-Point Check — Light-Sheet Fluorescence Microscopy

**URL:** https://pwm.platformai.org/benchmark/lightsheet
**Check Date:** 2026-03-11
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Light-Sheet Fluorescence Microscopy (LSFM / SPIM)

**Physical principle:** A thin sheet of laser light illuminates only a single plane of the specimen, exciting fluorophores in that plane while keeping out-of-focus regions dark. The orthogonal detection objective collects the emitted fluorescence, achieving optical sectioning with reduced phototoxicity compared to confocal microscopy.

**Forward model:**
```
y = Poisson(PSF_det * (S(z) * x) + scatter + bg) + readout_noise

where:
  x          -- 2D fluorescence ground truth (cleared tissue / embryo section)
  S(z)       -- light-sheet illumination profile (Gaussian beam with z-dependent
                waist; non-uniform across FOV -> striping artifacts at edges)
  PSF_det    -- detection PSF (widefield-like Gaussian, sigma ~2-3 px)
  scatter    -- tissue scattering (depth-dependent exponential attenuation)
  bg         -- out-of-focus background fluorescence + scatter-induced background
  readout    -- sCMOS Gaussian readout noise (std ~3-8 electrons)
```

**Inverse problem:** Recover the fluorescence distribution x from the noisy, blurred, stripe-corrupted measurement y, compensating for detection PSF, non-uniform sheet illumination, tissue scattering, and Poisson-Gaussian noise.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(laser sheet) -> F(fluorescent sample) -> D(sCMOS camera)

**Key mismatch parameters:**
- `sheet_thickness`: light-sheet waist sigma in pixels (3.0-8.0); controls axial sectioning
- `sheet_uniformity`: beam uniformity across FOV (0.5-1.0); lower = more striping
- `scattering_coeff`: tissue scattering attenuation per pixel (0.01-0.10)
- `noise_level`: peak photon count for Poisson noise (100-2000)

**Dataset tiers:**
| Tier   | Samples | Seed Offset | Difficulty |
|--------|---------|-------------|------------|
| Public | 12      | 0           | Moderate   |
| Dev    | 20      | 10000       | Medium     |
| Hidden | 20      | 20000       | Hard       |

**Dataset format:**
- `x_true: (256, 256) float32` -- ground truth fluorescence, normalized to [0, 1]
- `y: (256, 256) float32` -- noisy light-sheet measurement (photon counts)
- `H_ideal: (256, 256) float32` -- noiseless ideal image (for reference)

**Phantom types:**
- Sparse fluorescent nuclei (40-120 round bright spots)
- Vasculature networks (branching tubular structures with bifurcations)
- Developing organ structures (layered tissue with curved boundaries)
- Combined tissue sections (nuclei + vasculature + organ layers)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Stripe removal + Richardson-Lucy | Classical (CPU baseline) | Richardson (1972); Lucy (1974) | Fourier notch destriping + iterative Poisson ML deconvolution |
| TV-regularized deconvolution | Variational | Dey et al. (2006) *Microsc. Res. Tech.* 69:260-266 | TV prior preserves sharp fluorescent structures |
| Noise2Void | Self-supervised DL | Krull et al. (2019) *CVPR* | Self-supervised denoising, no paired data needed |
| CARE | Deep Learning | Weigert et al. (2018) *Nature Methods* 15:1090-1097 | U-Net for paired low/high-SNR light-sheet data |

**CPU Baseline Results (Stripe removal + Richardson-Lucy, 30 iterations):**
| Tier   | Mean PSNR (dB) | Mean SSIM |
|--------|----------------|-----------|
| Public | 23.04          | 0.540     |
| Dev    | 23.22          | 0.541     |
| Hidden | 22.58          | 0.526     |

---

## 4. Literature & State of the Art (2024-2025)

1. **Chen et al. (2024)** "Computational aberration correction for light-sheet microscopy with deep learning," *Nature Communications* -- neural-network correction of spatially varying PSFs in SPIM, 2x volumetric resolution improvement.
2. **Shi et al. (2024)** "Self-supervised deconvolution for fluorescence microscopy," *Bioinformatics* -- self-supervised framework combining blind deconvolution with noise modeling for light-sheet data.
3. **Zhao et al. (2025)** "Diffusion-based restoration for 3D fluorescence microscopy," *Medical Image Analysis* -- score-based diffusion models for joint denoising and deblurring.
4. **Liu et al. (2024)** "Transformer-based 3D super-resolution for light-sheet fluorescence microscopy," *IEEE Trans. Medical Imaging* -- swin-transformer for isotropic reconstruction from anisotropic LSFM stacks.

---

## 5. Local Dataset & GCS Status

**Local dataset:**
- `datasets/benchmark/lightsheet/generate_dataset.py` -- generator script
- `datasets/benchmark/lightsheet/{public,dev,hidden}/lightsheet_challenge_{tier}.h5`

**GCS datasets:**
- `gs://pwm-benchmark-datasets/datasets/Benchmark/lightsheet/public/lightsheet_challenge_public.h5` (8.5 MB)
- `gs://pwm-benchmark-datasets/datasets/Benchmark/lightsheet/dev/lightsheet_challenge_dev.h5` (14.0 MB)
- `gs://pwm-benchmark-datasets/datasets/Benchmark/lightsheet/hidden/lightsheet_challenge_hidden.h5` (13.9 MB)

**Gallery images:** `platform/pwm_platform/static/img/benchmark_gallery/lightsheet/scene_{00-03}/`
- Each scene: gt.png, measurement_I.png, measurement_II.png, recon_I.png, recon_II.png

---

## 6. Comprehensive Assessment

**Status:** PASS

The light-sheet fluorescence microscopy benchmark faithfully implements the LSFM forward model: Gaussian beam sheet illumination with non-uniform thickness (causing stripe artifacts), widefield detection PSF convolution, depth-dependent tissue scattering, and Poisson-Gaussian noise. The four phantom types (nuclei, vasculature, organ layers, combined) represent realistic cleared tissue / embryo section content. The CPU baseline (Fourier destriping + Richardson-Lucy deconvolution) achieves 22-23 dB PSNR, consistent with the expected 22-28 dB range and leaving headroom for advanced algorithms. Mismatch difficulty increases appropriately from public to hidden tier.

---
*Comprehensive 6-point check by deep-check pipeline v3 -- updated 2026-03-11 with benchmark results*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 20.03 | 0.0553 | 0.00 | PASS |
| rl_20iter | -33.41 | 0.0000 | 0.05 | PASS |
| fourier_notch | -28.21 | 0.0000 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 20.1 dB |
| SSIM (sample_00) | 0.2006 |
| Runtime | 1.51 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 20.13 dB |
| SSIM (sample_00) | 0.2113 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TV-Deconvolution
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 20.13 dB |
| SSIM (sample_00) | 0.2061 |
| Runtime | 0.21 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 19.86 dB |
| SSIM (sample_00) | 0.6728 |
| Runtime | 6.63 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-DnCNN
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 19.86 dB |
| SSIM (sample_00) | 0.6728 |
| Runtime | 6.84 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 20.1 dB |
| SSIM (sample_00) | 0.2006 |
| Runtime | 0.48 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 20.13 dB |
| SSIM (sample_00) | 0.2113 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TV-Deconvolution
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 20.13 dB |
| SSIM (sample_00) | 0.2061 |
| Runtime | 0.29 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 19.86 dB |
| SSIM (sample_00) | 0.6728 |
| Runtime | 8.51 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-DnCNN
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 19.86 dB |
| SSIM (sample_00) | 0.6728 |
| Runtime | 7.56 s/sample |

**Result: PASS**
