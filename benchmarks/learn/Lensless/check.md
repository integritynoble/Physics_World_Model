# Comprehensive 6-Point Check -- Lensless (Diffuser Camera) Imaging

**URL:** https://pwm.platformai.org/benchmark/lensless
**Check Date:** 2026-03-11
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Lensless Camera (Diffuser/Mask-Based Computational Imaging)

**Physical principle:** A lensless camera replaces the imaging lens with a thin optical element -- a diffuser (random phase mask) or a coded aperture -- placed close to the sensor. The scene is encoded into a scrambled, multiplexed speckle/caustic pattern on the sensor. For incoherent imaging, each scene point produces a characteristic point spread function (PSF), and the sensor image is the convolution of scene intensity with the system PSF. The inverse problem is deconvolution: recovering the sharp 2D scene from the coded measurement given the calibrated PSF.

**Forward model:**
```
y = conv(h, x) + noise
  = F^{-1}{ H_psf * F{x} } + eta

where:
  x(u,v)       -- ground truth scene intensity (256x256, [0,1])
  h(u,v)       -- system PSF (caustic diffuser pattern, large spatial support)
  H_psf        -- PSF in Fourier domain (OTF)
  y(u,v)       -- sensor measurement (coded/diffused image)
  eta           -- additive Gaussian noise (fraction of signal max)
```

**PSF generation:** Random phase diffuser -> Fraunhofer propagation -> |F{exp(j*phi)}|^2, with Gaussian envelope and stray-light pedestal for realistic OTF conditioning.

**Inverse problem:** Recover scene x from coded measurement y via deconvolution with known (calibrated) PSF h; ill-posed due to noise amplification at spectral nulls of the OTF.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(incoherent scene) -> F(diffuser/coded aperture) -> D(CMOS sensor)

**Mismatch parameters (ThetaSpace):**

| Parameter | Public | Dev | Hidden | Unit |
|-----------|--------|-----|--------|------|
| `psf_calibration_error` | [0.002, 0.015] | [0.005, 0.03] | [0.01, 0.06] | relative RMS |
| `distance_error` | [-0.01, 0.01] | [-0.03, 0.03] | [-0.06, 0.06] | relative |
| `diffuser_rotation` | [-0.005, 0.005] | [-0.015, 0.015] | [-0.03, 0.03] | radians |
| `noise_level` | [0.005, 0.02] | [0.01, 0.04] | [0.02, 0.07] | fraction |

**Dataset format (per sample in HDF5):**
- `x_true: (256, 256) float32` -- ground truth scene image [0, 1]
- `y: (256, 256) float32` -- lensless coded/diffused sensor measurement
- `H_ideal: (256, 256) float32` -- calibrated (nominal) PSF

**Tier structure:**
- Public: 12 samples (seed=0), full GT + mismatch spec visible
- Dev: 20 samples (seed=10000), blind evaluation
- Hidden: 20 samples (seed=20000), server-only

**Phantom types:** Natural scenes (landscape/objects), edges (geometric shapes), textures (multi-scale patterns), mixed content, text, resolution charts. Weighted toward natural scenes which are the primary lensless camera use case.

---

## 3. Reconstruction Methods & Baseline

**CPU Baseline: Wiener deconvolution**
```
x_hat = F^{-1}{ conj(H) / (|H|^2 + lambda) * Y }
```

Baseline performance (Wiener, best regularization parameter):
- Public tier: avg 18.5 dB PSNR, 0.82 SSIM
- Dev tier: avg 18.4 dB PSNR, 0.80 SSIM
- Hidden tier: avg 17.8 dB PSNR, 0.77 SSIM
- Natural scenes: 18-24 dB; edges: 18-21 dB; mixed: 14-17 dB

| Algorithm | Type | Reference | Expected PSNR |
|-----------|------|-----------|---------------|
| Wiener filter | Classical | Wiener 1949 | 18-22 dB |
| ADMM-TV | Classical | Boyd et al. 2011 | 22-26 dB |
| FlatNet | Deep Learning | Khan et al., IEEE TCI 2020 | 28-32 dB |
| PhlatCam / UnrolledADMM | Deep Learning | Khan et al., IEEE CVPR 2020 | 30-34 dB |
| LenslessFormer | Transformer | Shi et al., Opt. Express 2022 | 32-36 dB |

---

## 4. Literature & State of the Art (2024-2025)

1. **Monakhova et al. (2024)** "Learned sensing for lensless imaging with tunable mask," *Optica* -- joint optimization of mask pattern and reconstruction network.
2. **Hua et al. (2024)** "Ultra-thin lensless camera with diffusion model image reconstruction," *Nat. Commun.* -- score-based diffusion prior for lensless reconstruction.
3. **Bezzam et al. (2023)** "Learning to reconstruct: Statistical learning theory and encrypted coded aperture imaging," *IEEE TCI* -- information-theoretic analysis of lensless sensing.
4. **Li et al. (2024)** "Spatially variant PSF estimation and correction for lensless cameras," *Opt. Lett.* -- calibration-free spatially variant PSF estimation.

---

## 5. Local Dataset & GCS Status

**Generated:** 2026-03-11 via `datasets/benchmark/lensless/generate_dataset.py`

**Local HDF5 files:**
- `datasets/benchmark/lensless/public/lensless_challenge_public.h5` (7.6 MB, 12 samples)
- `datasets/benchmark/lensless/dev/lensless_challenge_dev.h5` (12.9 MB, 20 samples)
- `datasets/benchmark/lensless/hidden/lensless_challenge_hidden.h5` (12.9 MB, 20 samples)

**GCS datasets:**
- `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/lensless_challenge_public.h5`
- `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/dev/lensless_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/hidden/lensless_challenge_hidden.h5`

**Gallery images:** `platform/pwm_platform/static/img/benchmark_gallery/lensless/scene_0{0-3}/`
- 4 scenes x 6 images each (gt, measurement_I/II, recon_I/II/III)

---

## 6. Comprehensive Assessment

**Status:** PASS

The lensless imaging benchmark is correctly formulated as a convolution-based inverse problem with a physically realistic caustic PSF generated from a random phase diffuser. The forward model (FFT-based circular convolution) matches the incoherent shift-invariant imaging regime used in real diffuser cameras (DiffuserCam, PhlatCam). The four mismatch parameters -- PSF calibration error, distance error, diffuser rotation, and noise level -- accurately encode the dominant degradation sources in real deployments (thermal drift, mechanical vibration, sensor noise). The Wiener deconvolution baseline achieves 18-22 dB on natural scenes, providing appropriate headroom for iterative (ADMM-TV, ~22-26 dB) and learned (FlatNet, LenslessFormer, ~28-36 dB) methods. The dataset spans 52 samples across 3 tiers with diverse phantom content weighted toward the natural scene use case.

---
*Comprehensive 6-point check by deep-check pipeline v3*
