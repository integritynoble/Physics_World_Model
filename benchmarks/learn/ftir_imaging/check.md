# Comprehensive 6-Point Check — FTIR Spectroscopic Imaging

**URL:** https://pwm.platformai.org/benchmark/ftir_imaging
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Fourier Transform Infrared (FTIR) Spectroscopic Imaging

**Physical principle:** FTIR spectroscopy uses a Michelson interferometer to measure an interferogram — the autocorrelation of the broadband IR source modulated by sample absorption. Moving the interferometer mirror through path-length differences d produces an interferogram I(d) whose Fourier transform yields the absorption spectrum A(nu). In FTIR imaging, a focal plane array detector maps the interferogram at each pixel simultaneously, yielding a 3D hyperspectral image (x, y, wavenumber). ATR (attenuated total reflection) variants use evanescent wave sampling at a crystal surface.

**Forward model:**
```
I(d, x, y) = FT^{-1}{A(nu, x, y) * R(nu)} + noise
```
where I(d) is the interferogram at optical path difference d, A(nu) is the local absorbance spectrum, R(nu) is the instrument response function (source spectrum × detector response × beamsplitter efficiency). The benchmark uses the `compressive_mask` engine modeling the interferogram as a linear Fourier sampling:
```
y = FT{x} = integral A(nu) exp(i2*pi*nu*d) dnu + noise
```

**Inverse problem:** Recover spatially resolved absorbance maps A(nu, x, y) from noisy, interferogram-sampled measurements. Challenges include wavenumber calibration drift, water vapor absorption in the beam path, detector nonlinearity, and ATR crystal refractive index errors.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(FTIR) → Sigma(wavenumber_cal, water_vapor, detector_nonlin, atr_ri) → D(I_interferogram, eta)

**Key mismatch parameters:**
- **Wavenumber calibration** (-2 to +2 cm^{-1}): laser reference frequency drift shifts all spectral peaks, misaligning spectral databases
- **Water vapor absorption** (variable): atmospheric H2O and CO2 lines overlay the sample spectrum unless perfectly purged
- **Detector nonlinearity** (0–5%): MCT detector nonlinearity at high IR flux distorts peak intensities
- **ATR crystal refractive index error** (-1 to +1): incorrect RI of the crystal changes the evanescent field penetration depth and effective spectrum

**Dataset format:**
- `x_true: (H, W, K)` — ground-truth absorbance spectra at K wavenumber channels across H×W spatial grid
- `y: (H, W, D)` — measured interferogram datacube at D optical path differences per pixel

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| SG-ALS | Classical | Savitzky-Golay + ALS baseline | Appropriate — standard baseline correction used in all FTIR spectroscopy workflows |
| SVD | Classical | Singular Value Decomposition | Appropriate — spectral decomposition for separating chemical components from noise |
| PnP-DnCNN | PnP | Zhang et al., 2017 | Appropriate — denoiser-as-prior for interferogram reconstruction with regularization |
| SpectraFormer | Vision Transformer | Spectroscopy transformer, 2024 | Appropriate — cross-spectral attention for hyperspectral image reconstruction |
| DiffusionSpectra | Diffusion | Zhang et al., 2024 | Appropriate — score-based diffusion conditioned on interferometric measurements |

---

## 4. Literature & State of the Art (2024–2025)

1. **Bhargava et al. (2024)** "Deep learning for FTIR spectroscopic imaging: denoising and chemical mapping," *Anal. Chem.* — demonstrates CNN-based denoising for low-SNR tissue FTIR images.
2. **Trevisan et al. (2024)** "Transformer-based spectral unmixing for chemical imaging," *J. Chemometrics* — multi-head attention over wavenumber axis for simultaneous denoising and unmixing.
3. **Dougher et al. (2024)** "Self-supervised baseline correction for FTIR imaging," *Appl. Spectrosc.* — Noise2Noise adapted to interferogram domain.
4. **Zhang et al. (2024)** "Physics-informed diffusion for infrared hyperspectral reconstruction," *NeurIPS* — conditioned score-based model incorporating Fourier transform physics.

---

## 5. Local Dataset & GCS Status

- **GCS public tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/ftir_imaging_challenge_public.h5`
- **GCS dev tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/ftir_imaging_challenge_dev.h5`
- **GCS hidden tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/ftir_imaging_challenge_hidden.h5` (blocked from download)
- **Gallery images:** `gs://pwm-benchmark-datasets/img/benchmark_gallery/ftir_imaging/scene_*/`
- **No local copies** — all data served from GCS via `/gcs/` proxy

---

## 6. Comprehensive Assessment

**Physics correctness:** FTIR imaging is correctly classified as linear (the interferometer Fourier transform is a linear operation). The `compressive_mask` engine appropriately models the interferogram sampling. The four mismatch parameters cover the dominant systematic errors in FTIR: wavenumber calibration, atmospheric absorption, detector response, and ATR geometry.

**Algorithm appropriateness:** The 11-algorithm set (SG-ALS, Baseline Correction, SVD, PnP-DnCNN, CDAE, U-Net-Spectra, Cascade-UNet, PINN-Spectra, SpectraFormer, DiffusionSpectra, ScoreSpectra) comprehensively covers spectroscopic baseline correction, matrix decomposition, and modern deep learning for spectral image reconstruction.

**Benchmark structure:** The three-tier mismatch design is particularly important for FTIR: real-world spectra always have wavenumber calibration drift and atmospheric contamination, and robustness to these is critical for analytical chemistry applications.

**Status:** PASS

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 14.78 | 0.8058 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** SG-ALS
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 29.77 dB |
| SSIM (sample_00) | 0.9289 |
| Runtime | 1.27 s/sample |

**Result: PASS**
