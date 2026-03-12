# Comprehensive 6-Point Check — MALDI Mass Spectrometry Imaging (MALDI-MSI)

**URL:** https://pwm.platformai.org/benchmark/maldi_msi
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Matrix-Assisted Laser Desorption/Ionization Mass Spectrometry Imaging (MALDI-MSI)

**Physical principle:** MALDI-MSI produces spatially resolved molecular composition maps by rastering a pulsed UV laser across a tissue section coated with an organic matrix compound. The laser desorbs and ionizes analyte molecules co-crystallized with the matrix. The ions are accelerated into a time-of-flight (TOF) mass spectrometer, and the m/z spectrum at each spatial position is recorded. The ion intensity I(m/z, x, y) for a given molecule is approximately proportional to its local concentration c(x,y) modulated by matrix crystallization quality, laser fluence, and ion extraction efficiency.

**Forward model:**
```
I(m/z, x, y) = Y(m/z) * c(x,y) * primary_dose(x,y) + noise
```
where Y(m/z) is the ionization yield and transmission efficiency for molecular ion at m/z, c(x,y) is the analyte concentration, primary_dose(x,y) is the local laser fluence, and noise is detector shot noise plus chemical background. The benchmark uses the `microscopy_psf` engine modeling the spatial resolution limitation from the laser spot size and matrix co-crystal grain size:
```
I_spatial(m/z) = PSF_laser ⊛ c(x,y) * Y(m/z) + noise
```

**Inverse problem:** Recover molecular concentration maps c_k(x,y) for each m/z channel from the noisy, resolution-limited, laser-fluence-variable MALDI-MSI datacube. Challenges include mass accuracy drift, laser fluence heterogeneity, matrix crystallization variation, and extraction timing jitter.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(MALDI-TOF) → Sigma(laser_fluence, mass_accuracy, extraction_delay, matrix_crystal) → D(I_maldi, eta)

**Key mismatch parameters:**
- **Laser fluence drift** (0.8–1.2×): shot-to-shot laser energy variation changes ionization efficiency and requires normalization
- **Mass accuracy** (-5 to +5 ppm): m/z calibration drift causes peak misassignment when matching to molecular databases
- **Extraction delay** (80–120 ns): timing jitter in the ion extraction pulse changes TOF separation and mass resolution
- **Matrix crystallization** (0.7–1.3×): heterogeneous matrix crystal formation produces spatially variable ionization efficiency

**Dataset format:**
- `x_true: (H, W, K)` — ground-truth molecular concentration maps for K m/z channels
- `y: (H, W, M)` — MALDI-MSI datacube (counts vs. m/z at M channels per spatial pixel)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| SG-ALS | Classical | Savitzky-Golay + ALS baseline | Appropriate — standard spectral baseline correction for MALDI spectra (MALDI baseline is complex) |
| SVD | Classical | Singular Value Decomposition | Appropriate — PCA/NMF-based molecular image factorization for dimensionality reduction |
| PnP-DnCNN | PnP | Zhang et al., 2017 | Appropriate — denoiser prior for MALDI spectral image denoising |
| CDAE | Deep Learning | Zhang et al., Sensors 2024 | Appropriate — autoencoder architecture adaptable to MALDI m/z image denoising |
| SpectraFormer | Vision Transformer | Spectroscopy transformer, 2024 | Appropriate — cross-channel attention for joint spatial-spectral MALDI reconstruction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Alexandrov et al. (2024)** "Deep learning for MALDI-MSI data analysis: denoising and spatial reconstruction," *Nat. Methods* — convolutional network achieving 4× improvement in spatial resolution through deconvolution.
2. **Rappez et al. (2024)** "SpaceM-AI: machine learning pipeline for MALDI-MSI metabolomics," *Metabolites* — integrated pipeline for m/z peak picking, normalization, and spatial clustering.
3. **Inglese et al. (2024)** "Self-supervised denoising of MALDI mass spectrometry images," *Anal. Chem.* — Noise2Noise approach for MALDI without requiring replicate measurements.
4. **Tuck et al. (2025)** "Transformer-based MALDI-MSI reconstruction with learned m/z priors," *J. Proteome Res.* — multi-head attention across m/z channels for simultaneous denoising and peak resolution.

---

## 5. Local Dataset & GCS Status

- **GCS public tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/maldi_msi_challenge_public.h5`
- **GCS dev tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/maldi_msi_challenge_dev.h5`
- **GCS hidden tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/maldi_msi_challenge_hidden.h5` (blocked from download)
- **Gallery images:** `gs://pwm-benchmark-datasets/img/benchmark_gallery/maldi_msi/scene_*/`
- **No local copies** — all data served from GCS via `/gcs/` proxy

---

## 6. Comprehensive Assessment

**Physics correctness:** MALDI-MSI is correctly classified as nonlinear (matrix crystallization and ionization suppression effects make intensity-concentration nonlinear). The four mismatch parameters capture the dominant MALDI systematic errors: fluence, mass calibration, extraction timing, and matrix quality.

**Algorithm appropriateness:** The 11-algorithm spectroscopy pool is well-matched to MALDI-MSI, which shares the hyperspectral reconstruction challenge with FTIR, LIBS, and Raman imaging. The SVD/NMF approach is especially appropriate for MALDI where molecular factorization is standard.

**Benchmark structure:** Mass accuracy mismatch (±5 ppm) is particularly important for MALDI — algorithms that perform m/z peak matching against molecular databases will fail on hidden tier if they don't account for mass calibration drift.

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
| precomputed_baseline | 26.30 | 0.9418 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** Deconv
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 15.64 dB |
| SSIM (sample_00) | 0.4712 |
| Runtime | 0.53 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Calibration-Lookup
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 22.02 dB |
| SSIM (sample_00) | 0.6489 |
| Runtime | 0.51 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Peak Fitting
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 22.02 dB |
| SSIM (sample_00) | 0.6489 |
| Runtime | 0.44 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-BM3D
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 22.64 dB |
| SSIM (sample_00) | 0.6865 |
| Runtime | 0.69 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-NLM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 22.49 dB |
| SSIM (sample_00) | 0.6674 |
| Runtime | 0.51 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Deconv
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 15.64 dB |
| SSIM (sample_00) | 0.4712 |
| Runtime | 0.41 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Calibration-Lookup
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 22.02 dB |
| SSIM (sample_00) | 0.6489 |
| Runtime | 0.48 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Peak Fitting
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 22.02 dB |
| SSIM (sample_00) | 0.6489 |
| Runtime | 0.48 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-BM3D
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 22.64 dB |
| SSIM (sample_00) | 0.6865 |
| Runtime | 0.57 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-NLM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 22.49 dB |
| SSIM (sample_00) | 0.6674 |
| Runtime | 0.49 s/sample |

**Result: PASS**
