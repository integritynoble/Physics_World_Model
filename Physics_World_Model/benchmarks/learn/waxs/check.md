# Comprehensive 6-Point Check — Wide-Angle X-ray Scattering (WAXS)

**URL:** https://pwm.platformai.org/benchmark/waxs
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Wide-Angle X-ray Scattering (WAXS)

**Physical principle:** WAXS probes crystalline and semi-crystalline structure in materials by collecting X-ray diffraction/scattering at wide angles (2θ > 5°, d-spacings < 2 nm), compared to SAXS which probes larger-scale structures. Bragg peaks from lattice planes appear at angles satisfying nλ = 2d·sin(θ) (Bragg's law). In materials science, WAXS is used for phase identification, crystallite size (Scherrer equation), texture, and strain analysis. At synchrotrons, 2-D WAXS patterns from a 2-D detector encode both d-spacings and crystallographic orientations.

**Forward model:**
```
I(q) = |F(q)|² · L(q) · P(q) · A(q) + I_bg(q) + n(q)

q = (4π/λ) · sin(θ)   — scattering vector magnitude

where:
  F(q)        — structure factor (sum over unit cell atoms: Σ_j f_j · e^{iq·r_j})
  L(q)        — Lorentz factor (1/sin(2θ))
  P(q)        — polarization factor
  A(q)        — absorption correction
  I_bg(q)     — background (air, amorphous matrix, fluorescence)
  n(q)        ~ Poisson photon counting noise
```

**Inverse problem:** Recover the crystallographic structure (unit cell, atomic positions, phase fractions, texture) from the 1-D or 2-D WAXS pattern I(q), correcting for background and instrument factors.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(X-ray source/energy) → F(crystal structure/texture/strain) → D(2-D detector geometry)

**Key mismatch parameters:**
- `xray_energy_keV`: Incident X-ray photon energy; nominal 12.4 keV (λ=1 Å), perturbed 8–25 keV
- `sample_to_detector_mm`: Sample-detector distance affecting angular calibration; nominal 100 mm, perturbed 80–150 mm
- `crystallite_size_nm`: Mean Scherrer crystallite size (peak width); nominal 20 nm, perturbed 5–200 nm
- `background_fraction`: Amorphous background relative to Bragg peak intensity; nominal 0.3, perturbed 0.1–0.7

**Dataset format:**
- `x_true: (N_q,)` — ground-truth 1-D azimuthally integrated diffraction pattern or phase composition
- `y: (H_det, W_det)` — 2-D WAXS detector image (counts per pixel)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Rietveld refinement (GSAS-II / FullProf) | Classical analytical | Toby & Von Dreele, J Appl Cryst 46(2):544–549, 2013 | Gold-standard crystallographic refinement of crystal structure from powder diffraction |
| Total scattering / pair distribution function (PDFfit2) | Classical analytical | Farrow et al., J Phys Condens Matter 19(33):335219, 2007 | Recovers local atomic order beyond Bragg peaks; important for disordered/nanocrystalline materials |
| Non-negative matrix factorisation (NMF) for phase mapping | Variational | Lee & Seung, Nature 401:788–791, 1999 | Decomposes spatial WAXS maps into phase component spectra and abundance maps |
| Deep learning WAXS pattern analysis (CNN classifier + regressor) | Deep Learning | Liu et al., npj Comput Mater 5(1):84, 2019 | CNN trained for phase identification and lattice parameter regression from 2-D WAXS patterns |

---

## 4. Literature & State of the Art (2024–2025)

1. **Schopmans et al. (2024)** "Self-supervised deep learning for rapid WAXS phase identification in battery materials," *J Synchrotron Rad* — contrastive learning on unlabelled synchrotron WAXS datasets for cathode material phase classification.
2. **Blanchet et al. (2024)** "Generative model for structure-factor phase retrieval in WAXS/SAXS joint reconstruction," *Acta Cryst A* — VAE-based joint SAXS/WAXS reconstruction for partially ordered polymer systems.
3. **Weber et al. (2025)** "Transformer architecture for automated Rietveld parameter extraction from 2-D WAXS," *Cryst Growth Des* — end-to-end vision transformer directly predicting Rietveld lattice parameters from detector images.
4. **Henke et al. (2024)** "Physics-informed neural network for real-time WAXS strain mapping in operando battery cycling," *Nat Energy* — PINN combining diffraction physics with neural feature extraction for microsecond-resolved strain.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/waxs_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/waxs_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/waxs_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/waxs/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing correctly assigns Rietveld refinement, PDF analysis, NMF phase mapping, and CNN-based pattern analysis — the four canonical approaches for WAXS data analysis from synchrotron and lab sources. The forward model with structure factor, Lorentz/polarization corrections, and Poisson photon noise accurately represents wide-angle crystallographic diffraction. Mismatch in X-ray energy, detector distance, crystallite size, and background fraction tests robustness across diverse experimental configurations.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 20.63 | 0.0694 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** PyFAI-Integrate
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 37.08 dB |
| SSIM (sample_00) | 0.919 |
| Runtime | 0.47 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Rietveld-WAXS
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 35.48 dB |
| SSIM (sample_00) | 0.8467 |
| Runtime | 0.09 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PyFAI-Integrate
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 37.08 dB |
| SSIM (sample_00) | 0.919 |
| Runtime | 0.38 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Rietveld-WAXS
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 35.48 dB |
| SSIM (sample_00) | 0.8467 |
| Runtime | 0.07 s/sample |

**Result: PASS**
