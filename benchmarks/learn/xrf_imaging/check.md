# Comprehensive 6-Point Check — X-ray Fluorescence (XRF) Imaging

**URL:** https://pwm.platformai.org/benchmark/xrf_imaging
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** X-ray Fluorescence (XRF) Imaging / Elemental Mapping

**Physical principle:** XRF imaging irradiates a sample with a focused X-ray beam (synchrotron or laboratory micro-XRF), causing inner-shell photoionization of atoms. Electrons fill the vacancies emitting characteristic fluorescence X-rays at element-specific energies (e.g., Fe-Kα at 6.40 keV, Cu-Kα at 8.04 keV). Raster scanning across the sample while recording the full fluorescence spectrum per pixel produces spatially resolved elemental maps. Applications span cultural heritage conservation (paintings, manuscripts), geochemistry, environmental science, and materials characterisation.

**Forward model:**
```
I_E(x,y) = I_0 · μ_photoion(E_0, Z) · ρ_Z(x,y) · ω_Z · f_geom · exp(-μ_self·t) + B_E + n

XRF spectrum at pixel (x,y):
  S(E; x,y) = Σ_Z  I_E_Z(x,y) · G(E - E_Z; σ_det) + scatter_continuum + n

where:
  ρ_Z(x,y)   — areal density of element Z (µg/cm²)
  μ_photoion  — photoionization cross-section at primary energy E_0
  ω_Z         — fluorescence yield for element Z
  f_geom      — solid angle × detection efficiency factor
  exp(-μ_self·t) — self-absorption correction for thick samples
  G(·; σ_det) — Gaussian detector response (Si(Li)/SDD resolution ~130 eV FWHM)
  B_E         — background (Rayleigh/Compton scatter, bremsstrahlung)
  n           ~ Poisson photon counting noise
```

**Inverse problem:** Recover the elemental abundance maps ρ_Z(x,y) from the per-pixel XRF spectra S(E; x,y), performing spectral unmixing, peak fitting, background subtraction, and self-absorption correction.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(synchrotron/tube energy E_0) → F(elemental composition/matrix/thickness) → D(SDD detector/MCA)

**Key mismatch parameters:**
- `primary_energy_keV`: Incident X-ray energy E_0; nominal 17 keV, perturbed 10–30 keV
- `detector_resolution_eV`: SDD energy resolution (FWHM at 5.9 keV); nominal 130 eV, perturbed 100–180 eV
- `self_absorption_factor`: Matrix self-absorption severity; nominal 0.9 (10% loss), perturbed 0.6–1.0
- `beam_size_um`: Focused beam diameter; nominal 1 µm, perturbed 0.1–10 µm

**Dataset format:**
- `x_true: (N_elements, H, W)` — ground-truth elemental maps (or single target element map)
- `y: (N_E, H, W)` — per-pixel XRF energy spectra on a 2-D scan grid

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| PyMCA least-squares spectral fitting | Classical analytical | Solé et al., Spectrochim Acta B 62(1):63–68, 2007 | Reference standard for XRF spectrum fitting; batch least-squares per pixel with background modelling |
| AXIL / SNIP background subtraction + peak deconvolution | Classical analytical | Vekemans et al., X-Ray Spectrometry 23(6):278–285, 1994 | Statistics-sensitive non-linear iterative peak fitting; widely used at synchrotron beamlines |
| Non-negative matrix factorisation (NMF) for spectral unmixing | Variational | Lee & Seung, Nature 401:788–791, 1999 | Blind source separation into component spectra and spatial maps; no a-priori elemental knowledge needed |
| Deep learning XRF map enhancement (CNN denoising + super-resolution) | Deep Learning | Schoeder et al., npj Comput Mater 9:75, 2023 | CNN trained to predict high-quality elemental maps from low-dose / fast-scan XRF data |

---

## 4. Literature & State of the Art (2024–2025)

1. **De Nolf et al. (2024)** "Deep learning spectral unmixing for multi-element XRF imaging of Old Master paintings," *Heritage Sci* — CNN-based spectral demixing distinguishing overlapping elemental contributions in historical pigments (Pb white, vermilion, ultramarine).
2. **Longo et al. (2024)** "Compressed sensing XRF mapping: fewer beam dwell times via U-Net reconstruction," *Anal Chem* — sparse sampling strategies with deep reconstruction achieving 10× faster XRF scan acquisition.
3. **Betterton et al. (2025)** "Diffusion model for XRF elemental map super-resolution from low-flux synchrotron scans," *J Synchrotron Rad* — score-based model recovering sub-beam-size spatial detail in elemental maps from coarse raster scans.
4. **Gueriau et al. (2024)** "Automated machine learning pipeline for XRF elemental mapping of fossil specimens," *Sci Rep* — random forest + neural network pipeline for palaeontological XRF datasets, distinguishing taphonomic from biogenic elemental signals.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/xrf_imaging_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/xrf_imaging_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/xrf_imaging_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/xrf_imaging/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing correctly assigns PyMCA least-squares fitting, AXIL/SNIP peak deconvolution, NMF spectral unmixing, and deep-learning map enhancement — covering the standard to advanced XRF data analysis pipeline. The forward model with photoionization cross-sections, fluorescence yield, Gaussian detector response, self-absorption, and Poisson photon noise accurately represents synchrotron and laboratory µ-XRF physics. Mismatch in incident energy, detector resolution, self-absorption, and beam size tests algorithm robustness across different XRF instruments and sample types encountered in cultural heritage, geoscience, and materials applications.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 22.11 | 0.9626 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** FP-Quantify
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 34.51 dB |
| SSIM (sample_00) | 0.8477 |
| Runtime | 0.55 s/sample |

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
| PSNR (sample_00) | 34.56 dB |
| SSIM (sample_00) | 0.8496 |
| Runtime | 0.64 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FP-Quantify
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 34.51 dB |
| SSIM (sample_00) | 0.8477 |
| Runtime | 0.82 s/sample |

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
| PSNR (sample_00) | 34.56 dB |
| SSIM (sample_00) | 0.8496 |
| Runtime | 0.92 s/sample |

**Result: PASS**
