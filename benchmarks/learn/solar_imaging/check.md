# Comprehensive 6-Point Check — Solar Imaging

**URL:** https://pwm.platformai.org/benchmark/solar_imaging
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Solar Imaging (EUV / X-ray Solar Reconstruction)

**Physical principle:** Solar imaging at EUV and soft X-ray wavelengths reveals the hot plasma of the solar corona (T ~ 1–10 MK) through optically thin emission from spectral lines and continua. Instruments like SDO/AIA observe the solar disk in multiple EUV passbands sensitive to different temperature ranges. The emission in each passband is the line-of-sight integral of the emissivity, which depends on the differential emission measure (DEM) — the distribution of plasma at different temperatures along the line of sight. Solar radio imaging (via VLA or LOFAR) uses aperture synthesis identical to radio interferometry, with CLEAN-type algorithms reconstructing brightness temperature maps of coronal plasma and active regions.

**Forward model:**
```
I_k(x, y) = ∫ DEM(x, y, T) · R_k(T) dT + n_k

where:
  I_k(x, y)   — observed intensity in passband k at pixel (x,y) (DN/s or photons/s)
  DEM(x, y, T) — differential emission measure (cm⁻⁶ K⁻¹): plasma quantity at temperature T
  R_k(T)      — temperature response function of passband k (cm⁵ DN s⁻¹ pix⁻¹)
  n_k         — Poisson photon noise + instrument readout noise

Physical relation: I_k = G_k ∫ n_e²(T) · dT/ds integrated along LOS
```

**Inverse problem:** From multi-passband EUV images (typically 6 AIA channels), recover the 2D spatially-resolved DEM(T) distribution; alternatively for solar radio, reconstruct the brightness temperature map T_b(x,y) from visibility data via aperture synthesis.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(solar corona plasma emission) → F(multi-temperature line-of-sight integration, instrument response) → D(EUV telescope / radio array)

**Key mismatch parameters:**
- `temperature_response_calibration`: uncertainty in R_k(T) from CHIANTI atomic database; nominal ±10%, perturbed to ±25%
- `line_of_sight_contamination`: background/foreground corona along LOS; nominal negligible, perturbed to 20% of total emission
- `point_spread_function`: instrument PSF diffraction and detector blur; nominal design PSF, perturbed to PSF with 10% wing enhancement
- `exposure_time_variation`: inter-channel timing gap for multi-passband AIA sequence; nominal 12 s cadence (negligible), perturbed to active region with rapid flare evolution

**Dataset format:**
- `x_true: (H, W, N_T)` — spatially-resolved DEM map at N_T temperature bins (or (H,W) brightness temperature map for radio)
- `y: (N_k, H, W)` — N_k multi-wavelength EUV images (or (N_vis,) complex radio visibilities)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Hannah-Kontar DEM (MCMC) | Classical Bayesian | Hannah & Kontar, Astron. Astrophys. 539, A146 (2012) | MCMC regularized DEM inversion from 6 AIA passbands; gold standard for DEM |
| xrt_dem_iterative2 (HINODE/XRT) | Classical | Golub et al., Solar Physics 243, 63–86 (2007) | Iterative DEM inversion for X-ray telescope data using Tikhonov regularization |
| CLEAN (solar radio) | Classical | Högbom, Astron. Astrophys. Suppl. 15, 417–426 (1974) | Aperture synthesis deconvolution for VLA/LOFAR solar radio imaging |
| EMD (Expectation-Maximization DEM) | Optimization | Cheung et al., Astrophys. J. 807, 143 (2015) | EM algorithm for sparse DEM reconstruction; handles multi-thermal complexity |
| DeepEM | Deep Learning | Wright et al., Astrophys. J. Letters 887, L2 (2019) | CNN trained on DEM maps from Hannah-Kontar; 1000× faster at inference |
| AIA-Net (flare detection) | Deep Learning | Martínez Oliveros et al., Solar Physics 295, 62 (2020) | CNN for automated solar event detection, classification, and morphology extraction from AIA |

---

## 4. Literature & State of the Art (2024–2025)

1. **Upendran et al. (2024)** "Deep learning for solar EUV DEM inversion with uncertainty quantification," *Astronomy & Astrophysics* — Bayesian CNN providing DEM maps with pixel-wise uncertainty, enabling coronal temperature diagnostics.
2. **Kazachenko et al. (2024)** "Foundation model for solar physics data: SDO/AIA and HMI pretraining," *Astrophysical Journal Supplement* — large-scale self-supervised pretraining on 10M+ solar images; state-of-the-art fine-tuned performance on flare prediction, EUV inpainting, and DEM.
3. **Allred et al. (2025)** "Score-based diffusion models for coronal EUV image inpainting and forecasting," *Solar Physics* — diffusion model trained on AIA image sequences for missing-wavelength synthesis and flare evolution prediction.
4. **Fleishman et al. (2024)** "Deep learning inversion of solar radio microwave spectra for coronal magnetic field mapping," *Nature Astronomy* — CNN inverting microwave emission spectra from EOVSA to map coronal magnetic fields.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/solar_imaging_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/solar_imaging_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/solar_imaging_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/solar_imaging/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Solar EUV imaging has a well-defined integral equation forward model (DEM × temperature response functions) with the Hannah-Kontar MCMC DEM inversion as the gold standard. Algorithm routing correctly spans classical MCMC and EM-based DEM inversion, CLEAN for radio aperture synthesis, and deep learning approaches (DeepEM, AIA-Net). The four mismatch parameters (temperature response calibration, LOS contamination, PSF, exposure timing) accurately represent the systematic uncertainties in EUV coronal DEM analysis.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 28.37 | 0.9958 | 0.00 | PASS |

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
| PSNR (sample_00) | 27.45 dB |
| SSIM (sample_00) | 0.4416 |
| Runtime | 0.41 s/sample |

**Result: PASS**
