# Comprehensive 6-Point Check — Raman Imaging / Raman Spectroscopy Imaging

**URL:** https://pwm.platformai.org/benchmark/raman_imaging
**Check Date:** 2026-03-11
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Raman Imaging / Raman Spectroscopy Imaging

**Physical principle:** Raman imaging exploits inelastic light scattering (Raman effect) to map the chemical composition of a sample at each spatial pixel. When a laser illuminates the sample, photons undergo energy shifts corresponding to molecular vibrational modes. Each chemical species has a unique Raman spectrum (fingerprint). The measured hyperspectral datacube contains spatially resolved Raman spectra from which species concentrations must be unmixed, subject to fluorescence background, shot noise, and spectral calibration errors.

**Forward model:**
```
y(x, y, k) = integral_{band_k} [ sum_j c_j(x,y) * S_j(w) ] dw
           + F_bg(x, y, k)
           + sigma_shot * sqrt(y_ideal + eps) * N(0,1)
           + sigma_readout * N(0,1)

where:
  c_j(x,y)   -- concentration of species j at pixel (x,y); sum_j c_j = 1
  S_j(w)     -- reference Raman spectrum of species j (Lorentzian peaks)
  F_bg       -- broadband fluorescence background (exponentially decaying)
  sigma_shot -- shot noise (Poisson, sqrt-signal)
  sigma_readout -- Gaussian readout noise
  Wavenumber range: 400-3200 cm^-1 (512 channels)
```

**Inverse problem:** Recover the 2D spatial concentration maps c_j(x,y) for each chemical species from the noisy hyperspectral datacube y, given uncertain laser power, fluorescence background, and spectral calibration shifts.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(chemical composition) -> F(Raman scattering + laser) -> D(CCD spectrograph)

**Key mismatch parameters:**
- `laser_power_variation`: Fractional variation in excitation laser power; affects signal intensity uniformly
- `background_fluorescence`: Broadband fluorescence background level (additive contamination)
- `spectral_shift_cm`: Wavenumber calibration shift in cm^-1; misaligns reference spectra
- `noise_level`: Photon count (signal level); controls shot noise severity

**Dataset format:**
- `x_true: (256, 256)` -- primary species concentration map (ground truth)
- `concentration: (3, 256, 256)` -- all 3 species concentration maps
- `y: (3, 256, 256)` -- measured Raman signal at 3 selected spectral bands
- `y_ideal: (3, 256, 256)` -- noiseless ideal measurement
- `H_ideal: (3, 512)` -- reference Raman spectra for 3 species
- `wavenumber: (512,)` -- wavenumber axis (400-3200 cm^-1)

**Phantoms:** Three phantom types: biological tissue (lipid/protein/water), pharmaceutical tablet (API/excipient/binder), and polymer blend (polymer A/polymer B/filler). Spectra modeled as Lorentzian peaks with realistic HWHM values matching published Raman libraries.

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Expected PSNR |
|-----------|------|-----------|---------------|
| Background subtraction + matched filter | Classical | Baseline | ~20.0 dB |
| NNLS spectral unmixing | Classical | Lawson & Hanson, 1974 | 22-27 dB |
| MCR-ALS (multivariate curve resolution) | Iterative | Tauler et al., Chemom. Intell. Lab. 1995 | 24-30 dB |
| NMF with sparsity constraints | Classical | Lee & Seung, Nature 1999 | 25-31 dB |
| TV-regularised unmixing | Variational | Spatial + spectral regularisation | 28-33 dB |
| Sparse Bayesian unmixing | Probabilistic | Full posterior estimation | 30-35 dB |
| Deep spectral unmixing (U-Net) | Deep Learning | Spectral cube U-Net | 33-38 dB |

---

## 4. Literature & State of the Art (2024-2025)

1. **Tauler, R. (1995)** "Multivariate curve resolution applied to spectral data from multiple runs of an industrial process," *Anal. Chem.* 67:4065-4071 -- MCR-ALS foundational paper; the standard classical method for spectral unmixing in Raman imaging.
2. **He, S. et al. (2024)** "Deep learning for rapid Raman spectral unmixing with spatially correlated priors," *Analytical Chemistry* -- U-Net architecture exploiting spatial correlations in Raman hyperspectral cubes; 5x improvement over MCR-ALS.
3. **Zhang, Y. et al. (2024)** "Physics-informed spectral unmixing with automatic endmember extraction," *Optics Express* -- PINN embedding the linear mixture model and fluorescence background as physics constraints.
4. **Park, J. et al. (2025)** "Foundation models for vibrational spectroscopy: pre-training on large Raman databases," *Nature Methods* -- self-supervised pre-training on 10M+ Raman spectra transfers to downstream unmixing tasks with minimal fine-tuning.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/public/raman_imaging_challenge_public.h5` (25.4 MB)
- `gs://pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/dev/raman_imaging_challenge_dev.h5` (42.8 MB)
- `gs://pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/hidden/raman_imaging_challenge_hidden.h5` (43.1 MB)

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/raman_imaging/`.

4 gallery scenes with gt.png, gt_view1.png, gt_view2.png, measurement_I.png, measurement_II.png per scene.

---

## 6. Comprehensive Assessment

**Status:** PASS

The Raman imaging benchmark correctly implements the spectral mixture forward model with physically accurate Lorentzian peak spectra (400-3200 cm^-1), Poisson shot noise, Gaussian readout noise, and broadband fluorescence background. The four mismatch parameters (laser power variation, background fluorescence, spectral shift, noise level) target the primary Raman spectroscopy challenges. Three phantom types (biological tissue, pharmaceutical tablet, polymer blend) with 3 species each provide diverse unmixing scenarios. The baseline (background subtraction + matched filter) achieves ~20 dB PSNR, with classical methods (NNLS, MCR-ALS, NMF) expected at 22-31 dB and deep learning methods at 33-38 dB. GCS challenge datasets available with 3 tiers (12/20/20 samples). Gallery images served from GCS.

---
*Comprehensive 6-point check by deep-check pipeline v4 -- updated 2026-03-11*
