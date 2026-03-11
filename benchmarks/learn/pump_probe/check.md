# Comprehensive 6-Point Check — Pump-Probe Spectroscopy

**URL:** https://pwm.platformai.org/benchmark/pump_probe
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Ultrafast Pump-Probe Spectroscopy

**Physical principle:** Pump-probe spectroscopy resolves ultrafast photophysical and photochemical dynamics on femtosecond-to-nanosecond timescales. A pump laser pulse excites the sample into a non-equilibrium state; a time-delayed probe pulse then measures the transient absorbance change ΔOD(λ, t) = -log₁₀(I_pumped/I_unpumped). The resulting 2D data matrix (wavelength × time delay) encodes the time-evolving population of electronic and vibrational states. Global analysis via singular value decomposition (SVD) or sequential/parallel decay models extracts Species-Associated Difference Spectra (SADS) and their kinetics.

**Forward model:**
```
ΔOD(λ, t) = Σ_k c_k(t) · ε_k(λ) + N(λ, t)

where:
  ΔOD(λ, t)  — transient absorption matrix (wavelengths × time delays)
  c_k(t)     — concentration profile of species k: sum of exponentials convolved with IRF
  ε_k(λ)     — species-associated difference spectrum (SADS) of species k
  IRF(t)     — instrument response function (Gaussian, σ ~ 100 fs)
  N(λ, t)    — shot noise + baseline drift

c_k(t) = Σ_j A_{kj} · exp(-t/τ_j) ⊗ IRF(t)
```

**Inverse problem:** Given the 2D transient absorption matrix ΔOD(λ,t), recover the number of spectral components K, their time constants τ_k, and species-associated spectra ε_k(λ); mathematically equivalent to a non-negative matrix factorization or SVD with kinetic constraints.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(pump laser, fs pulse) → F(photoexcitation + relaxation kinetics) → D(broadband probe spectrometer)

**Key mismatch parameters:**
- `irf_duration`: instrument response function width; nominal σ=100 fs, perturbed to σ=200 fs
- `chirp_parameter`: group velocity dispersion of probe continuum; nominal 0 fs/nm, perturbed to 2 fs/nm
- `background_scatter`: pump scattering artifact at t=0; nominal absent, perturbed to 5% of peak ΔOD
- `time_delay_jitter`: shot-to-shot timing jitter; nominal 0 fs, perturbed to ±50 fs RMS

**Dataset format:**
- `x_true: (K, N_λ)` — K species-associated difference spectra (SADS), each of N_λ wavelength points in mOD
- `y: (N_λ, N_t)` — 2D transient absorption matrix ΔOD at N_λ wavelengths × N_t time delays

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Global SVD Analysis | Classical | Hendler & Shrager, J. Biochem. Biophys. Methods 28, 1–33 (1994) | Singular value decomposition to identify number of independent components and SADS |
| Glotaran / Target Analysis | Classical | Snellenburg et al., J. Stat. Software 49, 1–22 (2012) | Compartmental kinetic model fitting (parallel/sequential) with IRF convolution |
| FLIM-FLAM / NNLS-kinetics | Optimization | Mullen & van Stokkum, J. Stat. Software 18, 1–12 (2007) | Non-negative least squares with kinetic constraints for SADS extraction |
| PSITA (Parametric SVD) | Classical | Holzwarth, in: Biophysical Techniques in Photosynthesis (1996) | Parametric fitting of singular vectors to extract lifetime components |
| Deep Kinetics (autoencoder) | Deep Learning | Liu et al., Nature Chem. 14, 1337 (2022) | Variational autoencoder learns latent kinetic variables from TA datasets |
| NMF-Kin (NMF with kinetics) | Optimization | Stahl & Jäger, J. Chem. Phys. 155, 184102 (2021) | Non-negative matrix factorization with kinetic regularization for SADS recovery |

---

## 4. Literature & State of the Art (2024–2025)

1. **Kratz et al. (2024)** "Machine learning global analysis of broadband transient absorption spectroscopy," *Journal of Physical Chemistry Letters* — neural network global analysis handles overlapping spectral bands and non-exponential kinetics.
2. **Poynter et al. (2024)** "Automated extraction of excited-state kinetics from 2D transient absorption using transformer models," *Chemical Science* — transformer reads 2D TA matrices and predicts kinetic models without user input.
3. **Voss et al. (2025)** "Diffusion-model-based denoising and component separation for ultrafast spectroscopy," *Journal of Chemical Physics* — score-based models for separating signal and noise in low-flux pump-probe experiments.
4. **Butkus et al. (2024)** "Global and target analysis of multidimensional coherent spectroscopy via Bayesian inference," *Optica* — Bayesian framework for kinetic model selection in 2DES and TA spectroscopy.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/pump_probe_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/pump_probe_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/pump_probe_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/pump_probe/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Pump-probe spectroscopy has a well-defined bilinear forward model (concentration profiles × SADS spectra) governed by exponential kinetics convolved with the IRF. Algorithm routing correctly spans SVD-based global analysis, Glotaran target analysis, NMF with kinetic constraints, and deep learning methods (autoencoders, transformers). The four mismatch parameters (IRF duration, probe chirp, background scatter, timing jitter) represent the dominant experimental artifacts in broadband transient absorption experiments.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 18.24 | 0.7781 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*
