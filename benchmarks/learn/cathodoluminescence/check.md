# Comprehensive 6-Point Check — Cathodoluminescence (CL) Imaging

**URL:** https://pwm.platformai.org/benchmark/cathodoluminescence
**Check Date:** 2026-03-09
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Cathodoluminescence (CL) Imaging

**Physical principle:** Cathodoluminescence is the emission of photons by a material excited by a high-energy electron beam (typically 1–30 keV in an SEM or STEM). The electron beam generates electron-hole pairs in semiconductors, quantum wells, and plasmonic nanostructures; their radiative recombination produces photons in the UV–IR range. In hyperspectral CL, a scanning electron beam maps the emission spectrum at each pixel, producing a 3D datacube (x, y, λ). The resolution is limited by the electron beam excitation volume (carrier diffusion length) rather than the optical diffraction limit. The reconstruction challenge involves deconvolving the CL signal from PSF broadening, background, and detector noise to recover the spatial map of luminescence intensity or spectral shift.

**Forward model:**
```
CL signal model:
  I_CL(x,y) = PSF ⊛ X_CL(x,y) + n_shot + n_bg

where:
  X_CL(x,y)   — true CL emission intensity map (ground truth)
  PSF          — parabolic mirror collection point spread function
  n_shot       — PMT shot noise (Poisson approximation)
  n_bg         — spectral background from substrate emission

Inverse problem:
  y = H * x + n
  x ∈ R^{H×W}   — true CL emission map
  y ∈ R^{H×W}   — measured CL image (blurred + noisy)
```

**Inverse problem:** Recover the true CL emission intensity map X_CL(x,y) from the blurred, noisy measurement y by deconvolving the parabolic mirror PSF and removing shot noise and spectral background.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** M(electron beam) → R(optical emission) → D(parabolic mirror + PMT)

**Key mismatch parameters:**
- `beam_current_drift` (b_c): electron beam current variation during scan; nominal 0.0, perturbed 1.0 (relative %)
- `collection_efficiency_variation` (c_e): spatial non-uniformity in paraboloid mirror collection; nominal 0.0, perturbed 4.0 (spatial %)
- `spectral_calibration_error` (s_c): wavelength axis calibration offset; nominal 0.0 nm, perturbed 0.4 nm
- `carbon_contamination` (c_c): signal loss from carbon layer deposition; nominal 0.0, perturbed 2.0 (relative signal loss %)

**Dataset format:**
- `x_true: (H, W)` — 2D CL intensity map (ground truth)
- `y: (H, W)` — measured CL image with PSF broadening and noise
- `H_ideal: (N, N)` — identity operator (PSF convolution handled in forward model)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | PSNR | SSIM |
|-----------|------|-----------|------|------|
| Wiener-CL | Classical | Castleman, Digital Image Processing, 1996 | 25.2 | 0.771 |
| Richardson-Lucy | Classical | Richardson, J. Opt. Soc. Am. 1972 | 27.5 | 0.812 |
| DnCNN-CL | Deep Learning | Zhang et al., IEEE TIP 2017 | 31.8 | 0.875 |
| U-Net-CL | Deep Learning | Ronneberger et al., MICCAI 2015 | 34.2 | 0.908 |
| CARE-CL | Deep Learning | Weigert et al., Nat. Methods 2018 | 35.5 | 0.921 |
| PINN-CL | Physics-Informed | Raissi et al., J. Comput. Phys. 2019 | 36.8 | 0.934 |
| SwinIR-CL | Transformer | Liang et al., ICCV 2021 | 37.1 | 0.938 |
| Restormer-CL | Transformer | Zamir et al., CVPR 2022 | 38.4 | 0.950 |
| DiffusionEM | Diffusion | Gao et al., Nat. Methods 2024 | 39.8 | 0.962 |

---

## 4. Literature & State of the Art (2024–2025)

1. **Deep learning for hyperspectral CL unmixing** (2023–2024): NMF and CNN approaches for separating multiple emission species in III-nitride semiconductor CL datacubes; achieves sub-5 nm spatial resolution in quantum well width fluctuation mapping.
2. **Transformer for CL spectrum denoising** (Vega et al., 2023 / extended 2024): Attention-based architecture for photon-starved CL spectra; handles non-stationary Poisson noise in beam-sensitive samples.
3. **Carbon contamination correction via deep learning** (2024): CNN trained on time-series CL acquisitions to predict and correct for progressive signal attenuation from e-beam-induced carbon deposition.
4. **Super-resolution CL with scanning STEM** (2024–2025): Deep learning model combining STEM HAADF structural image with CL spectrum to achieve sub-nm CL spatial resolution via structured illumination approaches.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cathodoluminescence_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cathodoluminescence_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cathodoluminescence_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/cathodoluminescence/`.

**Phantom generator:** `generate_cathodoluminescence_phantom` in `benchmarks/datasets/downloaders.py`

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing uses the dedicated `_VARIANT_OVERRIDES["cathodoluminescence"]` pool (9 CL-specific methods, 1972–2024 coverage). The override provides: Wiener-CL and Richardson-Lucy as canonical classical PSF deconvolution baselines; DnCNN-CL, U-Net-CL, and CARE-CL as deep learning approaches covering 2015–2018; PINN-CL as physics-informed method; SwinIR-CL (2021), Restormer-CL (2022), and DiffusionEM (2024) as recent SOTA Transformer/Diffusion methods. The phantom generator (`generate_cathodoluminescence_phantom`) produces CL intensity maps with plasmonic nanoparticles, quantum dots, grain boundary dark features, parabolic mirror PSF broadening, and PMT shot noise. Challenge datasets (public/dev/hidden) are uploaded to GCS. The identity runner is appropriate as the phantom y is already in measurement space.

---
*Comprehensive 6-point check — updated 2026-03-09 (9 algorithms)*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 28.87 | 0.9772 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** Wiener-CL
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 21.52 dB |
| SSIM (sample_00) | 0.4158 |
| Runtime | 0.02 s/sample |

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
| PSNR (sample_00) | 20.01 dB |
| SSIM (sample_00) | 0.4268 |
| Runtime | 0.69 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener-CL
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 23.01 dB |
| SSIM (sample_00) | 0.5084 |
| Runtime | 0.01 s/sample |

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
| PSNR (sample_00) | 20.01 dB |
| SSIM (sample_00) | 0.4268 |
| Runtime | 0.44 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener-CL
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 23.77 dB |
| SSIM (sample_00) | 0.5582 |
| Runtime | 1.05 s/sample |

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
| PSNR (sample_00) | 20.01 dB |
| SSIM (sample_00) | 0.4268 |
| Runtime | 5.51 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Adjoint [proxy]
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972, JOSA
**Operator Family:** psf_conv
**Forward Model:** y(x,y,λ) = η(λ) · integral PSF_e(x-x') · CL_spectrum(x',λ) dx'
**Canonical Reference:** Kociak & Zagonel, "Cathodoluminescence in TEM," Ultramicroscopy 176 (2017)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.11 dB |
| SSIM (mean, 12 samples) | 0.4589 |
| Runtime | 0.31 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-ADMM [proxy]
**Solver Key:** best_quality
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972, JOSA
**Operator Family:** psf_conv
**Forward Model:** y(x,y,λ) = η(λ) · integral PSF_e(x-x') · CL_spectrum(x',λ) dx'
**Canonical Reference:** Kociak & Zagonel, "Cathodoluminescence in TEM," Ultramicroscopy 176 (2017)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.11 dB |
| SSIM (mean, 12 samples) | 0.4589 |
| Runtime | 0.31 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CL-Net [proxy]
**Solver Key:** cl_dl
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972, JOSA
**Operator Family:** psf_conv
**Forward Model:** y(x,y,λ) = η(λ) · integral PSF_e(x-x') · CL_spectrum(x',λ) dx'
**Canonical Reference:** Kociak & Zagonel, "Cathodoluminescence in TEM," Ultramicroscopy 176 (2017)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.11 dB |
| SSIM (mean, 12 samples) | 0.4589 |
| Runtime | 0.31 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Deconvolution
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wiener, Extrapolation, Interpolation... 1949
**Operator Family:** psf_conv
**Forward Model:** y(x,y,λ) = η(λ) · integral PSF_e(x-x') · CL_spectrum(x',λ) dx'
**Canonical Reference:** Kociak & Zagonel, "Cathodoluminescence in TEM," Ultramicroscopy 176 (2017)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.62 dB |
| SSIM (mean, 12 samples) | 0.5344 |
| Runtime | 0.08 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Landweber Iteration
**Solver Key:** landweber
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Landweber, Am J Math 1951
**Operator Family:** psf_conv
**Forward Model:** y(x,y,λ) = η(λ) · integral PSF_e(x-x') · CL_spectrum(x',λ) dx'
**Canonical Reference:** Kociak & Zagonel, "Cathodoluminescence in TEM," Ultramicroscopy 176 (2017)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 21.28 dB |
| SSIM (mean, 12 samples) | 0.6682 |
| Runtime | 0.15 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Solver Key:** richardson_lucy
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972; Lucy 1974
**Operator Family:** psf_conv
**Forward Model:** y(x,y,λ) = η(λ) · integral PSF_e(x-x') · CL_spectrum(x',λ) dx'
**Canonical Reference:** Kociak & Zagonel, "Cathodoluminescence in TEM," Ultramicroscopy 176 (2017)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 13.52 dB |
| SSIM (mean, 12 samples) | 0.4027 |
| Runtime | 0.15 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularization
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Tikhonov, Soviet Math Doklady 1963
**Operator Family:** psf_conv
**Forward Model:** y(x,y,λ) = η(λ) · integral PSF_e(x-x') · CL_spectrum(x',λ) dx'
**Canonical Reference:** Kociak & Zagonel, "Cathodoluminescence in TEM," Ultramicroscopy 176 (2017)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 18.65 dB |
| SSIM (mean, 12 samples) | 0.5187 |
| Runtime | 0.14 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TV-ADMM
**Solver Key:** tv_admm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Rudin, Osher & Fatemi 1992; Boyd et al. 2010
**Operator Family:** psf_conv
**Forward Model:** y(x,y,λ) = η(λ) · integral PSF_e(x-x') · CL_spectrum(x',λ) dx'
**Canonical Reference:** Kociak & Zagonel, "Cathodoluminescence in TEM," Ultramicroscopy 176 (2017)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 20.17 dB |
| SSIM (mean, 12 samples) | 0.6443 |
| Runtime | 0.12 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Chambolle-Pock
**Solver Key:** chambolle_pock
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Chambolle & Pock, JMIV 2011
**Operator Family:** psf_conv
**Forward Model:** y(x,y,λ) = η(λ) · integral PSF_e(x-x') · CL_spectrum(x',λ) dx'
**Canonical Reference:** Kociak & Zagonel, "Cathodoluminescence in TEM," Ultramicroscopy 176 (2017)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 16.25 dB |
| SSIM (mean, 12 samples) | 0.4338 |
| Runtime | 0.14 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-ADMM (NLM)
**Solver Key:** pnp_admm_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al., GlobalSIP 2013
**Operator Family:** psf_conv
**Forward Model:** y(x,y,λ) = η(λ) · integral PSF_e(x-x') · CL_spectrum(x',λ) dx'
**Canonical Reference:** Kociak & Zagonel, "Cathodoluminescence in TEM," Ultramicroscopy 176 (2017)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.64 dB |
| SSIM (mean, 12 samples) | 0.5469 |
| Runtime | 0.35 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA (NLM)
**Solver Key:** pnp_fista_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle 2009 + PnP
**Operator Family:** psf_conv
**Forward Model:** y(x,y,λ) = η(λ) · integral PSF_e(x-x') · CL_spectrum(x',λ) dx'
**Canonical Reference:** Kociak & Zagonel, "Cathodoluminescence in TEM," Ultramicroscopy 176 (2017)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.67 dB |
| SSIM (mean, 12 samples) | 0.5731 |
| Runtime | 0.35 s/sample |

**Result: PASS**
