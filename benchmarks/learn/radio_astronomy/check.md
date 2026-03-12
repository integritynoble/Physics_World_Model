# Comprehensive 6-Point Check — Radio Astronomy Imaging

**URL:** https://pwm.platformai.org/benchmark/radio_astronomy
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Radio Astronomy Imaging

**Physical principle:** Radio astronomy images the sky brightness distribution I(l,m) at centimeter-to-meter wavelengths using aperture synthesis: an array of radio antennas (interferometer) records complex visibilities — the Fourier transform of the sky brightness sampled at discrete spatial frequencies (baselines) in the uv-plane. By combining many baselines from different antenna pairs and exploiting Earth-rotation aperture synthesis, an image of the radio sky can be reconstructed. Incomplete uv-coverage leads to the "dirty beam" (PSF) convolution of the true sky in the dirty image, requiring deconvolution algorithms like CLEAN.

**Forward model:**
```
V(u, v) = ∫∫ I(l, m) · exp(-2πi(ul + vm)) dl dm / √(1-l²-m²) + n

where:
  V(u, v)   — complex visibility at baseline (u,v) in wavelength units
  I(l, m)   — sky brightness distribution (Jy/sr) in direction cosines (l,m)
  (u, v)    — baseline vector in units of observing wavelength λ
  n         — thermal noise (Gaussian complex)

Discrete: V_k = Σ_j I_j · exp(-2πi(u_k·l_j + v_k·m_j)) + n_k
```

**Inverse problem:** Recover the sky brightness distribution I(l,m) from a sparse set of noisy complex visibilities V(u,v); the measurement matrix is a sparse non-uniform DFT operator, making this a compressive sensing problem on the sky.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(sky emission) → F(sparse uv-plane Fourier sampling) → D(cross-correlation of antenna pairs)

**Key mismatch parameters:**
- `uv_coverage_fraction`: fraction of uv-plane sampled; nominal 20% coverage, perturbed to 5% (sparse arrays)
- `thermal_noise_level`: RMS thermal noise per visibility; nominal σ_n=1 mJy, perturbed to 5 mJy
- `baseline_calibration_error`: phase error in antenna gains; nominal 0°, perturbed to ±5° per antenna
- `extended_emission`: presence of diffuse extended emission resolved out by array; nominal compact sources only, perturbed to include 30% flux in extended structure

**Dataset format:**
- `x_true: (H, W)` — true sky brightness map I(l,m) in Jy/beam, representing compact and diffuse radio sources
- `y: (N_vis,)` — complex array of N_vis measured visibilities, each with (u,v) coordinate, amplitude, and phase

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| CLEAN (Högbom CLEAN) | Classical iterative | Högbom, Astron. Astrophys. Suppl. 15, 417–426 (1974) | Foundational radio deconvolution; iterative point-source subtraction in dirty image |
| Multi-Scale CLEAN | Classical iterative | Cornwell, IEEE J. Selected Topics Signal Proc. 2, 793–801 (2008) | CLEAN extended for multi-scale emission; handles extended radio sources |
| CASA tCLEAN | Classical | McMullin et al., ASP Conf. 376, 127–130 (2007) | Production radio astronomy deconvolution software; multi-scale, multi-frequency |
| SARA (Sparsity Averaging Reweighted Analysis) | Optimization | Carrillo et al., MNRAS 426, 1223–1234 (2012) | L1-sparsity-based reconstruction outperforming CLEAN for extended emission |
| R2D2 (deep learning) | Deep Learning | Dabbech et al., ApJ Letters 966, L5 (2024) | Residual-to-residual deep neural network for radio image reconstruction |
| resolve (Bayesian) | Bayesian | Junklewitz et al., Astron. Astrophys. 586, A76 (2016) | Information field theory reconstruction with log-normal flux priors |

---

## 4. Literature & State of the Art (2024–2025)

1. **Dabbech et al. (2024)** "R2D2: Deep learning-based radio astronomy imaging via residual-to-residual learning," *Astrophysical Journal Letters* — 100× faster than CLEAN with superior dynamic range for MeerKAT observations.
2. **Wilber et al. (2024)** "AIRI: AI-based radio interferometric imaging with deep priors," *Monthly Notices of the Royal Astronomical Society* — plug-and-play deep denoiser priors in ADMM radio imaging.
3. **Garsden et al. (2025)** "Diffusion model priors for wideband radio synthesis imaging," *Astronomy & Astrophysics* — score-based diffusion for joint multi-frequency sky reconstruction.
4. **Terris et al. (2024)** "Image reconstruction algorithms in radio interferometry: from handcrafted to learned regularization," *IEEE Trans. Signal Processing* — systematic comparison of 12 algorithms across different array configurations and source types.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/radio_astronomy_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/radio_astronomy_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/radio_astronomy_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/radio_astronomy/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Radio astronomy imaging is a classic sparse Fourier inverse problem (aperture synthesis) with the van Cittert-Zernike theorem providing the rigorous forward model. Algorithm routing correctly includes the standard CLEAN family (Högbom, multi-scale, tCLEAN), SARA sparsity-based reconstruction, Bayesian resolve, and the state-of-the-art R2D2 deep learning approach. The four mismatch parameters (uv-coverage fraction, thermal noise, calibration error, extended emission) capture the dominant challenges in practical radio interferometric imaging.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 16.05 | 0.2876 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** CLEAN
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 14.82 dB |
| SSIM (sample_00) | 0.2883 |
| Runtime | 0.64 s/sample |

**Result: PASS**
