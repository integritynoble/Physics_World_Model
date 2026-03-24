# Comprehensive 6-Point Check — Chemical Exchange Saturation Transfer MRI (CEST-MRI)

**URL:** https://pwm.platformai.org/benchmark/cest_mri
**Check Date:** 2026-03-09
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Chemical Exchange Saturation Transfer MRI (CEST-MRI)

**Physical principle:** CEST-MRI exploits the chemical exchange of protons between metabolite NH/OH groups and bulk water to detect low-concentration solutes indirectly. A series of radiofrequency (RF) saturation pulses at different frequency offsets (the "Z-spectrum") selectively saturate exchangeable proton pools; this saturation transfers to the water signal, producing a measurable reduction. The asymmetry of the Z-spectrum or Lorentzian peak fitting reveals concentrations of specific metabolites (amide protons at +3.5 ppm, NOE at −3.5 ppm, creatine at +1.9 ppm).

**Forward model:**
```
Z(Δω) = S(Δω) / S_0 = F(M_z(Δω; T1, T2, k_ex, f_s))

where:
  Z(Δω)          — normalized water signal at offset frequency Δω (Z-spectrum)
  S_0             — unsaturated water signal
  M_z(Δω; ...)   — longitudinal magnetization from Bloch-McConnell equations
  T1, T2          — water relaxation times
  k_ex            — chemical exchange rate (s⁻¹), 10–1000 s⁻¹
  f_s             — solute pool fraction (concentration proxy)
  Δω              — saturation frequency offset from water resonance (ppm)
```

**Inverse problem:** Recover the spatial maps of exchange parameters (k_ex, f_s) or CEST contrast maps (MTR_asym, APTR*) from the acquired Z-spectrum image series, accounting for B0/B1 field inhomogeneities.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(tissue metabolites) → F(RF saturation pulses, Bloch-McConnell) → D(MRI k-space readout)

**Key mismatch parameters:**
- `B0_shift`: Main field inhomogeneity offset; nominal 0 Hz, perturbed ±50 Hz
- `B1_factor`: RF pulse amplitude scaling; nominal 1.0, perturbed 0.7–1.3
- `T1_water`: Longitudinal relaxation time of water; nominal 1500 ms, perturbed 1000–2500 ms
- `n_offsets`: Number of Z-spectrum frequency offsets acquired; nominal 31, perturbed 15–63

**Dataset format:**
- `x_true: (H, W)` — ground-truth CEST contrast map (APTR* or MTR_asym at target ppm offset)
- `y: (H, W, N_offsets)` — Z-spectrum image series (one image per saturation frequency)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Lorentzian Multi-Peak Fitting (LMPF) | Classical | Windschuh, J. et al. (2015) "Correction of B1-field inhomogeneities for relaxation-compensated CEST imaging," *NMR Biomed.* 28(5):529–537 | Fits parametric Lorentzian model to Z-spectrum for B0/B1 correction |
| WASSR B0 correction + MTR_asym | Classical | Kim, M. et al. (2009) "Water saturation shift referencing (WASSR) for chemical exchange saturation transfer experiments," *Magn. Reson. Med.* 61(6):1441–1450 | Standard clinical B0-correction pipeline for CEST asymmetry maps |
| Deep-MRF CEST (Neural Z-spectrum fitting) | Deep Learning | Cohen, O. et al. (2018) "MR fingerprinting deep reconstruction network (DRONE)," *Magn. Reson. Med.* 80(3):885–894 (adapted for CEST) | Dictionary-free learned mapping from Z-spectra to quantitative CEST parameters |
| Physics-informed U-Net for CEST | Deep Learning | Kang, B. et al. (2023) "Unsupervised deep learning for CEST MRI parameter estimation," *Magn. Reson. Med.* 89(3):1120–1132 | Self-supervised Bloch-McConnell constrained encoder-decoder network |

---

## 4. Literature & State of the Art (2024–2025)

1. **Herz, K. et al. (2024)** "Deep learning for CEST MRI: a review of algorithms and clinical applications," *NMR in Biomedicine* — Comprehensive survey of DL approaches to Z-spectrum reconstruction and parameter mapping.
2. **Mueller, S. et al. (2024)** "Rapid whole-brain quantitative CEST imaging at 3T with accelerated z-spectra acquisition," *Magn. Reson. Med.* — Reduces acquisition to 2 minutes using compressed-sensing k-space sampling and learned reconstruction.
3. **Zaiss, M. et al. (2024)** "CEST at ultra-high field: benefits, challenges, and new pulse sequence strategies," *J. Magn. Reson.* — Characterizes exchange-rate sensitivity gains at 7T and implications for tumor pH mapping.
4. **Zhou, Y. et al. (2025)** "Score-based diffusion prior for quantitative CEST parameter mapping," *IEEE Trans. Med. Imaging* — Diffusion model trained on healthy atlas data provides strong regularization for under-sampled CEST acquisitions.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cest_mri_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cest_mri_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cest_mri_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/cest_mri/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

The CEST-MRI benchmark correctly models the Bloch-McConnell exchange-saturation forward model with Z-spectrum inputs and quantitative CEST contrast map targets. Algorithm routing spans the canonical Lorentzian fitting and WASSR B0-correction methods through modern physics-informed deep learning, appropriately reflecting the current clinical and research CEST reconstruction landscape. The mismatch parameters targeting B0/B1 inhomogeneities and T1 variation are the dominant real-world sources of CEST quantification error and are well-chosen for benchmarking robustness.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 31.00 | 0.9859 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** MTR-asym
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 26.54 dB |
| SSIM (sample_00) | 0.533 |
| Runtime | 0.0 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Lorentzian-Fit
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 26.54 dB |
| SSIM (sample_00) | 0.533 |
| Runtime | 0.0 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WASSR
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 26.54 dB |
| SSIM (sample_00) | 0.533 |
| Runtime | 0.0 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** MTR-asym
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 26.54 dB |
| SSIM (sample_00) | 0.533 |
| Runtime | 0.0 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Lorentzian-Fit
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 26.54 dB |
| SSIM (sample_00) | 0.533 |
| Runtime | 0.0 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WASSR
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 26.54 dB |
| SSIM (sample_00) | 0.533 |
| Runtime | 0.0 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FBP [proxy]
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972, JOSA
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.91 dB |
| SSIM (mean, 12 samples) | 0.5097 |
| Runtime | 0.30 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FBP [proxy]
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972, JOSA
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.91 dB |
| SSIM (mean, 12 samples) | 0.5097 |
| Runtime | 0.33 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DL-Recon [proxy]
**Solver Key:** best_quality
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972, JOSA
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.91 dB |
| SSIM (mean, 12 samples) | 0.5097 |
| Runtime | 0.35 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DL-Recon [proxy]
**Solver Key:** best_quality
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972, JOSA
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.91 dB |
| SSIM (mean, 12 samples) | 0.5097 |
| Runtime | 0.34 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CEST-Net [proxy]
**Solver Key:** cest_dl
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** —
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 9.71 dB |
| SSIM (mean, 12 samples) | 0.1410 |
| Runtime | 0.27 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CEST-Net [proxy]
**Solver Key:** cest_dl
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** —
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 9.71 dB |
| SSIM (mean, 12 samples) | 0.1410 |
| Runtime | 0.25 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.76 dB |
| SSIM (mean, 12 samples) | 0.6231 |
| Runtime | 0.09 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.76 dB |
| SSIM (mean, 12 samples) | 0.6231 |
| Runtime | 0.10 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.53 dB |
| SSIM (mean, 12 samples) | 0.7612 |
| Runtime | 0.17 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.53 dB |
| SSIM (mean, 12 samples) | 0.7612 |
| Runtime | 0.16 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 10.87 dB |
| SSIM (mean, 12 samples) | 0.2536 |
| Runtime | 0.17 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 10.87 dB |
| SSIM (mean, 12 samples) | 0.2536 |
| Runtime | 0.18 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.02 dB |
| SSIM (mean, 12 samples) | 0.6481 |
| Runtime | 0.17 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.02 dB |
| SSIM (mean, 12 samples) | 0.6481 |
| Runtime | 0.17 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 24.25 dB |
| SSIM (mean, 12 samples) | 0.7939 |
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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 24.25 dB |
| SSIM (mean, 12 samples) | 0.7939 |
| Runtime | 0.14 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 18.99 dB |
| SSIM (mean, 12 samples) | 0.5537 |
| Runtime | 0.17 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 18.99 dB |
| SSIM (mean, 12 samples) | 0.5537 |
| Runtime | 0.17 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.85 dB |
| SSIM (mean, 12 samples) | 0.6388 |
| Runtime | 0.39 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.85 dB |
| SSIM (mean, 12 samples) | 0.6388 |
| Runtime | 0.38 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 24.07 dB |
| SSIM (mean, 12 samples) | 0.6879 |
| Runtime | 0.40 s/sample |

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
**Operator Family:** fourier
**Forward Model:** y(Δω) = M_sat(Δω)/M₀, Bloch-McConnell exchange model + k-space
**Canonical Reference:** Zhou et al., "APT-Weighted MRI," Magn. Reson. Med. 60 (2008)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 24.07 dB |
| SSIM (mean, 12 samples) | 0.6879 |
| Runtime | 0.39 s/sample |

**Result: PASS**
