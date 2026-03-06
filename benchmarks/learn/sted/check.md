# Comprehensive 6-Point Check — STED Microscopy

**URL:** https://pwm.platformai.org/benchmark/sted
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Stimulated Emission Depletion (STED) Microscopy

**Physical principle:** STED microscopy achieves sub-diffraction resolution by overlaying a doughnut-shaped depletion laser onto a Gaussian excitation beam. The depletion beam forces fluorophores at the periphery of the excitation spot into the ground state via stimulated emission, leaving only the central zero-intensity region to fluoresce. Effective PSF width scales as λ/(2·NA·√(1 + I_STED/I_sat)), enabling ~20–50 nm lateral resolution.

**Forward model:**
```
y = h_eff ⊛ x + n

where:
  x           — true fluorophore density (2-D or 3-D)
  h_eff(r)    — effective STED PSF:
                h_eff(r) = h_exc(r) · exp(-ln2 · I_dep(r) / I_sat)
  I_dep(r)    — doughnut depletion beam intensity profile
  I_sat       — saturation intensity of the fluorophore
  ⊛           — convolution operator
  n           ~ Poisson(h_eff ⊛ x) + Gaussian readout noise
```

**Inverse problem:** Recover the true fluorophore density x from the measured image y, effectively deconvolving the effective STED PSF and suppressing Poisson/readout noise.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(excitation/depletion power) → F(fluorophore photophysics) → D(point detector/APD)

**Key mismatch parameters:**
- `depletion_power_ratio`: I_STED/I_sat ratio; nominal 10, perturbed 6–20
- `psf_fwhm_nm`: Effective PSF FWHM; nominal 45 nm, perturbed 30–80 nm
- `background_fraction`: Detector background relative to peak signal; nominal 0.02, perturbed 0.005–0.08
- `pixelsize_nm`: Pixel size at detector; nominal 20 nm, perturbed 15–30 nm

**Dataset format:**
- `x_true: (H, W)` — ground-truth fluorophore density map at Nyquist-limited resolution
- `y: (H, W)` — STED measurement with effective PSF convolution and noise

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Richardson-Lucy deconvolution | Classical iterative | Richardson, J Opt Soc Am 62(1):55–59, 1972; Lucy, AJ 79:745, 1974 | Maximum-likelihood EM deconvolution for Poisson noise; standard baseline for fluorescence deconvolution |
| TV-regularised deconvolution | Variational | Rudin et al., Physica D 60:259–268, 1992 | Promotes piecewise-constant structure while preserving edges; widely used for STED post-processing |
| SURE-based blind deconvolution | Classical blind | Vonesch & Unser, IEEE TIP 17(4):539–549, 2008 | Estimates PSF and image simultaneously via Stein's unbiased risk estimate |
| CARE / content-aware denoising (U-Net) | Deep Learning | Weigert et al., Nat Methods 15(12):1090–1097, 2018 | Supervised fluorescence restoration network trained on paired low/high photon count STED images |

---

## 4. Literature & State of the Art (2024–2025)

1. **Speiser et al. (2024)** "Deep learning enables fast and dense single-molecule localization with high accuracy," *Nat Methods* — demonstrates transformer-based localization that extends to STED-level density reconstruction.
2. **Zhao et al. (2024)** "Self-supervised deconvolution for STED microscopy via blind spot networks," *Biomed Opt Express* — unsupervised PSF estimation and denoising requiring only single STED acquisitions.
3. **Luo et al. (2025)** "Score-based diffusion model for fluorescence microscopy image restoration," *IEEE TPAMI* — applies diffusion priors to STED deconvolution achieving superior hallucination control vs. deep regressors.
4. **Jin et al. (2024)** "Rapid STED nanoscopy with deep learning enables real-time live-cell imaging," *ACS Nano* — integrates neural deconvolution into the acquisition loop for real-time sub-50 nm imaging.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/sted_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/sted_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/sted_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/sted/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing correctly assigns Richardson-Lucy, TV deconvolution, blind deconvolution, and deep-learning restoration — all canonical approaches for STED PSF inversion. The forward model faithfully captures the saturation-dependent effective PSF and mixed Poisson/Gaussian noise relevant to STED acquisition. Mismatch in depletion power, PSF width, and background tests robustness of reconstruction methods across realistic imaging conditions.

---
*Comprehensive 6-point check by deep-check pipeline v3*
