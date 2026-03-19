# Comprehensive 6-Point Check — Neutron Powder Diffraction

**URL:** https://pwm.platformai.org/benchmark/neutron_diffraction
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Neutron Powder Diffraction (NPD)

**Physical principle:** Thermal neutrons have de Broglie wavelengths comparable to interatomic spacings (0.5–3 Å), enabling Bragg diffraction from crystalline materials. Unlike X-rays (which scatter from electrons), neutrons scatter from atomic nuclei, giving unique sensitivity to light elements (H, Li, C, O) and the ability to distinguish isotopes. A powder diffraction pattern is the histogram of scattered neutron counts as a function of scattering angle 2θ (or time-of-flight d-spacing at pulsed sources), with Bragg peaks encoding the crystal structure.

**Forward model:**
```
I(d) = Σ_{hkl} S(hkl) · |F(hkl)|² · M(hkl) · L(hkl) · E(d) · P(d;hkl) + B(d) + η

where:
  d          — d-spacing (Å) = λ / (2 sin θ) or TOF channel
  F(hkl)     — structure factor for reflection hkl (encodes atomic positions)
  S(hkl)     — scale factor and multiplicity
  M(hkl)     — Lorentz-polarization factor
  E(d)       — extinction and absorption correction
  P(d;hkl)   — peak profile function (pseudo-Voigt for CW; back-to-back exponentials for TOF)
  B(d)       — background (inelastic scattering, air scattering)
  η          — Poisson counting noise
```

**Inverse problem:** Recover crystal structure parameters (unit cell, atomic positions, thermal factors, site occupancies) from the measured diffraction pattern I(d) via Rietveld refinement.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(neutron beam, λ or TOF) → F(polycrystalline sample) → D(detector bank)

**Key mismatch parameters:**
- `peak_fwhm_A`: Bragg peak full-width at half-maximum in d-spacing; nominal 0.01 Å, perturbed 0.02–0.04 Å
- `background_level`: ratio of background counts to Bragg peak maximum; nominal 0.05, perturbed 0.15–0.30
- `preferred_orientation`: degree of texture (March-Dollase parameter); nominal 1.0 (random), perturbed 0.7–0.9
- `counting_time_s`: total counting time affecting Poisson noise level; nominal 3600 s, perturbed 300–600 s

**Dataset format:**
- `x_true: (N_params,)` — vector of crystal structure parameters (unit cell + atomic coordinates + ADPs)
- `y: (N_bins,)` — diffraction pattern histogram with ~3000–8000 d-spacing bins

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Rietveld Refinement (GSAS-II / FullProf) | Classical | Rietveld (1969) *J. Appl. Cryst.* 2:65–71; Toby & Von Dreele (2013) *J. Appl. Cryst.* 46:544 | Gold-standard least-squares crystal structure refinement from powder patterns |
| Pawley / Le Bail Extraction | Classical | Pawley (1981) *J. Appl. Cryst.* 14:357; Le Bail et al. (1988) *Mater. Res. Bull.* 23:447 | Pattern decomposition to extract integrated intensities without structural model |
| Monte Carlo / Simulated Annealing (GSAS, DASH) | Variational | David et al. (2006) *Structure Determination from Powder Diffraction Data* (IUCr Monographs) | Global optimization for ab initio structure solution from powder data |
| Deep Learning Structure Prediction (CrystalNet / ML-RMC) | Deep Learning | Park et al. (2023) *npj Comput. Mater.* 9:12 | Graph neural network for crystal structure prediction from diffraction features |

---

## 4. Literature & State of the Art (2024–2025)

1. **Banerjee et al. (2024)** "Machine learning-accelerated Rietveld refinement for in-situ neutron diffraction," *Acta Crystallographica A* — trained a surrogate model to replace iterative Rietveld refinement, enabling real-time structural tracking during battery cycling at neutron beamlines.
2. **Samarakoon et al. (2024)** "Automated phase identification in neutron powder diffraction with transformer networks," *J. Appl. Cryst.* — vision transformer treating diffraction patterns as 1D sequences achieves >95% phase identification accuracy on ICDD database.
3. **Korolev et al. (2025)** "Variational autoencoder for latent-space crystal structure retrieval from powder diffraction," *IUCr J.* — VAE-based representation learning embeds powder patterns in a continuous structural space enabling fast nearest-neighbor structure retrieval.
4. **Merz et al. (2024)** "Physics-constrained neural network refinement of magnetic structures from neutron diffraction," *Phys. Rev. Materials* — incorporated magnetic symmetry constraints into a neural network optimizer for combined nuclear/magnetic Rietveld refinement.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/neutron_diffraction_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/neutron_diffraction_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/neutron_diffraction_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/neutron_diffraction/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Neutron powder diffraction is correctly formulated as a structure-from-pattern inverse problem where the forward model (Rietveld) maps crystal structure parameters to a computed diffraction pattern, and the challenge is nonlinear parameter recovery from noisy histogram data. The algorithm routing from Rietveld refinement through Pawley extraction to deep-learning prediction appropriately spans classical crystallography to modern ML approaches. The mismatch parameters (peak width, background level, texture, counting statistics) are the primary experimental sources of refinement uncertainty in neutron diffraction.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 8.55 | 0.0116 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** Rietveld-GSAS
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 35.48 dB |
| SSIM (sample_00) | 0.8467 |
| Runtime | 0.29 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Le Bail Fit
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 37.08 dB |
| SSIM (sample_00) | 0.919 |
| Runtime | 0.52 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Rietveld-GSAS
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

---

## CPU Algorithm Test Results

**Algorithm:** Le Bail Fit
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 37.08 dB |
| SSIM (sample_00) | 0.919 |
| Runtime | 0.46 s/sample |

**Result: PASS**
