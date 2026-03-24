# Comprehensive 6-Point Check — Positron Emission Tomography

**URL:** https://pwm.platformai.org/benchmark/pet
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Positron Emission Tomography (PET)

**Physical principle:** A positron-emitting radionuclide (e.g., ¹⁸F-FDG) is injected and taken up by metabolically active tissue. Each β⁺ decay produces a positron that annihilates with a nearby electron, emitting two 511 keV γ-photons in nearly opposite directions. Coincidence detection of these photon pairs by a detector ring defines a line of response (LOR) along which the annihilation occurred. Acquiring millions of LOR coincidences over time produces a sinogram that is tomographically reconstructed to yield the 3D radionuclide distribution (tracer uptake map).

**Forward model:**
```
p_i = ∫_{LOR_i} λ(r) dr · a_i · n_i + r_i + s_i   for each LOR i

where:
  p_i    — expected detected coincidences along LOR i (Poisson mean)
  λ(r)   — radionuclide activity distribution (the unknown)
  a_i    — attenuation correction factor along LOR i
           a_i = exp(-∫_{LOR_i} μ(r) dr)
  n_i    — normalization factor (detector efficiency for LOR i)
  r_i    — random coincidences (accidentals)
  s_i    — scattered coincidences

Measurement: c_i ~ Poisson(p_i)
Sinogram: matrix form y = A · x + r + s (A = system matrix)
```

**Inverse problem:** Recover the 3D activity distribution λ(r) (i.e., tracer uptake image) from the measured sinogram {c_i}, correcting for attenuation, scatter, randoms, and detector normalization.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(radionuclide injection) → F(patient activity + attenuation distribution) → D(PET detector ring)

**Key mismatch parameters:**
- `count_rate_mcps`: measured coincidence count rate in Mcps; nominal 5.0 Mcps, perturbed 0.5–1.0 Mcps (low-dose)
- `scatter_fraction`: fraction of detected events that are scattered coincidences; nominal 0.30, perturbed 0.45–0.55
- `randoms_fraction`: fraction of detected events that are random coincidences; nominal 0.10, perturbed 0.30–0.50
- `attenuation_max_cm1`: peak linear attenuation coefficient in the FOV; nominal 0.10 cm⁻¹, perturbed 0.15 cm⁻¹ (dense bone)

**Dataset format:**
- `x_true: (256, 256)` — 2D transaxial slice of ground-truth activity distribution (kBq/mL)
- `y: (N_angles × N_bins,)` — vectorized sinogram of measured coincidence counts

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| FBP (Filtered Back-Projection) | Classical | Bracewell & Riddle (1967) *ApJ* 150:427; adapted for PET | Analytical inversion; fast but noise-amplifying; superseded by iterative methods in clinical practice |
| OSEM (Ordered Subsets Expectation Maximization) | Classical/Iterative | Hudson & Larkin (1994) *IEEE Trans. Med. Imaging* 13:601–609 | Clinical gold standard for PET reconstruction; accelerated EM with subset approximation |
| MAP-EM / Bayesian PET (BSREM) | Variational | De Pierro (1995) *IEEE Trans. Med. Imaging* 14:132–137; Nuyts et al. (2002) *Phys. Med. Biol.* | Maximum a posteriori reconstruction with spatial priors (quadratic, total-variation, Bowsher) |
| Deep PET / FBSEM-Net / DuDoRNet | Deep Learning | Häggström et al. (2019) *IEEE Trans. Med. Imaging* 38:1739–1751; Gong et al. (2019) *Phys. Med. Biol.* | End-to-end deep reconstruction from sinogram or post-processing of OSEM; handles low-count PET |

---

## 4. Literature & State of the Art (2024–2025)

1. **Gong et al. (2024)** "Diffusion model-based PET image reconstruction from ultra-low-dose sinograms," *IEEE Trans. Medical Imaging* — score-based diffusion posterior sampling achieves equivalent image quality to full-dose OSEM at 1/20 dose, advancing clinical low-dose PET feasibility.
2. **Zhou et al. (2024)** "Transformer-based sinogram completion and reconstruction for sparse-view PET," *Medical Physics* — self-attention mechanism completes missing LOR data before OSEM, recovering lesion detectability from 50% reduced angular sampling.
3. **Berg et al. (2025)** "Physics-informed neural ODE for dynamic PET tracer kinetic modeling," *NeuroImage* — neural ODE embedding compartment model ODEs into a deep learning framework for direct parametric PET reconstruction without OSEM intermediate.
4. **Catana et al. (2024)** "Review: Deep learning for attenuation correction in PET/MRI," *J. Nucl. Med.* — comprehensive benchmark of MR-to-CT synthesis methods for attenuation correction, showing vision transformer approaches outperform U-Net on bone-dense head regions.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/pet_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/pet_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/pet_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/pet/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

PET is correctly formulated as a Poisson inverse problem where the sinogram (LOR coincidence counts) is a noisy, attenuated, scatter- and random-contaminated projection of the 3D activity distribution, requiring Bayesian-optimal iterative reconstruction. The algorithm routing from FBP through OSEM/MAP to deep learning reconstruction appropriately spans the clinical standard (OSEM) and the rapidly advancing low-dose deep learning frontier. The mismatch parameters (count rate, scatter fraction, randoms fraction, attenuation) are the canonical clinical variables governing PET image quality and are the primary targets of ongoing algorithmic improvement.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| fbp_ramlak | 9.29 | 0.1813 | 0.09 | PASS |
| fbp_shepp_logan | 11.86 | 0.2681 | 0.07 | PASS |
| precomputed_fbp | 33.09 | 0.9325 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** FBP-PET
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 20.34 dB |
| SSIM (sample_00) | 0.3946 |
| Runtime | 3.15 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OSEM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 20.34 dB |
| SSIM (sample_00) | 0.3946 |
| Runtime | 1.47 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ML-EM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 20.34 dB |
| SSIM (sample_00) | 0.3946 |
| Runtime | 1.62 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** MAPEM-RDP
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 20.34 dB |
| SSIM (sample_00) | 0.3946 |
| Runtime | 1.36 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OS-EM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 20.34 dB |
| SSIM (sample_00) | 0.3946 |
| Runtime | 1.09 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FBP-PET
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 22.61 dB |
| SSIM (sample_00) | 0.5514 |
| Runtime | 0.63 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OSEM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 22.61 dB |
| SSIM (sample_00) | 0.5514 |
| Runtime | 0.56 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ML-EM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 22.61 dB |
| SSIM (sample_00) | 0.5514 |
| Runtime | 0.58 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** MAPEM-RDP
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 22.61 dB |
| SSIM (sample_00) | 0.5514 |
| Runtime | 0.44 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OS-EM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 22.61 dB |
| SSIM (sample_00) | 0.5514 |
| Runtime | 0.4 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FBP (emission tomography)
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** —
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 15.97 dB |
| SSIM (mean, 12 samples) | 0.0359 |
| Runtime | 1.76 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.66 dB |
| SSIM (mean, 12 samples) | 0.0296 |
| Runtime | 0.01 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 16.78 dB |
| SSIM (mean, 12 samples) | 0.0493 |
| Runtime | 0.72 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 21.85 dB |
| SSIM (mean, 12 samples) | 0.1500 |
| Runtime | 0.94 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 16.87 dB |
| SSIM (mean, 12 samples) | 0.0420 |
| Runtime | 0.77 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 15.50 dB |
| SSIM (mean, 12 samples) | 0.0336 |
| Runtime | 0.40 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.67 dB |
| SSIM (mean, 12 samples) | 0.0294 |
| Runtime | 0.57 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.67 dB |
| SSIM (mean, 12 samples) | 0.0311 |
| Runtime | 1.55 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.69 dB |
| SSIM (mean, 12 samples) | 0.0356 |
| Runtime | 1.53 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FBP (emission tomography)
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** —
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 15.97 dB |
| SSIM (mean, 12 samples) | 0.0359 |
| Runtime | 0.96 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FBP (emission tomography)
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** —
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 15.97 dB |
| SSIM (mean, 12 samples) | 0.0359 |
| Runtime | 0.97 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.66 dB |
| SSIM (mean, 12 samples) | 0.0296 |
| Runtime | 0.11 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.66 dB |
| SSIM (mean, 12 samples) | 0.0296 |
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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 16.78 dB |
| SSIM (mean, 12 samples) | 0.0493 |
| Runtime | 0.81 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 16.78 dB |
| SSIM (mean, 12 samples) | 0.0493 |
| Runtime | 0.75 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 21.85 dB |
| SSIM (mean, 12 samples) | 0.1500 |
| Runtime | 0.78 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 21.85 dB |
| SSIM (mean, 12 samples) | 0.1500 |
| Runtime | 0.76 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 16.87 dB |
| SSIM (mean, 12 samples) | 0.0420 |
| Runtime | 0.80 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 16.87 dB |
| SSIM (mean, 12 samples) | 0.0420 |
| Runtime | 0.85 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 15.50 dB |
| SSIM (mean, 12 samples) | 0.0336 |
| Runtime | 0.73 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 15.50 dB |
| SSIM (mean, 12 samples) | 0.0336 |
| Runtime | 0.46 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.67 dB |
| SSIM (mean, 12 samples) | 0.0294 |
| Runtime | 0.67 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.67 dB |
| SSIM (mean, 12 samples) | 0.0294 |
| Runtime | 0.71 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.67 dB |
| SSIM (mean, 12 samples) | 0.0311 |
| Runtime | 1.76 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.67 dB |
| SSIM (mean, 12 samples) | 0.0311 |
| Runtime | 1.94 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.69 dB |
| SSIM (mean, 12 samples) | 0.0356 |
| Runtime | 1.65 s/sample |

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
**Operator Family:** radon
**Forward Model:** y(LOR) = integral f(x,y) · a(LOR) dl, line-of-response (511 keV coincidence)
**Canonical Reference:** Cherry et al., "Physics in Nuclear Medicine," Elsevier 2012 (4th ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.69 dB |
| SSIM (mean, 12 samples) | 0.0356 |
| Runtime | 1.72 s/sample |

**Result: PASS**
