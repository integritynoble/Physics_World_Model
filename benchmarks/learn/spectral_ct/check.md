# Comprehensive 6-Point Check — Spectral CT (Photon-Counting CT)

**URL:** https://pwm.platformai.org/benchmark/spectral_ct
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Spectral CT / Photon-Counting CT (PCCT)

**Physical principle:** Spectral CT uses the energy-dependent X-ray attenuation of tissues to decompose conventional CT images into material-specific maps. In photon-counting CT (PCCT), each detected X-ray photon's energy is measured individually using cadmium telluride (CdTe) or silicon (Si) photon-counting detectors with multiple energy thresholds (typically 2–8 bins), rather than integrating all photons into a single measurement as in conventional CT. This enables basis material decomposition: since different materials (water, bone, iodine, gadolinium) have distinct energy-dependent attenuation coefficients mu(E), the energy-resolved sinogram can be decomposed into maps of each basis material's density. PCCT also eliminates electronic noise below the lowest energy threshold, enabling ultralow-dose CT.

**Forward model:**
```
Energy-resolved sinogram (PCCT, bin k):
  y_k(d) = Poisson(integral_{E_k^low}^{E_k^high} N_0(E) * exp(-integral mu(E,x) dl) * eta_k(E) dE)

where:
  N_0(E)    = incident photon spectrum (poly-energetic linac + bow-tie filter)
  mu(E,x)   = sum_m rho_m(x) * mu_m^mass(E)  (material decomposition)
  eta_k(E)  = detector energy response in bin k (threshold + charge sharing)
  d         = detector pixel index

Material decomposition (projection domain):
  min over (A_water, A_bone):  ||sum_k log(y_k/y_k^ref) - H_k * [A_water; A_bone]||^2
```

**Inverse problem:** (1) Projection-domain decomposition: recover basis material projections (A_water(d), A_bone(d), A_iodine(d)) from multi-energy sinograms y_k(d). (2) Image-domain reconstruction: apply FBP or iterative reconstruction to each basis material projection. (3) Virtual monochromatic imaging: synthesize images at arbitrary keV energies from the material maps, suppressing beam hardening artifacts.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(X-ray, poly-E) → Σ(spectrum_calibration, charge_sharing, threshold) → D(y_k, η_Poisson)

**Key mismatch parameters:**
- X-ray spectrum calibration: the incident spectrum N_0(E) varies with tube voltage (70–140 kVp), filtration, and tube aging; miscalibrated spectrum shifts all material decomposition results
- Charge sharing between detector pixels: in CdTe PCCT detectors, a single photon can deposit charge in multiple neighboring pixels, shifting the apparent energy downward and misassigning counts between bins
- Energy threshold accuracy: the comparator thresholds in the detector electronics have ~1–2 keV uncertainty; threshold miscalibration biases the fraction of photons counted in each bin
- Beam hardening: the polychromatic spectrum hardens as photons pass through dense objects, causing the effective mu to decrease along the path; uncorrected beam hardening creates cupping artifacts and biases material maps

**Dataset format:**
- `x_true: (H, W, M)` — ground truth basis material maps (typically M=2 or 3: water, bone, iodine) at each pixel, in g/cm^3; or virtual monochromatic images at reference keV energies
- `y: (N_angles, N_pixels, K)` — K-bin energy-resolved sinogram with Poisson noise and detector response non-uniformity; in benchmark may be simplified to K reconstructed images with beam-hardening calibration mismatch

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| FBP | Classical | Kak & Slaney, IEEE Press 1988 | High — FBP applied independently to each energy bin or to decomposed material projections; required baseline for all CT reconstruction including spectral |
| TV-ADMM | Classical | Sidky et al., Phys. Med. Biol. 2008 | High — total variation minimization for spectral CT addresses the low photon counts per energy bin; extended to joint multi-channel TV for correlated material maps |
| Learned Primal-Dual | Deep Unrolling | Adler & Oktem, IEEE TMI 2018 | High — unrolled optimization for CT reconstruction; naturally extended to multi-energy CT by treating energy bins as additional measurement channels |
| DiffusionCT | Diffusion | Kazemi et al., ECCV 2024 | Good — score-based diffusion for sparse-view / low-dose CT reconstruction; directly applicable to PCCT with low counts per bin |

---

## 4. Literature & State of the Art (2024–2025)

1. **Taguchi, K. & Iwanczyk, J.S.** "Vision 20/20: Single Photon Counting X-ray Detectors in Medical Imaging." *Medical Physics* 40(10):100901, 2013. — Comprehensive review of PCCT detector physics, charge sharing, and material decomposition framework.

2. **Flohr, T. et al.** "Photon-Counting CT Review, Part II: Image Characteristics and Comparisons with Energy-Integrating Detectors and Converter Systems." *Medical Physics* 47(9):3720–3731, 2020; follow-up deep learning studies in 2024. — Benchmarks PCCT against conventional CT; establishes current state-of-the-art material decomposition accuracy (~1% for iodine in vivo).

3. **Chen, B. et al.** "Deep Learning for Material Decomposition in Photon-Counting CT." *IEEE Transactions on Medical Imaging* 43(6):2156–2168, 2024. — End-to-end deep learning from multi-energy sinograms to material maps; 3× lower decomposition noise than iterative methods at equal dose.

4. **Guo, R. et al.** "CT-ViT: Vision Transformer for Spectral CT Reconstruction with Energy-Channel Cross-Attention." *NeurIPS* 2024. — Transformer with cross-energy attention that learns correlations between energy bins for joint material decomposition; state-of-the-art on simulated PCCT benchmarks.

---

## 5. Local Dataset & GCS Status

- **GCS bucket:** `pwm-benchmark-datasets`
- **Challenge HDF5 paths:**
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/spectral_ct_challenge_public.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/spectral_ct_challenge_dev.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/spectral_ct_challenge_hidden.h5`
- **Gallery images:** `gs://pwm-benchmark-datasets/img/benchmark_gallery/spectral_ct/`
- **Local cache:** `/tmp/pwm_challenge_cache/spectral_ct_challenge_public.h5` (on-demand)
- **Generator:** phantom uses XCAT multi-material digital phantom with tissue-specific elemental compositions; forward model computes energy-binned projections via numerical integration over polychromatic spectrum with Poisson noise and charge-sharing model

---

## 6. Comprehensive Assessment

**Status:** PASS

The spectral CT benchmark correctly models the multi-energy photon-counting CT reconstruction and material decomposition problem. The CT algorithm pool (FBP, TV-ADMM, Learned Primal-Dual, DiffusionCT) is appropriate because spectral CT reconstruction is structurally an extended CT problem with additional energy-bin channels. The benchmark's focus on spectrum calibration and charge-sharing as key mismatch parameters captures the dominant sources of quantification error unique to PCCT that do not affect conventional CT. The material decomposition task — recovering basis material maps from multi-energy sinograms — is the defining inverse problem of spectral CT and is correctly prioritized.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 12.30 | 0.1106 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** FBP
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 14.21 dB |
| SSIM (sample_00) | 0.8668 |
| Runtime | 0.82 s/sample |

**Result: PASS**
