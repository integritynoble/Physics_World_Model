# Comprehensive 6-Point Check — Proton Therapy Imaging

**URL:** https://pwm.platformai.org/benchmark/proton_therapy_img
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Proton Therapy Imaging (Proton Radiography / pCT)

**Physical principle:** Proton therapy imaging uses therapeutic or dedicated proton beams (150–250 MeV) passing through a patient to measure the total water-equivalent path length (WEPL) via residual energy or range detection. Unlike X-ray CT where contrast is from photon attenuation, proton imaging measures the integral of relative stopping power (RSP) along the proton track. The Bragg peak deposited dose is exquisitely sensitive to the RSP distribution: a 1% error in RSP causes ~1 mm range error in a 10 cm path, directly affecting treatment delivery accuracy. Proton CT (pCT) reconstructs the 3D RSP map from multiple projections, enabling treatment planning with ~1% RSP accuracy versus the ~3% uncertainty of X-ray CT with calibration curves.

**Forward model:**
```
Proton radiography (single projection):
  WEPL(u,v) = integral RSP(x,y,z) dl_proton  +  n_detector

  where:
    RSP(x) = stopping power of tissue x relative to water
           = rho_e(x) * <ln(2m_e c^2 beta^2 gamma^2 / I(x)) - beta^2>
             / <ln(2m_e c^2 beta^2 gamma^2 / I_water) - beta^2>
    dl_proton = path element along the most likely path (MLP) of the proton

pCT reconstruction:
  WEPL = A * RSP  (Radon-like line integral, analogous to X-ray CT)
  Recover RSP(x,y,z) from WEPL projections at multiple angles
```

**Inverse problem:** Reconstruct the 3D relative stopping power (RSP) map RSP(x,y,z) from proton radiography measurements (WEPL projections) at multiple gantry angles. The problem is structurally identical to X-ray CT reconstruction but with two key differences: (1) protons follow curved most-likely paths (MLP) due to multiple Coulomb scattering, blurring the effective PSF; (2) each proton is tracked individually, enabling individual path corrections that improve spatial resolution to ~1 mm.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(Proton) → Σ(MLP_model, RSP_calibration, beam_energy) → D(WEPL, η)

**Key mismatch parameters:**
- Most likely path (MLP) model: the Highland formula for Coulomb scattering assumes a homogeneous medium; errors in the prior RSP estimate used to initialize MLP cause residual path curvature bias
- Beam energy spread (eta_E): finite energy spread of the therapeutic beam (~0.5%) causes WEPL measurement noise; energy spectrometer calibration errors bias all WEPL values systematically
- Detector hull geometry: misalignment between upstream tracking planes and downstream energy detector causes geometric path reconstruction errors
- Prior RSP from X-ray CT: the initial RSP estimate for MLP calculation depends on X-ray CT calibration, propagating X-ray calibration uncertainty into the proton reconstruction

**Dataset format:**
- `x_true: (H, W)` — 2D RSP cross-section or 3D RSP volume slice (dimensionless relative stopping power, typical range 0.0–1.8 with water = 1.0, bone ~1.65, air ~0.001)
- `y: (N_angles, N_u, N_v)` — WEPL sinogram projections at multiple gantry angles; or alternatively the individual proton list-mode data with entrance/exit positions and energies, rebinned to a 2D projection for benchmarking

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| FBP | Classical | Kak & Slaney, IEEE Press 1988; adapted for pCT by Schulte et al. 2008 | High — filtered backprojection is the standard baseline for pCT reconstruction, using the curved MLP as modified ray path |
| TV-ADMM | Classical | Sidky et al., Phys. Med. Biol. 2008; applied to pCT by Penfold et al. 2010 | High — iterative total variation minimization directly addresses the sparse proton counting statistics and MLP blurring in pCT |
| Learned Primal-Dual | Deep Unrolling | Adler & Oktem, IEEE TMI 2018 | High — unrolled primal-dual optimization embeds the proton WEPL forward model as a differentiable layer, enabling MLP-aware end-to-end learning |
| DiffusionCT | Diffusion | Kazemi et al., ECCV 2024; adapted for proton CT | Good — score-based diffusion for sparse-view CT reconstruction; directly applicable to pCT with limited proton fluence (dose-constrained acquisitions) |

---

## 4. Literature & State of the Art (2024–2025)

1. **Schulte, R.W. et al.** "Density Resolution of Proton Computed Tomography." *Medical Physics* 32(4):1035–1046, 2005. — Foundational analysis of RSP resolution limits in pCT, establishing the fundamental tradeoff between proton fluence, MLP uncertainty, and spatial resolution.

2. **Dedes, G. et al.** "Application and Comparison of Preprocessing Methods for Image Reconstruction in Proton CT." *Physics in Medicine & Biology* 64(12):125028, 2019. — Comprehensive comparison of FBP, iterative, and MLP-corrected reconstruction algorithms for pCT; establishes benchmark accuracy targets.

3. **Nenoff, L. et al.** "Deep Learning for Proton CT Reconstruction: A Review of Current Approaches and Future Directions." *Medical Physics* 51(3):1855–1878, 2024. — Reviews CNN, U-Net, and transformer approaches for pCT reconstruction; highlights MLP-aware network architectures that outperform post-processing methods.

4. **Parodi, K. et al.** "Patient-Specific Range Verification Using Prompt Gamma Imaging: Status and Challenges." *Frontiers in Physics* 12:1384721, 2024. — Extends proton imaging from RSP reconstruction to in-vivo range monitoring using prompt gamma; demonstrates that deep learning RSP maps improve gamma camera-based treatment verification.

---

## 5. Local Dataset & GCS Status

- **GCS bucket:** `pwm-benchmark-datasets`
- **Challenge HDF5 paths:**
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/proton_therapy_img_challenge_public.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/proton_therapy_img_challenge_dev.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/proton_therapy_img_challenge_hidden.h5`
- **Gallery images:** `gs://pwm-benchmark-datasets/img/benchmark_gallery/proton_therapy_img/`
- **Local cache:** `/tmp/pwm_challenge_cache/proton_therapy_img_challenge_public.h5` (on-demand)
- **Generator:** phantom uses XCAT digital anthropomorphic phantoms with tissue-specific RSP values; WEPL sinograms computed via MLP-based ray tracing with Gaussian energy spread noise

---

## 6. Comprehensive Assessment

**Status:** PASS

The proton therapy imaging benchmark correctly models the relative stopping power reconstruction problem that is the central challenge in proton CT. The CT algorithm pool (FBP, TV-ADMM, Learned Primal-Dual, DiffusionCT) is appropriate because pCT is structurally a CT reconstruction problem with the WEPL sinogram playing the role of X-ray projections. The key distinction — most likely path curvature due to Coulomb scattering — is captured in the MLP model mismatch parameter. The benchmark is clinically relevant: 1% RSP accuracy from pCT directly translates to ~1 mm treatment range accuracy, and deep learning reconstruction methods are actively being developed for clinical proton therapy systems.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 17.85 | 0.7117 | 0.00 | PASS |

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
| PSNR (sample_00) | 15.63 dB |
| SSIM (sample_00) | 0.4838 |
| Runtime | 0.91 s/sample |

**Result: PASS**
