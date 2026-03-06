# Comprehensive 6-Point Check — Bioluminescence Tomography (BLT)

**URL:** https://pwm.platformai.org/benchmark/bioluminescence_tomo
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Bioluminescence Tomography (BLT)

**Physical principle:** Bioluminescence tomography reconstructs the 3D distribution of bioluminescent sources (e.g., luciferase-expressing tumour cells) inside a living small animal from photon flux measurements on the body surface. Light transport in tissue is highly scattering and governed by the radiative transfer equation; in the diffusion approximation, photon propagation is characterised by absorption coefficient μ_a and reduced scattering coefficient μ_s'. The inverse problem is severely ill-posed because the surface measurement contains very limited angle-resolved information about deep sources.

**Forward model:**
```
Diffusion equation (steady-state):
  -∇·[D(r)∇Φ(r)] + μ_a(r) Φ(r) = S(r)    [in Ω]
  Φ(r) + 2A D(r) ∂_n Φ(r) = 0              [Robin BC on ∂Ω]

where:
  D(r) = 1/(3(μ_a + μ_s'))   — diffusion coefficient
  Φ(r)                        — photon fluence rate (W/cm²)
  S(r)                        — bioluminescent source distribution (W/cm³)
  A                           — boundary mismatch coefficient

Discrete forward model:
  y = A x + n
  y ∈ R^{N_surf}              — surface photon flux measurements
  x ∈ R^{N_vox}              — volumetric source distribution
  A ∈ R^{N_surf × N_vox}     — Green's function matrix from FEM solution
  n                           — Poisson + Gaussian detector noise
```

**Inverse problem:** Recover the 3D bioluminescent source distribution x from surface photon flux measurements y, given uncertain tissue optical properties (μ_a, μ_s') that are the primary source of model mismatch.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** Src(bioluminescent) → R(rotation views) → P(photon diffusion) → D(CCD camera)

**Key mismatch parameters:**
- `optical_property_error` (o_p): relative error in μ_a and μ_s' estimates; nominal 0.0%, perturbed 4.0%
- `source_depth_ambiguity` (s_d): uncertainty in depth of source reconstruction; nominal 0.0 mm, perturbed 1.0 mm
- `autofluorescence_background` (a_b): background autofluorescence signal level; nominal 0.0, perturbed 6.0 (relative)

**Dataset format:**
- `x_true: (H, W)` — 2D projection of bioluminescent source distribution (ground truth)
- `y: (N_views, H, W)` — multi-view surface photon flux images (rotational acquisition)
- `H_ideal: (N_views*H*W, H*W)` — linearised FEM-based diffusion forward operator

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Tikhonov | Classical | Tikhonov & Arsenin 1977; applied to BLT: Lv et al., PMB 2006 | L2-regularised inversion of the diffusion forward matrix; standard BLT baseline |
| Wiener Filter | Classical | — | Frequency-domain deconvolution; applicable to diffusion-blurred source maps |
| PnP-RED | Plug-and-Play | Romano et al., IEEE TIP 2017 | Regularisation-by-denoising applied to BLT source reconstruction |
| PnP-ADMM | Plug-and-Play | Venkatakrishnan et al., IEEE GlobalSIP 2013 | ADMM with denoising prior; handles large BLT inverse problems efficiently |
| ResUNet | Deep Learning | — | Residual U-Net for source localisation from surface measurement images |
| DiffusionExperimental | Diffusion | — | Score-based diffusion model for experimental science inverse problems |

---

## 4. Literature & State of the Art (2024–2025)

1. **Tikhonov BLT with permissible region** (Han et al., Opt. Express 2006 / updated 2024): Source permissible region constraints combined with Tikhonov regularisation; reduces ill-posedness and improves localisation accuracy by 40%.
2. **Deep learning for BLT** (Gao et al., Sci. Rep. 2018 / extended 2024): End-to-end CNN mapping surface photon images to 3D source maps; trained on Monte Carlo-simulated datasets.
3. **Uncertainty-aware BLT with diffusion models** (2024): Score-based posterior sampling providing uncertainty estimates on source depth and intensity — critical for pre-clinical tumour burden assessment.
4. **Physics-constrained deep learning for BLT** (2025): PINN incorporating the diffusion equation as a physics constraint; reduces dependence on tissue optical property calibration.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/bioluminescence_tomo_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/bioluminescence_tomo_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/bioluminescence_tomo_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/bioluminescence_tomo/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing uses the `experimental_science` category pool (11 methods: Tikhonov, Wiener Filter, Matched Filter, PnP-RED, PnP-ADMM, ResUNet, Domain-Adapted-CNN, SwinIR, ExpFormer, DiffusionExperimental, ScoreExperimental). Tikhonov is the standard BLT reconstruction baseline (Lv et al., 2006 is the canonical reference). The three mismatch parameters target the most critical BLT uncertainties: tissue optical properties (main source of model error), source depth ambiguity (fundamental ill-posedness), and autofluorescence background (experimental contamination). Note that SwinIR is a 2D image restoration transformer in a 3D volumetric domain — acceptable for 2D projection benchmarks but noted as a domain mismatch for full 3D BLT.

---
*Comprehensive 6-point check by deep-check pipeline v3*
