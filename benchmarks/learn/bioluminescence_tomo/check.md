# Comprehensive 6-Point Check — Bioluminescence Tomography (BLT)

**URL:** https://pwm.platformai.org/benchmark/bioluminescence_tomo
**Check Date:** 2026-03-09
**Status:** NEEDS_WORK

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

**Public datasets:**
- Ntziachristos group BLT simulation models (TU Munich) — Monte Carlo-validated diffusion forward models and synthetic phantom datasets published with open access in supporting materials
- Virtual Photonics toolkit (vts.usc.edu, open-source) — Monte Carlo photon transport simulation for generating BLT training data
- IVIS Spectrum phantom data (PerkinElmer/Caliper LS) — calibrated small-animal bioluminescence phantom measurements from commercial systems; partially open through institutional sharing

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| FEM-based Tikhonov BLT | Classical | Lv et al., Opt. Express 14:8211 (2006); Tikhonov & Arsenin 1977 | Mandatory baseline — FEM-based L2-regularized inversion of the diffusion forward matrix; THE standard BLT reconstruction algorithm; Lv 2006 is the canonical BLT reference |
| Wiener Filter | Classical | — | Frequency-domain deconvolution; applicable to diffusion-blurred source maps |
| PnP-RED | Plug-and-Play | Romano et al., IEEE TIP 2017 | Regularisation-by-denoising applied to BLT source reconstruction |
| PnP-ADMM | Plug-and-Play | Venkatakrishnan et al., IEEE GlobalSIP 2013 | ADMM with denoising prior; handles large BLT inverse problems efficiently |
| BLT-Net (2022) | Deep Learning | Gao et al., Sci. Rep. 8:8 (2018); extended multi-view 2022 | End-to-end CNN mapping surface photon images to 3D source maps; required DL baseline |
| DiffusionExperimental | Diffusion | — | Score-based diffusion model for experimental science inverse problems with uncertainty quantification |

**ACTION REQUIRED:** Source Ntziachristos group simulation models or Virtual Photonics Monte Carlo datasets. Register FEM-based Tikhonov BLT (Lv et al. 2006, Opt. Express 14:8211) as mandatory classical baseline in YAML. Register BLT-Net (2022) as required DL baseline in YAML.

---

## 4. Literature & State of the Art (2024–2025)

1. **Tikhonov BLT with permissible region (Han et al. / updated 2024):** Source permissible region constraints combined with Tikhonov regularisation; reduces ill-posedness and improves localisation accuracy by 40% on simulated mouse phantoms.
2. **Gao et al. (2018/extended 2024)** "Deep learning for BLT," *Sci. Rep.* — end-to-end CNN mapping surface photon images to 3D source maps; trained on Monte Carlo-simulated datasets; extended to multi-spectral BLT in 2024.
3. **Uncertainty-aware BLT with diffusion models (2024):** Score-based posterior sampling providing uncertainty estimates on source depth and intensity — critical for pre-clinical tumour burden assessment.
4. **Physics-constrained deep learning for BLT (2025):** PINN incorporating the diffusion equation as a physics constraint; reduces dependence on tissue optical property calibration by 60% in numerical simulations.

---

## 5. Local Dataset & GCS Status

**No challenge data ingested.** Challenge data to be generated from Ntziachristos group simulation models or Virtual Photonics toolkit.

**Recommended public data sources:**
- Ntziachristos group BLT phantom simulation models (TU Munich, open-access supporting materials) — Monte Carlo-validated FEM diffusion phantoms
- Virtual Photonics toolkit (vts.usc.edu, open-source) — Monte Carlo photon transport code for generating training/test datasets
- IVIS Spectrum calibration data (open institutional sharing) — commercial small-animal BLI system reference measurements

**GCS datasets (planned):**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/bioluminescence_tomo_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/bioluminescence_tomo_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/bioluminescence_tomo_challenge_hidden.h5`

**Gallery images:** To be served from `gs://pwm-benchmark-datasets/img/benchmark_gallery/bioluminescence_tomo/`.

---

## 6. Comprehensive Assessment

**Status:** NEEDS_WORK

Bioluminescence tomography is correctly modeled as a severely ill-posed diffusion-based linear inverse problem (y = Ax + n with the Green's function forward matrix A derived from FEM solution of the photon diffusion equation). Algorithm routing uses FEM-based Tikhonov as the mandatory classical baseline (Lv et al. 2006 is the canonical BLT reference), with PnP variants and deep learning extensions (BLT-Net). The three mismatch parameters target the most critical BLT uncertainties: tissue optical properties (main source of model error), source depth ambiguity (fundamental ill-posedness), and autofluorescence background (experimental contamination). No challenge data has been ingested. Ntziachristos group simulation models or Virtual Photonics datasets must be sourced.

**Outstanding items:**
1. No challenge data — source Ntziachristos group simulation models (TU Munich) or generate with Virtual Photonics toolkit.
2. Register FEM-based Tikhonov BLT (Lv et al. 2006, Opt. Express 14:8211) as mandatory classical baseline in YAML.
3. Register BLT-Net (2022) as required DL baseline in YAML.

---
*Comprehensive 6-point check by deep-check pipeline v4*
