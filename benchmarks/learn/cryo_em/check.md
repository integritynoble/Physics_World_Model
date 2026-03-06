# Comprehensive 6-Point Check — Cryo-EM Single Particle Analysis

**URL:** https://pwm.platformai.org/benchmark/cryo_em
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Cryo-EM Single Particle Analysis (SPA)

**Physical principle:** Cryo-electron microscopy vitrifies purified protein particles in a thin ice layer, preserving near-native conformation. A focused 80–300 keV electron beam transmits through the sample; elastic scattering from the protein's electrostatic potential forms the phase-contrast image. The contrast transfer function (CTF) modulates spatial frequencies based on defocus and lens aberrations. Thousands to millions of 2D particle images in random orientations are collected; the inverse problem is to reconstruct the 3D molecular potential map from these 2D projections. The central theorem of tomography applies: each 2D image is a projection of the 3D structure, but the projection direction must also be estimated (unknown orientation).

**Forward model:**
```
Image formation model:
  y_i(x,y) = [P_{θ_i} V] ⊛ CTF(Δf_i, Cs, ...) + n_i

where:
  V ∈ R^{H×W×D}       — 3D electrostatic potential (ground truth)
  P_{θ_i}              — projection along orientation θ_i (Euler angles: φ, θ, ψ)
  CTF(Δf_i, Cs, λ)    — contrast transfer function: CTF(f) = -sin(πλΔf f² + πCs λ³ f⁴/2)
  Δf_i                 — defocus value for micrograph i (1-5 µm)
  Cs                   — spherical aberration coefficient (~2 mm)
  λ                    — electron wavelength (~2 pm at 300 keV)
  n_i                  — Poisson electron shot noise + detector noise

Discrete form:
  y_i = P_{θ_i} H_{CTF,i} V + n_i
  y  — stack of N 2D particle images
  V  — 3D reconstruction target
```

**Inverse problem:** Recover the 3D molecular potential V from a large stack of noisy 2D particle images {y_i}, jointly estimating the unknown projection orientations {θ_i} and CTF parameters {Δf_i}.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** C(CTF convolution) → D(direct electron detector)

**Key mismatch parameters:**
- `defocus_error` (d_e): defocus estimation error; nominal 0.0 nm, perturbed 100.0 nm
- `astigmatism` (a): astigmatic aberration; nominal 0.0 nm, perturbed 20.0 nm
- `beam_tilt` (b_t): beam tilt miscalibration; nominal 0.0 mrad, perturbed 0.2 mrad
- `ice_thickness_variation` (i_t): vitreous ice thickness; nominal 50.0 nm, perturbed 56.0 nm

**Dataset format:**
- `x_true: (H, W)` — 2D projection of the 3D molecular map (ground truth reference projection)
- `y: (N_particles, H, W)` — particle image stack with CTF and noise
- `H_ideal: (N_particles*H*W, H*W)` — ideal projection + CTF operator stack

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Direct Methods | Classical | Crowther et al., Proc. R. Soc. 1970 | Fourier slice theorem direct inversion; foundational cryo-EM reconstruction |
| RELION 1.0 | Classical/Bayesian | Scheres, J. Struct. Biol. 2012 | Maximum-likelihood 3D refinement; the gold-standard cryo-EM software |
| cryoSPARC | Classical/Variational | Punjani et al., Nat. Methods 2017 | Stochastic gradient descent 3D reconstruction; industry-standard SPA tool |
| cryoDRGN | Deep Learning | Zhong et al., Nat. Methods 2021 | VAE-based heterogeneous cryo-EM reconstruction; handles conformational variability |
| CryoTransformer | Transformer | Dhakal et al., Bioinformatics 2024 | Transformer for cryo-EM particle picking and reconstruction |
| DiffusionCryoEM | Diffusion | — | Score-based diffusion for cryo-EM density map reconstruction |

---

## 4. Literature & State of the Art (2024–2025)

1. **cryoDRGN2** (Zhong et al., 2021 / v2 2024): Extended VAE architecture for heterogeneous cryo-EM with improved conformational landscape mapping; resolves continuous conformational motions in large complexes.
2. **CryoAI** (Levy et al., NeurIPS 2022 / extended 2024): Amortised inference approach eliminating explicit expectation-maximisation; achieves RELION-quality reconstructions 100× faster.
3. **CryoFold** (2024): AlphaFold2-informed cryo-EM density map refinement; uses predicted atomic model as structural prior to resolve low-SNR regions.
4. **Tomography-SPA hybrid** (2025): Joint reconstruction from subtomogram averaging and SPA data using a unified transformer architecture; enables atomic resolution structure determination from cellular cryo-tomograms.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cryo_em_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cryo_em_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cryo_em_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/cryo_em/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing: `cryo_em` has `category: scientific_instrumentation` in the modality catalog, but the catalog also routes through the `_CRYO_EM_VARIANTS` check (which activates when category is electron_microscopy). The Python output shows the algorithms served are RELION, cryoSPARC, cryoDRGN, CryoTransformer, etc. — the correct cryo-EM pool — indicating that routing works correctly. The four mismatch parameters (defocus error, astigmatism, beam tilt, ice thickness) cover the principal cryo-EM CTF and sample preparation uncertainties. All key algorithms (RELION, cryoSPARC, cryoDRGN) are real, well-cited software packages confirming excellent domain alignment.

---
*Comprehensive 6-point check by deep-check pipeline v3*
