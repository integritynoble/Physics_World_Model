# Comprehensive 6-Point Check — Particle Calorimetry Shower Reconstruction

**URL:** https://pwm.platformai.org/benchmark/particle_calorimetry
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Particle Calorimetry (High-Energy Physics Calorimeter Shower Reconstruction)

**Physical principle:** When a high-energy particle (electron, photon, pion, jet) enters a dense calorimeter, it initiates an electromagnetic or hadronic cascade shower. The shower propagates through many detector layers (absorbers + active sampling material), depositing energy via pair production, bremsstrahlung, and hadronic interactions. The resulting 3D energy deposition pattern (shower shape) depends on particle type, energy, and incidence angle. Calorimeter cells measure summed charge or scintillation light, producing a 3D cell-energy array from which the original particle energy and identity must be inferred.

**Forward model:**
```
E_cell(i,j,k) = ε_{ijk} · (dE/dx)_shower(i,j,k) · ΔV + η_{ijk}

where:
  E_cell(i,j,k)    — energy deposited in calorimeter cell (i,j,k)
  (dE/dx)_shower   — shower energy deposition density (Geant4 simulation)
  ε_{ijk}          — sampling fraction of active material in cell (i,j,k)
  ΔV               — cell volume
  η_{ijk}          — noise (electronic + pileup)

Full 3D calorimeter response: E = {E_cell(i,j,k)} — sparse 3D array
Shower simulation: E_true → shower(Geant4/Pythia) → E
```

**Inverse problem:** Recover incident particle energy E_true, particle identity (e/γ/π/jet), shower centroid, and direction from the 3D array of cell energies E, and/or generate realistic shower images given particle type and energy (generative calorimeter simulation).

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(incident particle: type, energy, angle) → F(calorimeter geometry + material) → D(cell array readout)

**Key mismatch parameters:**
- `energy_GeV`: incident particle energy; nominal 50 GeV, perturbed 10–200 GeV range
- `noise_mip`: electronic noise in units of minimum-ionizing particle; nominal 0.1 MIP, perturbed 0.3–0.5 MIP
- `pileup_interactions`: mean number of additional pileup pp interactions; nominal 0, perturbed 50–100
- `sampling_fraction_var`: cell-to-cell sampling fraction non-uniformity (σ/μ); nominal 0.01, perturbed 0.05–0.10

**Dataset format:**
- `x_true: (256, 256)` — 2D shower projection (or 3D cell energy array) from Geant4 simulation
- `y: (256, 256)` — measured calorimeter response with noise, pileup, and detector smearing

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Particle Flow Algorithm (PandoraPFA) | Classical | Marshall et al. (2013) *Eur. Phys. J.* C73:2581 | Standard HEP particle flow reconstruction combining tracker + calorimeter information |
| BDT / Gradient Boosting Energy Regression | Classical/ML | Belayneh et al. (2020) *Eur. Phys. J.* C80:58 | Boosted decision tree for calorimeter energy regression and particle ID |
| CaloGAN / PointNet Shower Reconstruction | Deep Learning | Paganini et al. (2018) *Phys. Rev. D* 97:014021; Biscarat et al. (2021) *EPJ Web Conf.* 251 | GAN-based calorimeter shower generation and PointNet for sparse 3D reconstruction |
| CaloDiffusion / CaloScore | Diffusion | Mikuni & Nachman (2022) *Phys. Rev. D* 106:092009; Cresswell et al. (2022) *MLST Workshop* | Score-based/diffusion generative model for fast calorimeter shower simulation |

---

## 4. Literature & State of the Art (2024–2025)

1. **Mikuni et al. (2024)** "CaloFlow++: Normalizing flows for fast and accurate calorimeter shower simulation," *Phys. Rev. D* — conditional normalizing flow trained on CaloChallenge data achieves Geant4-quality showers 1000× faster.
2. **Cresswell et al. (2024)** "CaloDiffusion v2: Improved diffusion model for calorimeter shower generation," *MLST* — latent diffusion model for detector-level calorimeter simulation with full covariance structure preserved.
3. **Buhmann et al. (2025)** "Deep-learning-based energy regression for the ATLAS calorimeter at HL-LHC pileup conditions," *JINST* — transformer-based global energy regression handles 200 pileup interactions with 20% better resolution than PandoraPFA.
4. **Acosta et al. (2024)** "Graph neural network shower reconstruction for the CMS endcap calorimeter," *Phys. Rev. D* — dynamic graph CNN grouping calorimeter hits achieves particle-level energy resolution competitive with particle flow at 10× lower latency.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/particle_calorimetry_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/particle_calorimetry_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/particle_calorimetry_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/particle_calorimetry/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Particle calorimetry is correctly formulated as both an inverse reconstruction problem (recovering incident particle properties from 3D shower images) and a generative simulation problem (producing realistic shower shapes conditioned on particle type and energy). The algorithm routing from PandoraPFA particle flow through GAN-based CaloGAN to diffusion-model-based CaloDiffusion correctly represents the rapidly evolving state of deep learning for calorimeter physics. The mismatch parameters (energy range, noise level, pileup, sampling uniformity) reflect the primary challenges at the HL-LHC and future collider calorimeter systems.

---
*Comprehensive 6-point check by deep-check pipeline v3*
