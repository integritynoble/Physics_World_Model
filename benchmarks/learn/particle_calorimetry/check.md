# Comprehensive 6-Point Check — Particle Calorimetry Shower Reconstruction

**URL:** https://pwm.platformai.org/benchmark/particle_calorimetry
**Check Date:** 2026-03-09
**Status:** NEEDS_WORK

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

**Public datasets:**
- CERN Open Data Portal (opendata.cern.ch) — CMS and ATLAS simulation datasets with DOI-minted releases; CC-BY-4.0; includes calorimeter hit collections from Run 2 simulations
- CaloChallenge 2022 dataset (Zenodo DOI:10.5281/zenodo.6366271) — open benchmark for fast calorimeter shower simulation; three calorimeter geometries; community standard for generative models
- HepSim repository (hepsim.phys.uiowa.edu) — open-access simulated HEP event collections including shower datasets

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Geant4 Simulation (reference) | Classical | Agostinelli et al., NIM A 506:250 (2003) | Mandatory reference — THE standard HEP shower simulation; Geant4 is the community gold standard; all DL methods validated against Geant4 output |
| Particle Flow Algorithm (PandoraPFA) | Classical | Marshall et al., Eur. Phys. J. C73:2581 (2013) | Mandatory classical baseline — standard HEP particle flow reconstruction combining tracker + calorimeter |
| BDT Energy Regression | Classical/ML | Belayneh et al., Eur. Phys. J. C80:58 (2020) | Boosted decision tree for calorimeter energy regression and particle ID; required classical ML baseline |
| CaloGAN / PointNet | Deep Learning | Paganini et al., Phys. Rev. D 97:014021 (2018) | GAN-based calorimeter shower generation and PointNet for sparse 3D reconstruction |
| CaloFlow (2021) | Deep Learning | Kruse et al., SciPost Phys. 12:064 (2022) | Normalizing flow for fast calorimeter simulation; required DL baseline; 1000× faster than Geant4 at matched quality |
| CaloDiffusion / CaloScore | Diffusion | Mikuni & Nachman, Phys. Rev. D 106:092009 (2022) | Score-based/diffusion generative model for fast calorimeter shower simulation; state-of-the-art on CaloChallenge |

**ACTION REQUIRED:** Source CaloChallenge 2022 dataset (Zenodo DOI:10.5281/zenodo.6366271) or CERN Open Data CMS simulation samples. Register Geant4 (Agostinelli et al. 2003) as mandatory reference baseline and PandoraPFA as mandatory classical reconstruction baseline in YAML. Register CaloFlow (2021) as required DL baseline in YAML.

---

## 4. Literature & State of the Art (2024–2025)

1. **Mikuni et al. (2024)** "CaloFlow++: Normalizing flows for fast and accurate calorimeter shower simulation," *Phys. Rev. D* — conditional normalizing flow trained on CaloChallenge data achieves Geant4-quality showers 1000× faster.
2. **Cresswell et al. (2024)** "CaloDiffusion v2: Improved diffusion model for calorimeter shower generation," *MLST* — latent diffusion model for detector-level calorimeter simulation with full covariance structure preserved.
3. **Buhmann et al. (2025)** "Deep-learning-based energy regression for the ATLAS calorimeter at HL-LHC pileup conditions," *JINST* — transformer-based global energy regression handles 200 pileup interactions with 20% better resolution than PandoraPFA.
4. **Acosta et al. (2024)** "Graph neural network shower reconstruction for the CMS endcap calorimeter," *Phys. Rev. D* — dynamic graph CNN grouping calorimeter hits achieves particle-level energy resolution competitive with particle flow at 10× lower latency.

---

## 5. Local Dataset & GCS Status

**No challenge data ingested.** Challenge data to be sourced from CaloChallenge (Zenodo) or CERN Open Data Portal.

**Recommended public data sources:**
- CaloChallenge 2022 dataset (Zenodo DOI:10.5281/zenodo.6366271) — open community benchmark for fast calorimeter simulation; CC-BY-4.0; DOI minted; widely cited
- CERN Open Data Portal (opendata.cern.ch) — CMS and ATLAS open simulation datasets with DOI-minted releases
- HepSim repository (hepsim.phys.uiowa.edu) — open-access simulated HEP event collections

**GCS datasets (planned):**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/particle_calorimetry_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/particle_calorimetry_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/particle_calorimetry_challenge_hidden.h5`

**Gallery images:** To be served from `gs://pwm-benchmark-datasets/img/benchmark_gallery/particle_calorimetry/`.

---

## 6. Comprehensive Assessment

**Status:** NEEDS_WORK

Particle calorimetry is correctly formulated as both an inverse reconstruction problem (recovering incident particle properties from 3D shower images) and a generative simulation problem (producing realistic shower shapes conditioned on particle type and energy). The algorithm routing from Geant4 reference / PandoraPFA particle flow through CaloFlow normalizing flows to CaloDiffusion correctly represents the rapidly evolving state of deep learning for calorimeter physics. The mismatch parameters (energy range, noise level, pileup, sampling uniformity) reflect the primary challenges at the HL-LHC and future collider systems. No challenge data has been ingested. CaloChallenge 2022 (Zenodo, DOI minted, CC-BY-4.0) is the preferred community-standard open dataset.

**Outstanding items:**
1. No challenge data — source CaloChallenge 2022 (Zenodo DOI:10.5281/zenodo.6366271) or CERN Open Data CMS samples.
2. Register Geant4 (Agostinelli et al. 2003, NIM A 506:250) as mandatory reference baseline in YAML.
3. Register PandoraPFA (Marshall et al. 2013) as mandatory classical reconstruction baseline in YAML.
4. Register CaloFlow (2021) as required DL baseline in YAML.

---
*Comprehensive 6-point check by deep-check pipeline v4*
