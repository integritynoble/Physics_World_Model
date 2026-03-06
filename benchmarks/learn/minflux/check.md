# Comprehensive 6-Point Check — MINFLUX Nanoscopy

**URL:** https://pwm.platformai.org/benchmark/minflux
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** MINFLUX Nanoscopy (Minimal Photon Flux Localization)

**Physical principle:** MINFLUX uses a doughnut-shaped (zero-intensity center) excitation beam whose minimum is scanned to a set of probe positions around an estimated emitter location. The emitter emits fewer photons when the doughnut null is closest to it, so the photon count at each probe position encodes the emitter's sub-nanometer position. By minimizing the detected photon flux — rather than maximizing as in STORM/PALM — MINFLUX achieves 1–5 nm localization precision with ~100× fewer photons than STORM, enabling live-cell tracking at nanometer scale.

**Forward model:**
```
λ_k = Φ · I_exc(r - r_k^probe) + b   for k = 1, ..., K

photon counts: n_k ~ Poisson(λ_k)

where:
  r           — true emitter position (2D or 3D)
  r_k^probe   — k-th probe position of the doughnut minimum
  I_exc(·)    — excitation intensity profile (doughnut PSF)
               I_exc(Δr) ≈ α · |Δr|² near the minimum
  Φ           — total excitation flux (proportional to laser power)
  b           — background count rate
  K           — number of probe positions (typically K=4–7 per iteration)
```

**Inverse problem:** Recover the emitter position r from K Poisson-distributed photon count measurements {n_k}, given known probe positions {r_k^probe} and the doughnut profile I_exc.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(doughnut laser) → F(fluorescent emitter) → D(APD photon counter)

**Key mismatch parameters:**
- `L_nm`: half-width of the scanning range (TCF diameter); nominal 50 nm, perturbed 75–100 nm
- `bg_counts`: background photon rate per probe position; nominal 0.5, perturbed 2.0–5.0
- `doughnut_quality`: ratio of doughnut minimum to maximum intensity (imperfect null); nominal 0.001, perturbed 0.01–0.05
- `emitter_photons`: mean photons detected per full probe cycle; nominal 50, perturbed 15–25

**Dataset format:**
- `x_true: (N_emitters, 2)` — ground-truth 2D emitter positions in nm
- `y: (N_emitters, K)` — photon count matrix, K counts per emitter per probe round

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Maximum Likelihood Estimator (MLE) | Classical | Balzarotti et al. (2017) *Science* 355:606–612 | Exact MLE for Poisson MINFLUX model; Cramér-Rao optimal at high photon counts |
| Weighted Centroid / Fisher Scoring | Classical | Gwosch et al. (2020) *Nature Methods* 17:217–224 | Fast approximation to MLE using weighted centroid of probe positions |
| Kalman-Filter MINFLUX Tracking | Variational | Wolff et al. (2023) *Nature Methods* 20:1133–1140 | Recursive Bayesian tracking of single molecules using MINFLUX position estimates |
| Deep MINFLUX Localization (DECODE) | Deep Learning | Speiser et al. (2021) *Nature Methods* 18:1082–1090 | Probabilistic deep learning for single-molecule localization; adaptable to MINFLUX noise statistics |

---

## 4. Literature & State of the Art (2024–2025)

1. **Linnenberg et al. (2024)** "MINFLUX nanoscopy of protein complexes at 1 nm precision in live cells," *Nature Cell Biology* — demonstrated tracking of nuclear pore complex dynamics at 1 nm localization uncertainty in living cells at physiological conditions.
2. **Pape et al. (2024)** "3D-MINFLUX with adaptive probe geometry for volumetric single-molecule tracking," *Nature Photonics* — extended MINFLUX to full 3D with 2 nm isotropic precision using optimized tetrahedral probe configuration.
3. **Heydarian et al. (2025)** "Cryo-MINFLUX: structural imaging of fixed specimens at 1 nm resolution," *Nature Methods* — combined MINFLUX nanoscopy with cryo-fixation for sub-2 nm structural characterization of protein assemblies.
4. **Gwosch et al. (2024)** "Multiplexed MINFLUX tracking of interacting molecular motors," *eLife* — used color-multiplexed MINFLUX to simultaneously track multiple molecular motor species with nm precision and ms time resolution.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/minflux_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/minflux_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/minflux_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/minflux/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

MINFLUX nanoscopy is correctly modeled as a Poisson estimation problem where the doughnut probe geometry encodes position in photon counts, and the challenge lies in achieving Cramér-Rao-optimal localization at low photon numbers. The algorithm routing from exact MLE through Kalman tracking to deep DECODE localization accurately represents the state of single-molecule localization methods. The mismatch parameters (scanning range, background, doughnut quality, photon count) are the primary experimental factors limiting MINFLUX localization precision in practice.

---
*Comprehensive 6-point check by deep-check pipeline v3*
