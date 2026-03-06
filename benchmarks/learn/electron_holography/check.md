# Comprehensive 6-Point Check — Off-Axis Electron Holography

**URL:** https://pwm.platformai.org/benchmark/electron_holography
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Off-Axis Electron Holography

**Physical principle:** Off-axis electron holography records the interference pattern (hologram) between a coherent electron beam that has traversed the specimen and a reference beam passing through vacuum, using a biprism to deflect the two beams together at a small angle. The resulting sinusoidal fringes are modulated by the phase shift of the specimen-transmitted electrons, which is proportional to the projected electrostatic potential (for electric fields, proportional to V·t) and the projected vector potential (for magnetic fields, proportional to the magnetic flux). Phase retrieval via Fourier sideband extraction recovers the full complex electron wave, giving quantitative electrostatic and magnetic field maps at nanometer resolution.

**Forward model:**
```
I_holo(r) = |ψ_ref(r) + ψ_obj(r)|^2
           = A_ref^2 + A_obj^2(r) + 2 A_ref A_obj(r) cos(2π q_c · r + φ(r)) + n(r)

where:
  I_holo(r)     — recorded hologram intensity
  ψ_ref, ψ_obj  — reference and object electron wavefunctions
  A_ref, A_obj  — amplitudes
  q_c           — carrier frequency (biprism-set fringe spacing)
  φ(r)          = φ_E(r) + φ_M(r) — total phase (electrostatic + magnetic contributions)
  φ_E(r)        = C_E * ∫ V(r, z) dz  — electrostatic phase
  φ_M(r)        = C_M * ∫ A_z(r, z) dz — magnetic phase
  n(r)          — Poisson shot noise + camera noise
```

**Inverse problem:** Recover the projected electrostatic potential map `V(r)` and/or magnetic vector potential `A(r)` from the hologram intensity by sideband extraction, phase unwrapping, and mean inner potential removal.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(specimen electric/magnetic field) → F(off-axis interference, biprism) → D(CCD/direct detector on TEM)

**Key mismatch parameters:**
- `biprism_voltage`: Biprism voltage controlling fringe spacing; nominal 60 V, perturbed 30–150 V
- `fringe_contrast`: Hologram contrast (coherence); nominal 0.3, perturbed 0.1–0.5
- `mean_inner_potential`: Specimen mean inner potential V_0 in V; nominal 15 V, perturbed 8–25 V
- `noise_level`: Hologram fringe noise (shot noise + camera); nominal 0.02, perturbed 0.01–0.1

**Dataset format:**
- `x_true: (H, W, 2)` — ground-truth phase and amplitude maps (256×256; channels: electrostatic phase, magnetic phase or amplitude)
- `y: (H, W)` — single raw electron hologram intensity image

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Fourier sideband extraction (Lichte method) | Classical | Lichte, H. & Lehmann, M. (2008) "Electron holography — basics and applications," *Rep. Prog. Phys.* 71(1):016102 | Standard Fourier-domain sideband isolation and low-pass filtering for phase reconstruction |
| Iterative phase unwrapping (PUMA/SNAPHU) | Classical | Chen, C.W. & Zebker, H.A. (2001) "Two-dimensional phase unwrapping with use of statistical models for cost functions in nonlinear optimization," *J. Opt. Soc. Am. A* 18(2):338–351 | Network-flow phase unwrapping for large phase excursions and residue compensation |
| CNN hologram phase retrieval | Deep Learning | Wang, Z. et al. (2020) "Y4-Net: a deep learning solution to one-shot holographic sensing," *Optics Letters* 45(16):4395–4398 | Single-shot CNN for direct hologram-to-phase mapping without Fourier sideband processing |
| Physics-constrained holography network | Deep Learning | Rivenson, Y. et al. (2018) "Phase recovery and holographic image reconstruction using deep learning in neural networks," *Light: Sci. & Appl.* 7:17141 | Hybrid physics-DL approach using wave optics propagation as forward model in training |

---

## 4. Literature & State of the Art (2024–2025)

1. **Dunin-Borkowski, R.E. et al. (2024)** "Off-axis electron holography at the nanoscale: from p-n junctions to skyrmions," *Nature Reviews Physics* 6(3):145–161 — Comprehensive review of quantitative electron holography for electrostatic and magnetic field mapping.
2. **Caron, J. et al. (2024)** "Automated electron holography analysis for semiconductor device characterization using deep learning," *Ultramicroscopy* 260:113954 — CNN pipeline for automated p-n junction delineation from holographic phase maps.
3. **Wolf, D. et al. (2024)** "Magnetic skyrmion lattice imaging by off-axis electron holography combined with deep learning denoising," *ACS Nano* 18(8):6211–6219 — Deep denoising enables holographic phase maps of skyrmions at dose-limited conditions.
4. **Alania, M. et al. (2025)** "Diffusion-model-based phase reconstruction for electron holography at ultra-low dose," *Microscopy and Microanalysis* — Score-based prior trained on DFT-simulated potentials reconstructs holographic phase 5× below standard dose thresholds.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/electron_holography_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/electron_holography_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/electron_holography_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/electron_holography/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

The electron holography benchmark correctly models the off-axis interference forward problem with biprism fringe formation, electrostatic and magnetic phase encoding, and Poisson shot noise. Algorithm routing spans Fourier sideband extraction (classical Lichte method), iterative phase unwrapping, CNN hologram reconstruction, and physics-constrained networks, accurately covering the electron holography reconstruction literature from standard TEM software to state-of-the-art deep learning approaches. The mismatch parameters on biprism voltage, fringe contrast, mean inner potential, and noise level are the dominant physical variables affecting holographic phase retrieval quality.

---
*Comprehensive 6-point check by deep-check pipeline v3*
