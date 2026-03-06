# Comprehensive 6-Point Check — Photoacoustic Tomography

**URL:** https://pwm.platformai.org/benchmark/photoacoustic
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Photoacoustic Tomography

**Physical principle:** Photoacoustic tomography (PAT) combines the high contrast of optical absorption with the low scattering of ultrasound. A short (nanosecond) laser pulse illuminates biological tissue; chromophores (hemoglobin, melanin) absorb the light and undergo rapid thermoelastic expansion, launching broadband acoustic pressure waves. An array of ultrasonic transducers surrounding the tissue records these waves as time-domain pressure signals, which are then reconstructed into an initial pressure distribution map proportional to the optical absorption coefficient times the local fluence.

**Forward model:**
```
p(r_d, t) = ∫ h(r_d, t; r') · p_0(r') dr' + n

where:
  p(r_d, t)  — pressure recorded at detector r_d at time t
  p_0(r')    — initial pressure distribution: p_0 = Γ · μ_a(r') · Φ(r')
  h(·)       — Green's function of the acoustic wave equation (spherical wave kernel)
  Γ          — Grüneisen parameter (dimensionless thermo-acoustic efficiency)
  μ_a(r')    — optical absorption coefficient at r'
  Φ(r')      — local optical fluence
  n          — transducer electronic noise
```

**Inverse problem:** Recover the initial pressure distribution p_0(r) (and hence the optical absorption map μ_a(r)) from time-domain acoustic signals p(r_d, t) recorded on a detector aperture; requires solving the time-reversal of the acoustic wave equation.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(pulsed laser, 532–1064 nm) → F(acoustic wave propagation in tissue) → D(ultrasonic transducer array)

**Key mismatch parameters:**
- `speed_of_sound`: acoustic propagation speed c; nominal 1540 m/s (water), perturbed ±3% (soft tissue heterogeneity)
- `transducer_bandwidth`: detector frequency response; nominal 1–15 MHz, perturbed to limited-view (180° aperture)
- `grueneisen_parameter`: Γ; nominal 0.2, perturbed ±15% (temperature/tissue variation)
- `optical_fluence_correction`: fluence heterogeneity Φ(r); nominal uniform, perturbed to depth-dependent exponential decay

**Dataset format:**
- `x_true: (H, W)` — 2D initial pressure map p_0 in arbitrary units, representing the cross-sectional optical absorption × fluence
- `y: (N_det, N_t)` — time-domain pressure signals from N_det transducers over N_t time samples

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Delay-and-Sum (DAS) back-projection | Classical | Xu & Wang, Physical Review E 71, 016706 (2005) | Standard filtered back-projection adapted for PAT; analytic and fast |
| Universal Back-Projection (UBP) | Classical | Xu & Wang, IEEE Trans. Medical Imaging 24, 1208–1221 (2005) | Exact inversion formula for spherical, cylindrical, and planar detector geometries |
| Time-reversal (k-Wave) | Simulation-based | Treeby & Cox, J. Biomedical Optics 15, 021314 (2010) | Time-reversal reconstruction via k-space pseudo-spectral wave solver |
| Model-based TV regularization | Optimization | Arridge et al., Inverse Problems 32, 115012 (2016) | Iterative model-based reconstruction with total variation prior; handles limited-view |
| PAT-Net (U-Net) | Deep Learning | Antholzer et al., Photoacoustics 14, 1–9 (2019) | CNN artifact removal applied to DAS initial reconstructions for PAT |
| Score-based PAT reconstruction | Diffusion | Song et al., IEEE Trans. Medical Imaging 42, 1750 (2023) | Diffusion posterior sampling for limited-data PAT reconstruction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Hauptmann et al. (2024)** "Deep learning in photoacoustic tomography: current approaches and future directions," *Journal of Biomedical Optics* — comprehensive review of learned reconstruction covering supervised, unsupervised, and physics-informed methods.
2. **DiSpirito et al. (2024)** "Reconstructing undersampled photoacoustic data using an implicit neural representation network," *IEEE Trans. Medical Imaging* — implicit neural representation (NeRF-style) trained on single volumes; 4× undersampling recovery.
3. **Gröhl et al. (2025)** "Foundation models for photoacoustic image reconstruction," *Nature Biomedical Engineering* — large-scale pretrained transformer fine-tuned on PAT; generalizes across geometries.
4. **Vu et al. (2024)** "3D photoacoustic reconstruction from sparse ring arrays using score-based diffusion," *Medical Image Analysis* — extends diffusion priors to 3D cylindrical scanner geometry with sparse transducers.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/photoacoustic_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/photoacoustic_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/photoacoustic_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/photoacoustic/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Photoacoustic tomography has a clearly defined forward model (Green's function acoustic propagation from optical absorption sources) with well-understood inversion strategies. Algorithm routing spans the foundational DAS/UBP back-projection methods, k-Wave time-reversal simulation, TV-regularized iterative reconstruction, and modern deep learning approaches. The four mismatch parameters capture the dominant sources of model error in PAT experiments (speed of sound heterogeneity, limited aperture, Grüneisen parameter uncertainty, and fluence non-uniformity).

---
*Comprehensive 6-point check by deep-check pipeline v3*
