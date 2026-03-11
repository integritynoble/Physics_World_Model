# Comprehensive 6-Point Check — Ground Penetrating Radar (GPR)

**URL:** https://pwm.platformai.org/benchmark/gpr
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Ground Penetrating Radar (GPR)

**Physical principle:** Ground Penetrating Radar transmits short electromagnetic pulses (typically 100 MHz–2.5 GHz) into the subsurface and records time-delayed reflections caused by contrasts in electrical permittivity ε and conductivity σ of buried objects, voids, pipes, or geological layers. The two-way travel time t = 2d/v (where v = c/√ε) encodes depth, and the reflection amplitude encodes dielectric contrast. A B-scan (profile) is acquired by moving the antenna along a survey line, producing a 2D image (distance vs. two-way time) with characteristic hyperbolic reflection patterns from point scatterers.

**Forward model:**
```
d(x, t) = Σ_j A_j · h(t − t_j(x)) + η

t_j(x) = (2/v) · √[(x − x_j)² + z_j²]   (hyperbola apex at (x_j, z_j))

where:
  d(x, t)     — B-scan amplitude at along-track position x, two-way time t
  A_j          — reflection coefficient of j-th scatter (dielectric contrast)
  h(t)         — transmitted pulse shape (Ricker wavelet)
  t_j(x)      — hyperbolic travel-time curve for scatterer j at position (x_j, z_j)
  v = c/√ε    — EM wave velocity in the medium
  η           — clutter, multi-path, and receiver noise
```

**Inverse problem:** Reconstruct the 2D permittivity/reflectivity cross-section σ(x,z) from the B-scan d(x,t), collapsing hyperbolic diffraction patterns to point scatterers and layer interfaces.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(Ricker pulse, 500 MHz–1 GHz) → F(soil/concrete subsurface) → D(dipole antenna receiver)

**Key mismatch parameters:**
- `center_frequency`: antenna center frequency; nominal 900 MHz, perturbed 500 MHz (deeper penetration, lower resolution)
- `soil_permittivity`: dielectric constant of background; nominal ε_r=9 (dry soil), perturbed ε_r=25 (wet clay, slower velocity)
- `clutter_level`: surface and subsurface clutter amplitude; nominal −30 dB, perturbed −15 dB (heavy clutter)
- `antenna_offset`: separation between transmit and receive antennas; nominal 0.12 m, perturbed 0.05 m (different geometry)

**Dataset format:**
- `x_true: (H, W)` — ground-truth 2D permittivity or reflectivity cross-section (depth × along-track)
- `y: (T, X)` — B-scan (T time samples × X trace positions)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Kirchhoff Migration | Classical | Yilmaz, "Seismic Data Analysis," SEG 2001 | Standard diffraction-stack migration collapsing hyperbolas to scatterer positions |
| FMCW back-projection | Classical | Daniels, "Ground Penetrating Radar," IET 2004 | Frequency-modulated CW back-projection for focused GPR images |
| RPCA clutter removal + TSVD | Classical | Levent et al., IEEE TGRS 52:5507 (2014) | Robust PCA for background removal prior to migration |
| GPRNet (deep learning) | Deep Learning | Giannopoulos et al., IEEE TGRS 59:1 (2021) | CNN trained on simulated B-scans for automatic hyperbola detection and classification |
| TransGPR (Transformer) | Transformer | Zhou et al., IEEE Trans. Geosci. Remote Sens. 61:1 (2023) | Transformer-based GPR inversion mapping B-scans to subsurface maps |

---

## 4. Literature & State of the Art (2024–2025)

1. **Rasol et al. (2024)** "Deep learning for GPR-based subsurface utility detection and localization," *Autom. Constr.* — review of CNN/transformer GPR methods for buried pipe detection across 12 real field surveys.
2. **Yang et al. (2024)** "Physics-informed neural networks for GPR full-waveform inversion," *IEEE Trans. Geosci. Remote Sens.* — PINNs enforcing Maxwell's equations as constraints for permittivity reconstruction from B-scans.
3. **Chen et al. (2024)** "Diffraction hyperbola fitting with transformer attention for GPR buried object detection," *IEEE Geosci. Remote Sens. Lett.* — attention-based localization of hyperbolic patterns outperforming template-matching methods.
4. **Giannakis et al. (2023)** "A machine learning scheme for estimating the diameter of reinforcing bars using ground penetrating radar," *NDT E Int.* — random forest regression on GPR features for rebar diameter estimation in concrete.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/gpr_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/gpr_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/gpr_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/gpr/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

GPR is correctly modeled as an electromagnetic wave scattering inverse problem with hyperbolic diffraction pattern reconstruction via migration, and the algorithm routing spans classical Kirchhoff migration, back-projection, RPCA-based clutter suppression, and modern deep learning methods. The mismatch parameters — center frequency, soil permittivity, clutter level, and antenna offset — accurately capture the primary sources of performance degradation in real GPR surveys across varying soil conditions and target depths. The benchmark is physically well-grounded and appropriate for evaluating robustness in near-surface geophysical imaging.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 10.60 | 0.0059 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*
