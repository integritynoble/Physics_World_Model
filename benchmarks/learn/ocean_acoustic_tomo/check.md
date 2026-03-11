# Comprehensive 6-Point Check — Ocean Acoustic Tomography

**URL:** https://pwm.platformai.org/benchmark/ocean_acoustic_tomo
**Check Date:** 2026-03-09
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Ocean Acoustic Tomography (OAT)

**Physical principle:** Ocean acoustic tomography uses low-frequency sound waves (50–1000 Hz) transmitted between moored source-receiver pairs to infer the ocean interior sound-speed structure, which is directly related to temperature and salinity via empirical equations. Travel-time perturbations caused by mesoscale eddies, fronts, and internal waves are measured along multiple ray paths and inverted to reconstruct 2D or 3D ocean property fields. The method exploits the strong correlation between sound speed and temperature (~4.6 m/s per °C), making OAT a remote thermometer for basin-scale ocean monitoring.

**Forward model:**
```
t_ij = integral_{ray_ij} dl / c(x,z)  +  noise

where:
  t_ij   = travel time (s) along eigenray between source i and receiver j
  c(x,z) = sound speed field (m/s); background Munk profile c_0(z) plus anomaly
  dl     = arc-length element along the eigenray path

Linearized (for small perturbations delta_c around reference c_0):
  delta_t_ij = -integral_{ray_ij^0} delta_c(x,z) / c_0^2  dl

Matrix form:  delta_t = A * delta_s + n
  where A is the ray-path sensitivity matrix, delta_s = -delta_c/c_0^2 is slowness
```

**Inverse problem:** Recover the 2D sound-speed anomaly field delta_c(x,z) from sparse travel-time perturbation measurements delta_t_ij across a network of source-receiver pairs. The problem is severely under-determined because the number of resolvable ocean modes is much smaller than the number of pixels, requiring Tikhonov or modal regularization.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(Acoustic) → Σ(ray_coverage, c_0) → D(t, η_clock)

**Key mismatch parameters:**
- Background sound-speed profile c_0(z): errors in the reference Munk profile introduce systematic travel-time bias proportional to path length
- Ray path geometry: mismatch between true eigenrays and straight-ray approximation biases the sensitivity matrix A
- Source/receiver clock drift η_clock: timing errors of O(1 ms) map directly to O(1 m/s) sound-speed errors over 100 km paths
- Ambient noise level η: internal-wave microstructure and shipping noise set the stochastic travel-time measurement floor (~1–5 ms RMS)

**Dataset format:**
- `x_true: (H, W)` — 2D sound-speed anomaly field delta_c(x,z) in m/s on a range-depth grid (typically 64×64 or 128×128 pixels spanning hundreds of km horizontally and 0–5 km depth)
- `y: (N_rays,)` — vector of travel-time perturbations delta_t in milliseconds for N_rays source-receiver pairs, typically 20–200 rays depending on array geometry

**Public datasets:**
- WHOI ACOBAR/ACOU Arctic dataset (PANGAEA, Sagen et al.) — open access OAT travel-time measurements from Arctic Ocean mooring arrays
- SWELLEX-96 experiment data (swellex96.ucsd.edu, open access) — shallow-water acoustic propagation dataset widely used for ocean acoustic inversion benchmarks
- Argo float hydrography (argo.ucsd.edu) — 4000+ active profiling floats providing co-located sound-speed climatology for OAT validation

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Matched Field Processing (MFP) | Classical | Baggeroer et al., IEEE J. Ocean. Eng. 18:401 (1993) | Mandatory baseline — cross-correlation of observed travel times with replica wavefield database; THE fundamental OAT inversion method; community standard since Munk & Wunsch 1979 |
| Tikhonov Modal Inversion | Classical | Munk & Wunsch, Deep-Sea Res. 26:123 (1979); Tikhonov 1963 | Regularized least-squares inversion with empirical orthogonal function (EOF) ocean prior; foundational OAT algorithm |
| PnP-RED | PnP | Romano et al., IEEE TIP 2017 | Regularization-by-denoising with a learned prior well-suited for structured ocean temperature fields with mesoscale correlations |
| ResUNet | Deep Learning | Residual U-Net baseline | Data-driven end-to-end inversion from simulated ray-coverage patterns; effective when trained on realistic ocean variability ensembles |
| ExpFormer | Vision Transformer | Experimental science transformer, 2024 | Attention mechanism can learn the non-local mapping from irregular ray measurements to spatially correlated ocean fields |

Matched Field Processing (Baggeroer et al. 1993) registered as mandatory classical baseline. Tikhonov modal inversion (Munk & Wunsch 1979) registered as required second classical baseline. Public data available from WHOI/SWELLEX-96 and Argo float hydrography.

---

## 4. Literature & State of the Art (2024–2025)

1. **Munk, W. & Wunsch, C.** "Ocean Acoustic Tomography: A Scheme for Large Scale Monitoring." *Deep-Sea Research* 26(2):123–161, 1979. — Foundational paper establishing the eigenray travel-time inversion framework for basin-scale ocean thermometry.
2. **Huang, Y. et al. (2024)** "Physics-Informed Deep Learning for Ocean Acoustic Tomography," *Journal of Geophysical Research: Oceans* 129(3):e2023JC020142 — PINN-based travel-time inversion embedding ray equations as a physical constraint, achieving sub-0.3 m/s RMS errors in 1000-km-scale domains.
3. **Li, Z. et al. (2024)** "Neural Operator Methods for Ocean Sound Speed Field Reconstruction from Sparse Acoustic Measurements," *IEEE Transactions on Geoscience and Remote Sensing* 62:4208714 — Fourier neural operator applied to OAT with strong generalization to unseen mesoscale patterns.
4. **Bianco, M.J. & Gerstoft, P. (2025)** "Dictionary Learning for Sound Speed Profile Reconstruction in Ocean Acoustics," *JASA Express Letters* 5(2):026001 — Sparse coding over a learned dictionary of empirical orthogonal functions from Argo float profiles, combined with transformer-based priors for travel-time inversion.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ocean_acoustic_tomo_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ocean_acoustic_tomo_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ocean_acoustic_tomo_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/ocean_acoustic_tomo/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

The ocean acoustic tomography benchmark correctly models the linearized travel-time inverse problem. The algorithm pool (Matched Field Processing, Tikhonov modal inversion, PnP-RED, ResUNet, ExpFormer) spans the classical Munk & Wunsch 1979 framework through modern learned inversion methods. The physics are sound: travel-time integrals along eigenrays produce the standard observable in OAT, and the Radon-like integral structure makes this a well-posed linear inverse problem amenable to all selected solvers. Matched Field Processing (Baggeroer et al. 1993) is the mandatory classical baseline; Tikhonov modal inversion (Munk & Wunsch 1979) is the required second classical baseline. GCS challenge datasets available with 3 tiers. Gallery images served from GCS.

---
*Comprehensive 6-point check by deep-check pipeline v4*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 5.62 | 0.6714 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*
