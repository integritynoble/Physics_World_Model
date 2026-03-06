# Comprehensive 6-Point Check — InSAR (Interferometric Synthetic Aperture Radar)

**URL:** https://pwm.platformai.org/benchmark/insar
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Interferometric Synthetic Aperture Radar (InSAR)

**Physical principle:** InSAR measures surface deformation and topography by comparing the phase of two SAR images acquired from slightly different satellite positions (or times). The interferometric phase φ = 4π·Δr/λ is proportional to the range change Δr (displacement toward/away from the satellite) at centimeter to millimeter precision for displacement measurements. The wrapped phase (−π to π) must be unwrapped to recover the continuous deformation field. InSAR enables monitoring of earthquake deformation, volcanic inflation, glacier flow, subsidence from groundwater/oil extraction, and permafrost thaw.

**Forward model:**
```
φ_int(x,y) = φ_topo(x,y) + φ_defo(x,y) + φ_atm(x,y) + φ_noise(x,y)

φ_defo = (4π/λ) · d_LOS(x,y)

where:
  φ_int(x,y)   — observed wrapped interferometric phase at pixel (x,y) [−π, π]
  φ_topo(x,y)  — topographic phase (DEM contribution, from reference DEM)
  φ_defo(x,y)  — surface deformation phase (line-of-sight displacement d_LOS)
  φ_atm(x,y)   — atmospheric delay phase (tropospheric + ionospheric)
  φ_noise(x,y) — decorrelation noise (thermal noise + temporal decorrelation)
  λ             — SAR wavelength (C-band: 5.6 cm; L-band: 24 cm; X-band: 3.1 cm)
```

**Inverse problem:** Recover the continuous 2D deformation map d_LOS(x,y) from the wrapped phase φ_int(x,y) via phase unwrapping and atmospheric correction; the problem is underdetermined wherever coherence is low.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(C/L-band SAR pulse) → F(terrain + atmosphere) → D(SAR processor + interferogram)

**Key mismatch parameters:**
- `temporal_baseline`: time between acquisitions; nominal 12 days (Sentinel-1), perturbed 365 days (heavy decorrelation)
- `atmospheric_delay_std`: tropospheric phase delay variation; nominal 1 cm equivalent, perturbed 5 cm (humid tropics)
- `coherence`: mean interferometric coherence; nominal 0.8, perturbed 0.3 (vegetated terrain, low coherence)
- `dem_error`: error in reference DEM used for topographic phase removal; nominal 5 m, perturbed 30 m

**Dataset format:**
- `x_true: (H, W)` — ground-truth unwrapped deformation map d_LOS in cm
- `y: (H, W)` — wrapped interferometric phase φ_int in radians [−π, π]

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| SNAPHU (statistical-cost network flow) | Classical | Chen & Zebker, J. Geophys. Res. 106:20043 (2001) | Standard open-source phase unwrapping; SNAPHU is the community benchmark tool |
| SBAS (Small Baseline Subset) | Classical time-series | Berardino et al., IEEE TGRS 40:2375 (2002) | Multi-temporal InSAR time-series for slow deformation monitoring |
| StaMPS (PS-InSAR) | Classical | Hooper et al., Geophys. Res. Lett. 31:L23611 (2004) | Persistent scatterer InSAR for urban deformation in low-coherence areas |
| PhaseNet (deep learning) | Deep Learning | Sica et al., IEEE Trans. Geosci. Remote Sens. 60:1 (2022) | CNN for SAR phase unwrapping trained on simulated interferograms |
| InSAR-Transformer | Transformer | Li et al., IEEE TGRS 61:1 (2023) | Transformer-based joint phase unwrapping and atmospheric correction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Zheng et al. (2024)** "Deep learning for InSAR time series deformation extraction with attention-based temporal modeling," *Remote Sens. Environ.* — transformer temporal attention for SBAS-like time-series decomposition with improved atmospheric correction.
2. **Jiang et al. (2024)** "Combining physics-based and data-driven models for InSAR phase unwrapping in challenging terrains," *IEEE TGRS* — hybrid PINN + SNAPHU approach handling severe decorrelation in tropical forests.
3. **Ansari et al. (2023)** "Sequential Estimator: Toward Efficient InSAR Time Series Analysis," *IEEE TGRS 69:1* — efficient sequential SBAS enabling near-real-time deformation monitoring from Sentinel-1.
4. **Liu et al. (2024)** "Ionospheric correction for L-band InSAR using machine learning," *J. Geophys. Res. Solid Earth* — ML-based ionospheric phase estimation improving L-band InSAR accuracy over equatorial regions.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/insar_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/insar_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/insar_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/insar/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

InSAR is correctly modeled as a phase unwrapping + atmospheric correction inverse problem on wrapped interferometric phase, with the forward model appropriately decomposing φ_int into topographic, deformation, atmospheric, and noise components. Algorithm routing spans the canonical SNAPHU unwrapper, multi-temporal SBAS/PS-InSAR time-series methods, and deep learning (PhaseNet, InSAR-Transformer) approaches that are increasingly used for operationally challenging unwrapping tasks. The mismatch parameters — temporal baseline, atmospheric delay, coherence, and DEM error — capture the primary sources of InSAR quality degradation across different climatic zones and sensor configurations. The benchmark is physically rigorous and covers the core InSAR algorithm ecosystem.

---
*Comprehensive 6-point check by deep-check pipeline v3*
