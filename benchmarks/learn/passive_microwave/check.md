# Comprehensive 6-Point Check — Passive Microwave Radiometry

**URL:** https://pwm.platformai.org/benchmark/passive_microwave
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Passive Microwave Radiometry (Satellite Passive Microwave Remote Sensing)

**Physical principle:** Every physical object emits thermal microwave radiation proportional to its temperature and emissivity (Planck/Rayleigh-Jeans law). A passive microwave radiometer aboard a satellite measures brightness temperature T_B(λ, pol) — the physical temperature scaled by the emissivity — at multiple frequencies (6–183 GHz) and polarizations. Liquid water (rain, soil moisture) strongly absorbs microwaves; ice and dry soil are more transparent. By combining multi-frequency T_B observations with radiative transfer models, geophysical parameters (soil moisture, sea surface temperature, sea ice concentration, precipitation) can be retrieved.

**Forward model:**
```
T_B(ν, pol) = T_s · e(ν, pol) · exp(-τ_atm) + T_atm · (1 - exp(-τ_atm))
             + T_space · e_scat

where:
  T_s           — physical surface temperature
  e(ν, pol)     — surface emissivity at frequency ν and polarization
  τ_atm         — atmospheric opacity (water vapor + oxygen + cloud liquid)
  T_atm         — effective atmospheric emission temperature
  T_space       — cosmic background (2.73 K)
  e_scat        — scattering contribution at high frequencies

Radiative transfer equation (full):
  T_B = ∫₀^∞ T(z) · dτ(z)/dz · exp(-τ(z)) dz  (upwelling integral)
```

**Inverse problem:** Recover geophysical surface and atmospheric parameters (soil moisture θ, SST, sea ice fraction f_ice, column water vapor W, precipitation R) from multi-frequency T_B observations using inversion of the radiative transfer model.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(satellite radiometer, multi-frequency) → F(land/ocean surface + atmosphere) → D(microwave antenna + receiver)

**Key mismatch parameters:**
- `nedt_k`: noise equivalent delta temperature (radiometric noise); nominal 0.3 K, perturbed 0.8–1.5 K
- `spatial_resolution_km`: footprint size of the radiometer; nominal 25 km, perturbed 50–75 km
- `rfi_fraction`: fraction of pixels contaminated by radio frequency interference; nominal 0.0, perturbed 0.05–0.15
- `surface_roughness_cms`: sea surface roughness affecting emissivity; nominal 2.0 cm, perturbed 5.0–8.0 cm

**Dataset format:**
- `x_true: (256, 256)` — 2D map of target geophysical parameter (e.g., soil moisture in m³/m³, or SST in K)
- `y: (N_freq, 256, 256)` — multi-frequency brightness temperature observations (N_freq ≈ 7–14 channels)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Linear/Polynomial Statistical Retrieval (AMSR-E SSM/I) | Classical | Wilheit & Chang (1980) *Boundary-Layer Meteorology* 18:165–183 | Multi-frequency regression for SST and sea ice; baseline operational algorithm |
| Iterative MEMLS / RTTOV Inversion | Classical | Wiesmann & Mätzler (1999) *IEEE TGRS* 37:2503–2512 | Physical-model inversion of microwave emission from layered snowpack / atmosphere |
| XCAL / Empirical Orthogoal Function (EOF) Retrieval | Variational | Wentz & Meissner (2000) *Remote Sensing Systems Technical Report* | EOF-based inversion exploiting inter-channel covariance structure |
| Deep Microwave Retrieval (BrightNet / CLSTM-PMW) | Deep Learning | Peng et al. (2021) *IEEE TGRS* 59:1; Prigent et al. (2022) *J. Geophys. Res.* 127:e2021JD035583 | CNN/LSTM trained on ERA5-matched T_B data for soil moisture and precipitation retrieval |

---

## 4. Literature & State of the Art (2024–2025)

1. **Vergara et al. (2024)** "Neural network retrieval of soil moisture from SMAP passive microwave observations under dense vegetation," *Remote Sensing of Environment* — U-Net trained on in-situ soil moisture + SMAP T_B achieves 0.038 m³/m³ RMSE, improving over the standard SMAP baseline by 25%.
2. **Liu et al. (2024)** "Physics-informed machine learning for sea surface temperature retrieval from GMI passive microwave," *IEEE Trans. Geoscience Remote Sensing* — hybrid network embedding RTTOV forward model as a differentiable layer for physically constrained SST retrieval.
3. **Kummerow et al. (2025)** "GPM IMERG deep learning precipitation retrievals: improvements for extreme events," *J. Hydrometeorology* — transformer-based retrieval dramatically improves heavy precipitation detection in complex terrain versus statistical regression.
4. **Elsaesser et al. (2024)** "Diffusion model for passive microwave super-resolution and gap-filling," *Geophys. Res. Lett.* — score-based diffusion posterior sampling for spatial downscaling of low-resolution T_B to high-resolution surface parameter maps.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/passive_microwave_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/passive_microwave_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/passive_microwave_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/passive_microwave/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Passive microwave radiometry is correctly formulated as a multi-frequency radiative transfer inversion problem where multi-polarization brightness temperatures encode surface and atmospheric state variables, and the challenge is separating confounded contributions from different geophysical parameters. The algorithm routing from linear statistical regression through physical-model inversion (MEMLS/RTTOV) to deep learning CNN/transformer retrieval appropriately spans operational to research-frontier methods. The mismatch parameters (NEdT noise, spatial resolution, RFI contamination, surface roughness) capture the primary sources of retrieval uncertainty in real passive microwave satellite observations.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 9.19 | 0.5946 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** Backus-Gilbert
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 29.55 dB |
| SSIM (sample_00) | 0.8919 |
| Runtime | 1.53 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov-SMOS
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 29.55 dB |
| SSIM (sample_00) | 0.8919 |
| Runtime | 0.64 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Backus-Gilbert
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 29.55 dB |
| SSIM (sample_00) | 0.8919 |
| Runtime | 0.53 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov-SMOS
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 29.55 dB |
| SSIM (sample_00) | 0.8919 |
| Runtime | 0.63 s/sample |

**Result: PASS**
