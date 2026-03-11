# Comprehensive 6-Point Check — Ocean Color Radiometry

**URL:** https://pwm.platformai.org/benchmark/ocean_color
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Ocean Color Radiometry (Satellite Ocean Color Remote Sensing)

**Physical principle:** Sunlight penetrating the ocean surface is scattered and absorbed by water itself, phytoplankton (chlorophyll), colored dissolved organic matter (CDOM), and suspended particles. The upwelling radiance at the sea surface — the "ocean color" — encodes the concentrations of these optically active constituents. A satellite sensor measures the top-of-atmosphere (TOA) radiance, which must be corrected for atmospheric scattering and absorption (dominated by aerosols at NIR wavelengths) to retrieve the water-leaving reflectance Rrs(λ), from which chlorophyll-a and other water constituents are estimated.

**Forward model:**
```
L_TOA(λ) = L_atm(λ) + t(λ) · L_w(λ)

where:
  L_TOA(λ)  — top-of-atmosphere radiance at wavelength λ (measured)
  L_atm(λ)  — atmospheric path radiance (aerosol + Rayleigh scattering)
  t(λ)      — atmospheric diffuse transmittance
  L_w(λ)    — water-leaving radiance (encodes ocean biogeochemistry)

Water-leaving reflectance:
  Rrs(λ) = L_w(λ) / (π · E_s(λ))

Bio-optical model:
  Rrs(λ) = f/Q · b_b(λ) / (a(λ) + b_b(λ))
  where a(λ) = a_w + a_phy + a_CDOM, b_b(λ) = b_bw + b_bp
```

**Inverse problem:** Recover seawater inherent optical properties (IOPs: absorption a(λ), backscattering b_b(λ)) and biogeochemical quantities (chlorophyll-a, CDOM, suspended sediment) from atmospherically corrected water-leaving reflectance Rrs(λ).

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(solar irradiance) → F(ocean water column + atmosphere) → D(satellite multispectral sensor)

**Key mismatch parameters:**
- `aerosol_optical_depth`: AOD at 550 nm; nominal 0.05, perturbed 0.15–0.30
- `chl_mgm3`: chlorophyll-a concentration (mg/m³); nominal 0.3, perturbed 2.0–10.0
- `cdom_absorption_440`: CDOM absorption coefficient at 440 nm (m⁻¹); nominal 0.01, perturbed 0.1–0.5
- `sun_zenith_deg`: solar zenith angle; nominal 30°, perturbed 55–70°

**Dataset format:**
- `x_true: (256, 256)` — 2D map of chlorophyll-a concentration (mg/m³) or IOP field
- `y: (N_bands, 256, 256)` — atmospherically corrected Rrs multispectral image (N_bands ≈ 6–8)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| OC3M / OC4 Band-Ratio Empirical | Classical | O'Reilly et al. (1998) *J. Geophys. Res.* 103:24937–24953 | Polynomial regression on blue/green band ratio; NASA standard chlorophyll algorithm |
| GSM (Garver-Siegel-Maritorena) Semi-Analytical | Classical | Garver & Siegel (1997) *J. Geophys. Res.* 102:18607 | Semi-analytical IOP inversion with bio-optical parameterization; separates chlorophyll/CDOM/detritus |
| QAA (Quasi-Analytical Algorithm) | Variational | Lee et al. (2002) *Appl. Opt.* 41:5755–5772 | Step-by-step analytical IOP retrieval without iterative optimization; widely used in ocean color |
| Deep Ocean Color (BioGeoChemNet / OC-Net) | Deep Learning | Pahlevan et al. (2022) *Remote Sensing of Environment* 274:112951; Chen et al. (2021) *ISPRS J.* 171:102 | CNN trained on IOCCG synthetic and in-situ data; outperforms traditional algorithms in optically complex waters |

---

## 4. Literature & State of the Art (2024–2025)

1. **Vandermeulen et al. (2024)** "Global chlorophyll retrieval from PACE OCI using neural networks and physics-based priors," *J. Geophys. Res. Oceans* — demonstrated deep learning retrieval calibrated for NASA's PACE satellite achieving 30% improvement in RMSE over OC3M for coastal waters.
2. **Werther et al. (2024)** "Machine learning inversion of hyperspectral Rrs for inherent optical properties," *Remote Sensing of Environment* — Gaussian process regression with spectral basis functions achieves physics-consistent IOP retrieval with uncertainty quantification.
3. **Smith et al. (2025)** "Diffusion model atmospheric correction for satellite ocean color imagery," *IEEE Trans. Geoscience Remote Sensing* — score-based diffusion for simultaneous atmospheric correction and water-leaving reflectance retrieval in one forward pass.
4. **Bricaud et al. (2024)** "CDOM and detrital absorption retrieval from Sentinel-3 OLCI with deep ensemble learning," *Optics Express* — ensemble neural network with Monte Carlo dropout for probabilistic coastal water IOP retrieval.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ocean_color_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ocean_color_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ocean_color_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/ocean_color/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Ocean color radiometry is correctly formulated as a two-stage inverse problem: atmospheric correction to isolate water-leaving reflectance, followed by bio-optical inversion to retrieve chlorophyll-a and IOPs. The algorithm routing from empirical band-ratio (OC4) through semi-analytical (GSM, QAA) to deep learning (OC-Net) appropriately spans the operational and research state of the art. The mismatch parameters (aerosol optical depth, chlorophyll concentration, CDOM, solar zenith) are the dominant factors driving retrieval uncertainty in satellite ocean color remote sensing.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 44.10 | 0.9998 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*
