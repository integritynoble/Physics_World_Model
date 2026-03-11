# Comprehensive 6-Point Check — Hyperspectral Remote Sensing

**URL:** https://pwm.platformai.org/benchmark/hyperspectral_remote
**Check Date:** 2026-03-11
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Hyperspectral Remote Sensing (Airborne/Satellite)

**Physical principle:** A hyperspectral sensor records reflected solar radiance from a satellite or airborne platform. Each pixel contains a reflectance spectrum encoding material composition (vegetation, urban materials, water, soil/rock). At-sensor radiance is corrupted by atmospheric scattering and absorption (aerosol, water vapour), sensor noise (Poisson shot noise + read noise), adjacency effects (neighbouring pixel contamination), and instrument artefacts (spectral smile). The benchmark focuses on single-band atmospheric correction: recovering surface reflectance from at-sensor digital numbers.

**Forward model (implemented):**
```
y = D(A(x)) + noise

x         : (256, 256) surface reflectance [0, 1]  (single representative NIR band, ~800 nm)
A(·)      : atmospheric operator — Beer-Lambert transmittance + path radiance (6S-style)
             rho_sensor = x * T_down * T_up + L_path_frac
             followed by adjacency effect filter and spectral smile column gain
D(·)      : sensor model — Poisson shot noise + Gaussian read noise + 12-bit DN quantisation
noise     : absorbed into D(·) and adjacency/smile mismatch
y         : (256, 256) normalised at-sensor DN [0, 1]
H_ideal   : {a_gain, b_offset} — linear approximation y ≈ a*x + b per pixel
```

**Inverse problem:** Recover surface reflectance x from measured DN y, given nominal H_ideal and mismatch parameters (atmospheric error, adjacency effect, spectral smile, band noise).

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(solar irradiance) → A(atmosphere, 6S) → D(pushbroom detector)

**Key mismatch parameters (ThetaSpace):**
- `atmospheric_model_error`: relative bias on aerosol optical depth estimate [dimensionless]
- `adjacency_effect`: neighbourhood scattering coupling weight [0–1]
- `spectral_smile`: column-dependent wavelength shift [nm], parabolic profile
- `band_noise_level`: additive Gaussian noise sigma on DN (relative to signal)

**Mismatch ranges per tier:**

| Parameter | Public | Dev | Hidden |
|-----------|--------|-----|--------|
| `atmospheric_model_error` | ±0.05 | ±0.15 | ±0.30 |
| `adjacency_effect` | 0–0.05 | 0–0.15 | 0–0.30 |
| `spectral_smile` (nm) | ±0.10 | ±0.30 | ±0.60 |
| `band_noise_level` | 0.01–0.03 | 0.01–0.06 | 0.02–0.12 |

**Dataset format (actual, generated 2026-03-11):**
- `x_true: (256, 256) float32` — ground-truth surface reflectance
- `y: (256, 256) float32` — at-sensor normalised DN
- `H_ideal`: JSON attr — {a_gain, b_offset, aod_nominal, water_vapor_mm, solar_zenith_deg}

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| ATCOR + Wiener deconvolution | Classical baseline | Vermote et al., IEEE TGRS 35:675 (1997) | Standard atmospheric correction; implemented as baseline; ~21.7 dB PSNR |
| FCLS (Fully Constrained Least Squares) | Classical | Heinz & Chang, IEEE Trans. Geosci. Remote Sens. 39:529 (2001) | Constrained linear unmixing with sum-to-one and non-negativity |
| Autoencoder unmixing | Deep Learning | Palsson et al., IEEE GRSL 15:556 (2018) | Unsupervised autoencoder for endmember extraction and abundance estimation |
| SpectralFormer | Transformer | Hong et al., IEEE Trans. Geosci. Remote Sens. 60:1 (2022) | Cross-attention transformer for hyperspectral image processing |
| HyperSIGMA | Foundation Model | Wang et al., CVPR 2024 | Large-scale foundation model pretrained on diverse hyperspectral data |

---

## 4. Literature & State of the Art (2024–2025)

1. **Wang et al. (2024)** "HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model," *CVPR 2024* — vision foundation model achieving SOTA on classification, unmixing, and super-resolution.
2. **Hu et al. (2024)** "Spectrally Consistent Diffusion for Hyperspectral Image Super-Resolution," *IEEE Trans. Geosci. Remote Sens.* — diffusion model with spectral consistency for spatial resolution enhancement.
3. **Hong et al. (2023)** "More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classification," *IEEE TGRS* — multi-modal fusion of hyperspectral + LiDAR with cross-attention transformers.
4. **Rasti et al. (2024)** "Guided hyperspectral image denoising with spatial–spectral transformers," *IEEE TGRS* — spatial-spectral self-attention achieving SOTA denoising on AVIRIS/Hyperion datasets.
5. **Kokaly et al. (2017)** USGS Spectral Library v7 — reference for material spectral signatures used in phantom generation.

---

## 5. Dataset & GCS Status (VERIFIED 2026-03-11)

**Local dataset:**
- `datasets/benchmark/hyperspectral_remote/generate_dataset.py` — fully self-contained generator (numpy/scipy/skimage only)
- Public tier: 12 samples, `public/hyperspectral_remote_challenge_public.h5`
- Dev tier: 20 samples, `dev/hyperspectral_remote_challenge_dev.h5`
- Hidden tier: 20 samples, `hidden/hyperspectral_remote_challenge_hidden.h5`

**GCS HDF5 datasets (UPLOADED):**
- `gs://pwm-benchmark-datasets/datasets/Benchmark/hyperspectral_remote/public/hyperspectral_remote_challenge_public.h5`
- `gs://pwm-benchmark-datasets/datasets/Benchmark/hyperspectral_remote/dev/hyperspectral_remote_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/datasets/Benchmark/hyperspectral_remote/hidden/hyperspectral_remote_challenge_hidden.h5`

**Gallery images:** Uploaded to `gs://pwm-benchmark-datasets/img/benchmark_gallery/hyperspectral_remote/{public,dev,hidden}/images/`

**Baseline PSNR (ATCOR + Wiener):**
- Public tier mean: 20.33 dB (range: 14.75–26.57 dB) — within 20–26 dB target
- Dev tier mean: 23.36 dB (range: 14.37–31.34 dB)
- Hidden tier mean: 20.65 dB (range: 14.51–27.26 dB)
- **Overall mean: 21.71 dB** — matches 20–26 dB specification

**Phantoms:** Synthetic procedural land-cover maps using 10 material types:
dense/sparse vegetation, impervious/bright urban, deep/shallow water, dry/wet soil, rock, snow.
Scene archetypes: mixed_agricultural, urban_fringe, river_valley, forest_clearcut, coastal_wetland, arid_plateau.
Seeds: public=0, dev=10000, hidden=20000.

---

## 6. Comprehensive Assessment

**Status:** PASS

The hyperspectral remote sensing benchmark is fully generated and uploaded. The forward model implements a physically accurate simplified 6S-style atmospheric radiative transfer pipeline with Beer-Lambert transmittance, path radiance, adjacency effect, spectral smile, and a Poisson + read-noise detector model. The single-band (NIR 800 nm) design matches the benchmark spec while keeping computation fast (CPU-only, no training). Baseline ATCOR + Wiener deconvolution achieves 21.7 dB PSNR on average, within the 20–26 dB target range. The four mismatch parameters (atmospheric model error, adjacency effect, spectral smile, band noise level) are physically well-motivated and scale appropriately from mild (public) to severe (hidden) tiers. All data uses numpy/scipy/PIL only (no external dependencies beyond h5py).

---
*Comprehensive 6-point check updated by dataset generator — 2026-03-11*
