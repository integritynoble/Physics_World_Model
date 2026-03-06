# Comprehensive 6-Point Check — Hyperspectral Remote Sensing

**URL:** https://pwm.platformai.org/benchmark/hyperspectral_remote
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Hyperspectral Remote Sensing (Airborne/Satellite)

**Physical principle:** A hyperspectral sensor records reflected solar radiance in hundreds of narrow contiguous spectral bands (typically 400–2500 nm, ~5–10 nm spectral resolution) from an airborne or spaceborne platform using a pushbroom or whiskbroom scanner. Each pixel contains a full reflectance spectrum encoding material composition (vegetation, minerals, urban materials). The measured at-sensor radiance is corrupted by atmospheric absorption/scattering, illumination variation, and sensor noise. The inverse problems include: (1) atmospheric correction to recover surface reflectance, (2) spectral unmixing to decompose mixed pixels into endmember abundances, and (3) classification/target detection.

**Forward model:**
```
L_sensor(λ) = (1/π) · [ρ_s(λ) · E_↓(λ) · T_↑(λ)] / [1 − ρ_env(λ) · s_atm(λ)] + L_path(λ) + η

For spectral unmixing:
  ρ_s(x,y,λ) = Σ_k a_k(x,y) · e_k(λ) + ε

where:
  L_sensor(λ)  — at-sensor radiance spectrum [W m⁻² sr⁻¹ μm⁻¹]
  ρ_s(λ)       — surface reflectance spectrum to recover
  E_↓(λ)       — downwelling solar irradiance
  T_↑(λ)       — upward atmospheric transmittance
  L_path(λ)    — path radiance (atmospheric scattering)
  a_k(x,y)     — abundance of k-th endmember at pixel (x,y); Σ a_k = 1, a_k ≥ 0
  e_k(λ)       — endmember spectrum k (pure material signature)
```

**Inverse problem:** Recover surface reflectance spectra ρ_s(x,y,λ) via atmospheric correction, then decompose into endmember abundances a_k(x,y) via spectral unmixing.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(solar irradiance) → F(atmosphere + surface) → D(pushbroom imaging spectrometer)

**Key mismatch parameters:**
- `aerosol_optical_depth`: AOD at 550 nm; nominal 0.10 (clear), perturbed 0.40 (hazy atmosphere)
- `water_vapor_column`: precipitable water vapor; nominal 15 mm, perturbed 40 mm (humid tropics)
- `sensor_snr`: signal-to-noise ratio per band; nominal 300:1, perturbed 100:1 (less sensitive sensor)
- `spectral_smile`: non-uniform wavelength shift across detector rows; nominal 0.1 nm, perturbed 0.5 nm (keystone/smile artifacts)

**Dataset format:**
- `x_true: (H, W, C)` — ground-truth surface reflectance cube, H×W pixels × C spectral bands
- `y: (H, W, C)` — at-sensor radiance hyperspectral image (same dimensions)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| ATCOR / 6S atmospheric correction | Classical | Vermote et al., IEEE TGRS 35:675 (1997) | Second Simulation of Satellite Signal in Solar Spectrum; standard atmospheric correction |
| FCLS (Fully Constrained Least Squares) | Classical | Heinz & Chang, IEEE Trans. Geosci. Remote Sens. 39:529 (2001) | Constrained linear unmixing with abundance sum-to-one and non-negativity constraints |
| Autoencoder unmixing | Deep Learning | Palsson et al., IEEE GRSL 15:556 (2018) | Unsupervised autoencoder for simultaneous endmember extraction and abundance estimation |
| SpectralFormer | Transformer | Hong et al., IEEE Trans. Geosci. Remote Sens. 60:1 (2022) | Cross-attention transformer for hyperspectral image classification |
| HyperSIGMA | Transformer / Foundation | Wang et al., CVPR 2024 | Large-scale foundation model pretrained on diverse hyperspectral data |

---

## 4. Literature & State of the Art (2024–2025)

1. **Wang et al. (2024)** "HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model," *CVPR 2024* — vision foundation model for hyperspectral remote sensing achieving SOTA on classification, unmixing, and super-resolution.
2. **Hu et al. (2024)** "Spectrally Consistent Diffusion for Hyperspectral Image Super-Resolution," *IEEE Trans. Geosci. Remote Sens.* — diffusion model enforcing spectral consistency for hyperspectral spatial resolution enhancement.
3. **Hong et al. (2023)** "More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classification," *IEEE TGRS* — multi-modal fusion of hyperspectral + LiDAR with cross-attention transformers.
4. **Rasti et al. (2024)** "Guided hyperspectral image denoising with spatial–spectral transformers," *IEEE TGRS* — spatial-spectral self-attention achieving state-of-the-art denoising on AVIRIS/Hyperion datasets.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/hyperspectral_remote_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/hyperspectral_remote_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/hyperspectral_remote_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/hyperspectral_remote/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Hyperspectral remote sensing is correctly modeled as a two-stage inverse problem (atmospheric correction followed by spectral unmixing), with the forward model capturing atmospheric radiative transfer physics accurately. Algorithm routing appropriately covers classical FCLS unmixing and 6S/ATCOR atmospheric correction, deep learning autoencoders, and transformer-based SpectralFormer and HyperSIGMA foundation models that now lead the field. The mismatch parameters — aerosol optical depth, water vapor, sensor SNR, and spectral smile — represent the primary sources of domain shift between laboratory calibration and real airborne/spaceborne deployments. The benchmark is physically and algorithmically comprehensive.

---
*Comprehensive 6-point check by deep-check pipeline v3*
