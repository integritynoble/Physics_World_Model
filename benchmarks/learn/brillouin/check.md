# Comprehensive 6-Point Check — Brillouin Microscopy

**URL:** https://pwm.platformai.org/benchmark/brillouin
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Brillouin Microscopy

**Physical principle:** Brillouin microscopy measures the inelastic scattering of laser photons from thermally-excited acoustic phonons (Brillouin scattering). The frequency shift of scattered light Ω_B is proportional to the acoustic velocity in the sample, which in turn reflects the longitudinal storage modulus M'. In cells and tissues, Brillouin shift (typically 4–12 GHz in biological samples at 532 nm) maps viscoelastic properties at diffraction-limited spatial resolution. A VIPA (Virtually Imaged Phase Array) spectrometer disperses the scattered light onto a CCD for spectral analysis; the key reconstruction challenge is extracting accurate Brillouin shift from noisy, elastically-scattered-contaminated spectra.

**Forward model:**
```
Measured spectrum:
  I(ν) = I_elastic * δ(ν-ν_0) + I_Brillouin * L(ν; Ω_B, Γ_B) + I_noise(ν)

Lorentzian Brillouin peak:
  L(ν; Ω_B, Γ_B) = (Γ_B/2π) / [(ν - ν_0 - Ω_B)² + (Γ_B/2)²]

where:
  ν_0    — laser frequency (Hz)
  Ω_B    — Brillouin frequency shift (GHz): Ω_B = 2nV_s sin(θ/2) / λ
  Γ_B    — Brillouin linewidth (GHz, related to phonon lifetime)
  V_s    — longitudinal sound velocity (m/s)
  n      — refractive index
  θ      — scattering angle (180° in backscattering geometry)

Inverse problem reduces to spectral peak fitting with elastic leakage subtraction.
```

**Inverse problem:** Extract the Brillouin frequency shift map Ω_B(x,y) from noisy VIPA spectra I(ν,x,y), where the main challenges are elastic scattering leakage, spectral calibration errors, and photon shot noise.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** M(pump laser) → R(VIPA spectrometer) → D(CCD camera)

**Key mismatch parameters:**
- `brillouin_shift_calibration` (b_s): systematic offset in frequency axis calibration; nominal 0.0 MHz, perturbed 10.0 MHz
- `vipa_fsr_error` (v_f): VIPA free spectral range calibration error; nominal 0.0, perturbed 0.1 (relative)
- `elastic_scattering_leakage` (e_s): ratio of elastic to Brillouin peak intensity; nominal 0.0, perturbed -6.0 (dB)

**Dataset format:**
- `x_true: (H, W)` — Brillouin shift map in GHz (ground truth spatial map)
- `y: (H, W, N_freq)` — spatially resolved spectra; H×W positions, N_freq spectral channels
- `H_ideal: (H*W*N_freq, H*W)` — spectral forward operator (VIPA dispersion + Lorentzian model)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| SG-ALS | Classical | Savitzky & Golay 1964 (smoothing); Eilers 2003 (ALS baseline) | Spectral smoothing + asymmetric least squares baseline correction; standard Raman/Brillouin preprocessing |
| Baseline Correction | Classical | — | Polynomial or spline baseline subtraction to remove elastic background |
| SVD | Classical | — | Singular value decomposition for spectral denoising and peak separation |
| PnP-DnCNN | Plug-and-Play | Zhang et al., IEEE TIP 2017 | DnCNN denoising prior applied to spectral data; removes noise without distorting peak shapes |
| CDAE | Deep Learning | Zhang et al., Sensors 2024 | Convolutional denoising autoencoder for spectral data restoration |
| SpectraFormer | Transformer | — | Transformer architecture for spectral sequence analysis and peak extraction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Deep learning for Brillouin spectral analysis** (Remer & Bhatt, Biomed. Opt. Express 2020 / extended 2024): CNN directly extracts Brillouin shift and linewidth from raw spectra without explicit peak fitting; handles high elastic background scenarios.
2. **Stimulated Brillouin scattering microscopy** (Ballmann et al., 2024): SBS reduces acquisition time 100× vs spontaneous Brillouin; associated reconstruction algorithms handle coherent artefacts.
3. **Brillouin imaging of cell mechanics** (Antonacci & Braakman, 2024): Review of spectral analysis methods for biological Brillouin microscopy; benchmarks Lorentzian fitting vs DL methods across noise levels.
4. **VIPA calibration deep learning** (2025): Neural network for VIPA FSR and etalon calibration correction; improves spatial Brillouin shift accuracy from ~20 MHz to ~5 MHz.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/brillouin_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/brillouin_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/brillouin_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/brillouin/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing uses the `spectroscopy` category pool (11 methods: SG-ALS, Baseline Correction, SVD, PnP-DnCNN, CDAE, U-Net-Spectra, Cascade-UNet, PINN-Spectra, SpectraFormer, DiffusionSpectra, ScoreSpectra). SG-ALS and Baseline Correction are standard spectral preprocessing methods appropriate for Brillouin data. The three mismatch parameters (Brillouin shift calibration, VIPA FSR error, elastic scattering leakage) target the three principal sources of spectral artefacts in VIPA-based Brillouin microscopy. Note: Cascade-UNet is mislabelled as "Transformer" in the catalog (it is a UNet architecture) — minor cosmetic issue with no functional impact.

---
*Comprehensive 6-point check by deep-check pipeline v3*
