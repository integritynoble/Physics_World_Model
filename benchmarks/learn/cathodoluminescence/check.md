# Comprehensive 6-Point Check — Cathodoluminescence (CL) Imaging

**URL:** https://pwm.platformai.org/benchmark/cathodoluminescence
**Check Date:** 2026-03-06
**Status:** PASS (with noted limitations)

---

## 1. Physics & Forward Model

**Modality:** Cathodoluminescence (CL) Imaging

**Physical principle:** Cathodoluminescence is the emission of photons by a material excited by a high-energy electron beam (typically 1–30 keV in an SEM or STEM). The electron beam generates electron-hole pairs in semiconductors, quantum wells, and plasmonic nanostructures; their radiative recombination produces photons in the UV–IR range. In hyperspectral CL, a scanning electron beam maps the emission spectrum at each pixel, producing a 3D datacube (x, y, λ). The resolution is limited by the electron beam excitation volume (carrier diffusion length) rather than the optical diffraction limit. The reconstruction challenge involves correcting for spectrometer response, detector efficiency variations, carbon contamination, and beam-induced sample modification.

**Forward model:**
```
CL signal model:
  I_CL(x,y,λ) = η(λ) * [E_beam(x,y) * R(x,y,λ)] * T(λ) * D(λ) + n(x,y,λ)

where:
  E_beam(x,y)   — electron excitation volume (beam current × generation efficiency)
  R(x,y,λ)     — position-dependent emission spectrum (material property, ground truth)
  η(λ)          — collection efficiency (paraboloid mirror)
  T(λ)          — spectrometer transmission function
  D(λ)          — detector quantum efficiency
  n(x,y,λ)     — Poisson + readout noise

Inverse problem:
  y = A * diag(η ⊗ T ⊗ D) * x + n
  x ∈ R^{H×W×N_λ}   — true emission spectrum datacube
  y ∈ R^{H×W×N_λ}   — measured CL hyperspectral image
```

**Inverse problem:** Recover the true CL emission spectrum map R(x,y,λ) from the measured datacube y by correcting for system response (detector efficiency, spectrometer transmission, mirror collection efficiency) and removing noise and artefacts.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** M(electron beam) → R(optical emission) → D(spectrometer + CCD)

**Key mismatch parameters:**
- `beam_current_drift` (b_c): electron beam current variation during scan; nominal 0.0, perturbed 1.0 (relative %)
- `collection_efficiency_variation` (c_e): spatial non-uniformity in paraboloid mirror collection; nominal 0.0, perturbed 4.0 (spatial %)
- `spectral_calibration_error` (s_c): wavelength axis calibration offset; nominal 0.0 nm, perturbed 0.4 nm
- `carbon_contamination` (c_c): signal loss from carbon layer deposition; nominal 0.0, perturbed 2.0 (relative signal loss %)

**Dataset format:**
- `x_true: (H, W)` — 2D CL intensity map at peak emission wavelength (ground truth)
- `y: (H, W, N_λ)` — measured hyperspectral CL datacube; H×W spatial positions, N_λ spectral channels
- `H_ideal: (H*W*N_λ, H*W)` — ideal system response operator

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Deconv | Classical | Richardson-Lucy / Wiener deconvolution | Spatial deconvolution to correct for excitation volume broadening; appropriate for spatial CL |
| Calibration-Lookup | Classical | — | Empirical system response calibration using reference samples; standard CL spectral correction |
| Peak Fitting | Classical | — | Gaussian/Voigt peak fitting for emission wavelength and intensity extraction from CL spectra |
| PnP-BM3D | Plug-and-Play | Danielyan et al., IEEE TIP 2012 | BM3D denoising prior for low-count CL image restoration |
| ResNet-Calib | Deep Learning | — | ResNet-based calibration artefact correction (generic; real CL DL reference: Vega et al. 2023) |
| CalibFormer | Transformer | — | Transformer for instrument response correction (generic archetype) |

---

## 4. Literature & State of the Art (2024–2025)

1. **Deep learning for hyperspectral CL unmixing** (2023–2024): NMF and CNN approaches for separating multiple emission species in III-nitride semiconductor CL datacubes; achieves sub-5 nm spatial resolution in quantum well width fluctuation mapping.
2. **Transformer for CL spectrum denoising** (Vega et al., 2023 / extended 2024): Attention-based architecture for photon-starved CL spectra; handles non-stationary Poisson noise in beam-sensitive samples.
3. **Carbon contamination correction via deep learning** (2024): CNN trained on time-series CL acquisitions to predict and correct for progressive signal attenuation from e-beam-induced carbon deposition.
4. **Super-resolution CL with scanning STEM** (2024–2025): Deep learning model combining STEM HAADF structural image with CL spectrum to achieve sub-nm CL spatial resolution via structured illumination approaches.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cathodoluminescence_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cathodoluminescence_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cathodoluminescence_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/cathodoluminescence/`.

---

## 6. Comprehensive Assessment

**Status:** PASS (with noted limitations)

Algorithm routing uses the `scientific_instrumentation` category pool (11 methods). Deconv, Calibration-Lookup, and Peak Fitting are genuinely appropriate classical methods for CL image processing. PnP-BM3D (Danielyan et al., IEEE TIP 2012) is real and applicable to low-count CL denoising. ResNet-Calib and CalibFormer have generic citations that represent algorithm archetypes rather than specific published CL papers. The four mismatch parameters address the key CL system calibration issues: beam current drift, collection efficiency variation, spectral calibration, and carbon contamination. No functional code changes are required.

---
*Comprehensive 6-point check by deep-check pipeline v3*
