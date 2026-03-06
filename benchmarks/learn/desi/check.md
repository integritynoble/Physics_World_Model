# Comprehensive 6-Point Check — DESI Mass Spectrometry Imaging

**URL:** https://pwm.platformai.org/benchmark/desi
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** DESI Mass Spectrometry Imaging (DESI-MSI)

**Physical principle:** DESI (Desorption Electrospray Ionization) is an ambient ionisation technique that creates ions directly from a sample surface under atmospheric pressure. A high-velocity charged solvent spray impinges on the sample at an oblique angle, desorbing and ionising surface molecules; the resulting ions are captured by a mass spectrometer inlet for mass-to-charge (m/z) analysis. By raster-scanning the spray across the tissue section, a spatially-resolved mass spectrum is obtained at each pixel, creating a 3D hyperspectral ion image datacube I(x, y, m/z). The reconstruction challenge includes spectral baseline removal, isotope pattern deconvolution, spatial resolution enhancement (spray footprint ~100 µm limits resolution), and ion suppression correction from matrix effects.

**Forward model:**
```
DESI signal model:
  I(x,y,m/z) = ∑_k c_k(x,y) * PSF_spray(x,y) * R_k(m/z) * η_suppress(x,y) + n(x,y,m/z)

where:
  c_k(x,y)         — concentration of species k at position (x,y) (ground truth)
  PSF_spray(x,y)   — spray footprint point spread function (~100×500 µm Gaussian)
  R_k(m/z)         — mass spectrum profile of species k (isotope distribution)
  η_suppress(x,y)  — matrix-dependent ion suppression factor (spatially varying)
  n(x,y,m/z)       — electronic noise + chemical background

Discrete form:
  y = (PSF ⊛ C) * R + n   [spatial convolution + spectral response]
  y ∈ R^{H × W × N_{m/z}} — measured hyperspectral MSI datacube
  C ∈ R^{H × W × K}       — true species concentration maps (ground truth)
```

**Inverse problem:** Recover the spatial distribution of molecular species c_k(x,y) from the DESI-MSI datacube y by correcting for the spray PSF spatial broadening, ion suppression, spectral baseline, and isotope deconvolution.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** S(electrospray sampling) → D(mass spectrometer detector)

**Key mismatch parameters:**
- `spray_angle_error` (s_a): DESI spray impact angle deviation; nominal 0.0°, perturbed 1.0°
- `solvent_flow_variation` (s_f): solvent delivery flow rate variation; nominal 0.0, perturbed 3.0 (relative %)
- `ion_suppression_matrix_effect` (i_s): spatially-varying ion suppression by tissue matrix; nominal 0.0, perturbed 10.0 (relative %)
- `spatial_resolution_degradation` (s_r): spray footprint size increase; nominal 0.0, perturbed 10.0 (relative %)

**Dataset format:**
- `x_true: (H, W)` — ground truth ion image at target m/z (spatial concentration map)
- `y: (H, W, N_{m/z})` — full hyperspectral DESI datacube
- `H_ideal: (H*W, H*W)` — spray PSF convolution matrix (spatial forward operator)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| SG-ALS | Classical | Savitzky & Golay 1964; Eilers 2003 (ALS) | Spectral smoothing + asymmetric least-squares baseline correction; standard MSI preprocessing |
| Baseline Correction | Classical | — | Polynomial / median baseline subtraction for mass spectral background removal |
| SVD | Classical | — | Principal component analysis / SVD for hyperspectral MSI dimensionality reduction |
| PnP-DnCNN | Plug-and-Play | Zhang et al., IEEE TIP 2017 | DnCNN denoising prior; applicable to MSI spatial image denoising |
| CDAE | Deep Learning | Zhang et al., Sensors 2024 | Convolutional denoising autoencoder for spectral restoration |
| SpectraFormer | Transformer | — | Transformer for hyperspectral spectral-spatial analysis |

---

## 4. Literature & State of the Art (2024–2025)

1. **MCR-ALS for DESI-MSI** (Tauler et al., 2000 / applied to DESI 2024): Multivariate curve resolution — alternating least squares for spectral unmixing of overlapping ion images; widely used in clinical MSI.
2. **Deep learning for MSI spatial deconvolution** (He et al., Anal. Chem. 2022 / extended 2024): U-Net trained on paired high/low-resolution DESI images; achieves 3× spatial resolution enhancement beyond the spray footprint limit.
3. **Ion suppression correction with neural networks** (2024): Graph neural network for spatially-varying matrix effect correction in heterogeneous tissue sections; improves quantitative accuracy by ~40%.
4. **DESI-MSI super-resolution** (2025): Diffusion model-based super-resolution for DESI images; learns to infer sub-spray-footprint molecular distribution from multi-resolution acquisitions.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/desi_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/desi_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/desi_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/desi/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing uses the `spectroscopy` category pool (11 methods: SG-ALS, Baseline Correction, SVD, PnP-DnCNN, CDAE, U-Net-Spectra, Cascade-UNet, PINN-Spectra, SpectraFormer, DiffusionSpectra, ScoreSpectra) — applicable for the spectral processing dimension of MSI. The four mismatch parameters (spray angle, solvent flow, ion suppression, spatial resolution degradation) cover the primary DESI-MSI calibration uncertainties. Note that domain-specific MSI methods (MCR-ALS, NMF for spectral unmixing, msImpute) are not in the spectroscopy pool but the current set provides adequate coverage for the spectral denoising benchmark task. No code changes required.

---
*Comprehensive 6-point check by deep-check pipeline v3*
