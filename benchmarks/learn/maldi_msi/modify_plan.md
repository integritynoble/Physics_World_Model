# Modify Plan — maldi_msi

## Current State

- **Category:** scientific_instrumentation
- **Carrier:** Ion
- **Score key:** scientific_instrumentation
- **Algorithms (from catalog):**
  1. Deconv (Classical) -- Analytical baseline
  2. PnP-BM3D (PnP) -- Danielyan et al., 2012
  3. ResNet-Calib (Deep Learning) -- ResNet for calibration, 2022
  4. CalibFormer (Transformer) -- Transformer calibration, 2024
- **Leaderboard (live):** Deconv, PnP-BM3D, ResNet-Calib, CalibFormer (4 entries)

## Assessment

The algorithms are **acceptable but generic**. MALDI Mass Spectrometry Imaging is in the "scientific_instrumentation" category which groups mass spec, atom probe, and diffraction instruments together.

- **Deconv** is a reasonable classical baseline for MALDI-MSI signal processing (deconvolution of overlapping peaks / spatial deconvolution of ion signals).
- **PnP-BM3D** is acceptable as a generic denoising-based reconstruction prior for spatial denoising of ion images.
- **ResNet-Calib** and **CalibFormer** are described as calibration-focused networks. For MALDI-MSI, calibration of m/z axes and signal normalization are real concerns, so these are defensible, though generic.

MALDI-MSI has specific algorithmic needs around peak picking, spectral deconvolution, and spatial super-resolution of ion images. Domain-specific methods include:
- Cardinal (Bemis et al., Bioinformatics 2015) for spatial segmentation
- SCiLS (Thiele et al., 2014) for preprocessing
- msImpute (Hediyeh-zadeh et al., 2023) for missing value imputation

However, the benchmark frames the problem as signal reconstruction (inverse problem), not as a chemometrics pipeline. Under that framing, the generic algorithms are acceptable.

## Current Algorithm Count (updated 2026-03-06)

Full pool (11 algorithms, now using spectroscopy pool): SG-ALS, Baseline Correction, SVD, PnP-DnCNN, CDAE, U-Net-Spectra, Cascade-UNet, PINN-Spectra, SpectraFormer, DiffusionSpectra, ScoreSpectra. This is a better assignment than the old scientific_instrumentation pool.

**Status:** PASS — check.md written 2026-03-06

## Verdict

No code changes needed.
