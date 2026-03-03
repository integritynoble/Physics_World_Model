# Comprehensive Benchmark QA Check -- DESI Mass Spectrometry Imaging

**URL:** https://pwm.platformai.org/benchmark/desi
**HTTP Status:** 200
**Check Date:** 2026-03-03 (comprehensive 6-point review)
**Reviewer:** Local server (automated + manual deep analysis)

---

## Table of Contents

1. [Benchmark Page Errors](#1-benchmark-page-errors)
2. [Local Dataset Inspection](#2-local-dataset-inspection)
3. [Public Dataset Source Assessment](#3-public-dataset-source-assessment)
4. [Algorithm Coverage Assessment](#4-algorithm-coverage-assessment)
5. [Improvement Suggestions](#5-improvement-suggestions)
6. [Action Items](#6-action-items)

---

## 1. Benchmark Page Errors

### Summary

| Severity | Count |
|----------|-------|
| HIGH     | 5     |
| MEDIUM   | 5     |
| LOW      | 3     |

### HIGH Severity

**H1. No local dataset directory exists (`datasets/benchmark/desi/` missing)**
- `ls datasets/benchmark/desi` returns exit code 2 (directory not found)
- Benchmark page references HDF5 data for public and dev tiers, but no local data is present
- Learning materials exist at `benchmarks/learn/desi/` but no actual measurement data
**Fix:** Build or download DESI MSI HDF5 datasets for all three tiers (public, dev, hidden).

**H2. Mismatch ranges on webpage differ from local desi.yaml config**
- Webpage: spray_angle [-1.0, 2.0], solvent_flow [-3.0, 6.0], ion_suppression [-10.0, 20.0], spatial_degradation [-10.0, 20.0]
- Local config (`benchmarks/configs/desi.yaml`): spray_angle [-5.0, 5.0], solvent_flow [0.0, 15.0], ion_suppression [0.0, 50.0], spatial_degradation [0.0, 50.0]
- Asymmetric webpage ranges are inconsistent with symmetric/wider local ranges
**Fix:** Reconcile webpage and local config. Decide authoritative source and sync both.

**H3. Config image shape 64x64 but expanded config offers 128/256/512**
- `desi.yaml` specifies x_shape [64, 64] and y_shape [64, 64]
- `desi_expanded.yaml` defines small (128x128), standard (256x256), large (512x512)
- Webpage does not clarify which resolution the actual challenge data uses
**Fix:** Specify the canonical challenge resolution on the webpage and in the config.

**H4. PSNR_norm undefined in scoring formula**
- Webpage scoring: `0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * (1 - ||y - Hx||/||y||)`
- PSNR_norm normalization method is not defined anywhere
- Local config lists `psnr` (raw) and `ssim` as metrics; no formula for normalization
**Fix:** Define PSNR_norm explicitly (e.g., PSNR / 50 or min-max scaling to [0,1]).

**H5. Forward model type mismatch: config says `nonlinear_operator` but category module is `microscopy_psf`**
- DESI MSI is an ambient ionization technique, not a microscopy PSF convolution
- The forward model should reflect the DESI desorption-ionization-collection process
- `microscopy_psf` is appropriate for optical microscopy, not mass spectrometry
**Fix:** Develop a dedicated `desi_ambient_ionization` category module or document why `microscopy_psf` is a valid approximation.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Webpage leaderboard shows "Cascade-UNet + gradient" at rank 1 (0.712) and "CDAE + gradient" at rank 2 (0.655), but no references or DOIs are provided for either method |
| M2 | Webpage leaderboard shows "MCR-ALS + gradient" as rank 4 method, but local config names it "SG-ALS + gradient" (Savitzky-Golay + ALS) -- naming inconsistency |
| M3 | Synthetic generator is `shepp_logan` -- a CT phantom has no chemical relevance for mass spectrometry imaging |
| M4 | Webpage says "RRUFF Raman Database (Lafuente et al. 2016)" is the evaluation reference, but this is a mineral spectroscopy database, not a DESI MSI dataset |
| M5 | Data source priority lists "experimental" first but dataset_id and dataset_url are both empty -- no experimental data path exists |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Wavelength range listed as "0-0 nm" in 01_physics_fundamentals.md -- should be m/z range or removed |
| L2 | Physics parameters section says "No specific physics parameters defined" -- spray voltage, solvent composition, and capillary-to-surface distance should be listed |
| L3 | Placeholder links: /benchmark/desi/compete, /benchmark/desi/contribute likely non-functional |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | File | Size | Samples | Status |
|------|------|------|---------|--------|
| Public | -- | -- | -- | MISSING |
| Dev | -- | -- | -- | MISSING |
| Hidden | -- | -- | -- | MISSING |

**No local dataset directory exists.** `datasets/benchmark/desi/` does not exist (exit code 2).

### What Exists Locally

| Path | Content | Status |
|------|---------|--------|
| `benchmarks/configs/desi.yaml` | Modality config | Present (94 lines) |
| `benchmarks/expanded_configs/desi_expanded.yaml` | Expanded config with variants | Present (95 lines) |
| `benchmarks/learn/desi/` | 6 learning docs + check.md + modify_plan.md | Present |
| `docs/modality_benchmarks/desi.md` | Maturity ladder (M0-M4) | Present |

### Config Schema Summary

| Key | Value |
|-----|-------|
| modality_id | desi |
| display_name | DESI Mass Spectrometry Imaging |
| category | Spectroscopy & Spectral Imaging |
| carrier | Ion |
| canonical DAG | S --> D |
| maturity | M0 |
| forward_model_type | nonlinear_operator |
| default_solver | mass_image_recon |
| x_shape | [64, 64] |
| y_shape | [64, 64] |
| operator_id | desi |
| has_dedicated_operator | true |

### Mismatch Parameters (from local config)

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Spray angle error | 0.0 | [-5.0, 5.0] | deg |
| Solvent flow variation | 0.0 | [0.0, 15.0] | - |
| Ion suppression (matrix effect) | 0.0 | [0.0, 50.0] | - |
| Spatial resolution degradation | 0.0 | [0.0, 50.0] | - |

### Dataset Integrity Assessment: **FAIL -- No data files present**

---

## 3. Public Dataset Source Assessment

### Current State: **NO REAL DATA**

- Data source priority: experimental > synthetic_web > generated
- Actual fallback: `generated` (Shepp-Logan phantom)
- Dataset ID: empty
- Dataset URL: empty
- Citation: empty
- License: empty

### Shepp-Logan Phantom as DESI Data: **POOR**

The Shepp-Logan phantom is a standard CT test image (ellipses simulating a head cross-section). It has zero chemical or molecular relevance for mass spectrometry imaging, where the ground truth should be spatially-resolved molecular ion maps (m/z images).

### Recommended Real Datasets for DESI MSI

| Dataset | Year | Content | Access | Suitability |
|---------|------|---------|--------|-------------|
| METASPACE public DESI datasets | 2016+ | Annotated metabolite images from multiple tissues | Open | HIGH |
| Human cancer tissue atlases (Eberlin et al.) | 2019-2024 | Brain, breast, kidney DESI-MSI | Published | HIGH |
| Waters DESI-XS application datasets | 2024 | High-resolution DESI at 5-20 um | Vendor | MEDIUM |
| 10x-DESI ExMSI mouse brain | 2024 | Cellular-level DESI at ~5 um resolution | bioRxiv | MEDIUM |

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Public: Well-known? | FAIL | No dataset exists; Shepp-Logan is irrelevant to MSI |
| Public: Accepted by professors? | FAIL | Shepp-Logan would not be accepted for MSI benchmarking |
| Dev: Protected? | FAIL | No dev data exists |
| Hidden: Protected? | FAIL | No hidden data exists |

---

## 4. Algorithm Coverage Assessment

### Currently on Leaderboard: 4 algorithms -- **MINIMAL COVERAGE**

| # | Algorithm | Type | Score | Notes |
|---|-----------|------|-------|-------|
| 1 | Cascade-UNet + gradient | Physics-informed DL | 0.712 | Top scorer, 2025 |
| 2 | CDAE + gradient | Convolutional DAE | 0.655 | Zhang et al., Sensors 2024 |
| 3 | PnP-DnCNN + gradient | Plug-and-play | 0.586 | Zhang et al., 2017 |
| 4 | SG-ALS + gradient | Classical baseline | 0.572 | Savitzky-Golay + ALS |

### Local Solvers (from config)

| Tier | Name | Module | GPU |
|------|------|--------|-----|
| traditional_cpu | Adjoint | `pwm_core.recon.adjoint` | No |
| best_quality | PnP-ADMM | `pwm_core.recon.pnp_admm` | Yes |

### Missing Famous/Recent Algorithms

| Priority | Algorithm | Year | Why Important |
|----------|-----------|------|---------------|
| HIGH | MCR-ALS (Multivariate Curve Resolution) | 2012-2024 | Gold standard for MSI spectral unmixing; applied directly to DESI images of rat brain (Sherma et al.) |
| HIGH | De-MSI (deep denoising) | 2025 | Purpose-built DL denoiser for MSI leveraging chemical prior knowledge |
| HIGH | DLADS (Dynamic Sampling) | 2023-2025 | 70-80% reduction in acquisition time; reconstructs from sparse DESI data |
| MEDIUM | DeepS (3D-SSNet) | 2023 | 3D sparse sampling network for accelerated MSI |
| MEDIUM | Peak Learning (ANN) | 2021 | Neural network for MSI peak detection and image reconstruction |
| MEDIUM | NMF (Non-negative Matrix Factorization) | Classical | Standard factorization baseline for spectral imaging |
| MEDIUM | t-SNE / UMAP spatial segmentation | 2018+ | Dimensionality reduction for MSI spatial analysis |
| LOW | PCA + ICA spectral decomposition | Classical | Fundamental baseline for spectral MSI |
| LOW | Model-Based MSI Reconstruction | 2025 | Integrates forward model with DL prior for accelerated MSI |

### Algorithm Gap Analysis

DESI benchmark has only 4 algorithms spanning 3 categories (DL, plug-and-play, classical). Main gaps:
- MSI-specific multivariate methods (MCR-ALS, NMF) are entirely missing
- Purpose-built MSI deep learning (De-MSI, DLADS, DeepS) not represented
- No spectral unmixing or factorization baselines
- All 4 leaderboard methods append "+ gradient" suggesting a shared gradient descent refinement step; standalone performance not reported

**Total gap: 9 algorithms across 4 categories (substantial deficiency)**

---

## 5. Improvement Suggestions

### 5.1 Dataset (CRITICAL -- no data exists)

1. **Create DESI MSI dataset directory** at `datasets/benchmark/desi/` with public/dev/hidden HDF5 files
2. **Replace Shepp-Logan phantom** with real or realistic molecular ion images -- consider METASPACE public DESI datasets or synthetic m/z images based on tissue lipid distributions
3. **Build proper ground truth** with spatial molecular maps (multiple m/z channels), not 2D grayscale phantoms
4. **Define HDF5 schema** appropriate for MSI: keys for `ion_images` (x,y,m/z), `mass_spectrum` (y), `forward_model` (H), `spec_params`

### 5.2 Forward Model

5. **Replace `microscopy_psf` category module** with a DESI-specific module that models: electrospray desorption geometry, solvent-surface interaction, ion transport to inlet, mass analyzer transfer function
6. **Fix image dimensions** -- 64x64 is unrealistically small for MSI; typical DESI images are 100-500 pixels per side with 100-1000 m/z channels
7. **Add m/z dimension** -- current config treats DESI as a 2D spatial problem, but MSI is inherently 3D (x, y, m/z)

### 5.3 Algorithms

8. **Add MCR-ALS baseline** -- most widely used and cited method for MSI data analysis
9. **Add De-MSI** (2025) -- state-of-the-art deep denoiser designed specifically for MSI
10. **Add DLADS** -- dynamic sparse sampling approach with proven 70-80% speedup
11. **Report standalone algorithm scores** without the shared "+ gradient" refinement step

### 5.4 Infrastructure

12. **Sync webpage mismatch ranges** with local config values
13. **Define PSNR_norm** in scoring formula
14. **Add proper references with DOIs** for all leaderboard methods
15. **Clarify RRUFF database role** -- it is a Raman spectroscopy database, not DESI-specific

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Create `datasets/benchmark/desi/` with real MSI data (public/dev/hidden HDF5) | Dataset team | TODO |
| CRITICAL | Replace Shepp-Logan phantom with molecular ion image ground truth | Dataset team | TODO |
| CRITICAL | Develop DESI-specific forward model (replace `microscopy_psf`) | Physics team | TODO |
| CRITICAL | Define PSNR_norm formula in scoring | Main server | TODO |
| HIGH | Sync webpage mismatch ranges with local `desi.yaml` | Main server | TODO |
| HIGH | Resolve config image shape (64x64 vs 128/256/512) | Config team | TODO |
| HIGH | Add MCR-ALS multivariate baseline to leaderboard | Algorithm team | TODO |
| HIGH | Add De-MSI (2025) deep denoiser | Algorithm team | TODO |
| MEDIUM | Add DLADS dynamic sparse sampling method | Algorithm team | TODO |
| MEDIUM | Add m/z spectral dimension to data schema (3D: x, y, m/z) | Physics team | TODO |
| MEDIUM | Fix wavelength range "0-0 nm" in physics fundamentals | Docs team | TODO |
| MEDIUM | Add proper references with DOIs for all leaderboard entries | Main server | TODO |
| LOW | Report standalone algorithm scores (without "+ gradient") | Algorithm team | TODO |
| LOW | Add NMF and PCA/ICA spectral baselines | Algorithm team | TODO |

---

## Appendix: Key References

- Takats et al. "Mass Spectrometry Sampling Under Ambient Conditions with DESI." Science 306:471-473 (2004). -- Foundational DESI paper.
- Eberlin et al. "Classifying Human Brain Tumors by Lipid Imaging with Mass Spectrometry." Cancer Res. 72(3):645-54 (2012).
- Sherma et al. "Imaging multivariate analysis to improve biochemical and anatomical discrimination in DESI." Analyst 137:5525-5532 (2012).
- De Laeter et al. "MCR-ALS for Enhanced Metabolomic Data Analysis of MSI." Anal. Chim. Acta 1032:67-77 (2018).
- Zhang et al. "CDAE for Mass Spectrometry Imaging." Sensors (2024). -- CDAE + gradient, rank 2 on leaderboard.
- De-MSI: Deep Learning-Based Data Denoising for Mass Spectrometry Imaging. Anal. Chem. (2025).
- DLADS: Deep Learning Approach for Dynamic Sparse Sampling for High-Throughput MSI. Anal. Chem. 94(2):1063-1071 (2022).
- DeepS: Accelerating 3D Mass Spectrometry Imaging via Deep Neural Network. Anal. Chem. 95(10):4767-4776 (2023).
- Lafuente et al. "The power of databases: the RRUFF project." Handbook of Mineralogy Spectroscopy (2016). -- Referenced on webpage.
- Peak Learning: Abdelmoula et al. "Peak learning of MSI data using ANNs." Nat. Commun. 12:5544 (2021).

---

*Comprehensive 6-point review on 2026-03-03. DESI benchmark is at M0 maturity with no local data, an inappropriate phantom generator, and a mismatched forward model module. Four leaderboard algorithms provide minimal coverage; MSI-specific multivariate and deep learning methods are entirely absent. Critical dataset and physics model work is required before this benchmark can be considered credible for the mass spectrometry imaging community.*