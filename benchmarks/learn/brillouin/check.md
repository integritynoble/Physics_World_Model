# Comprehensive Benchmark QA Check — Brillouin Microscopy

**URL:** https://pwm.platformai.org/benchmark/brillouin
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
| MEDIUM   | 7     |
| LOW      | 4     |

### HIGH Severity

**H1. Mismatch parameter ranges differ between webpage and local config**
- Webpage shows narrowed per-tier ranges (e.g., b_s public [-10.0, 20.0], dev [-12.0, 18.0], hidden [-7.0, 23.0])
- Local YAML `brillouin.yaml` shows a single global range: [-50.0, 50.0] for Brillouin shift calibration
- Similar discrepancies for VIPA FSR error (YAML: [-0.5, 0.5] vs webpage: [-0.1, 0.2] public) and elastic scattering leakage (YAML: [-30.0, 0.0] vs webpage: [-12.0, 6.0] public)
- The webpage ranges are narrower and tier-specific; the YAML has only a single tier-agnostic range
**Fix:** Sync YAML to contain tier-specific ranges matching webpage, or document the mapping from global to per-tier.

**H2. Leaderboard algorithms differ between webpage and auto-generated check**
- Webpage (fetched): Cascade-UNet + gradient, CDAE + gradient, PnP-DnCNN + gradient, SG-ALS + gradient
- Prior auto-generated check.md: CDAE + gradient, SpecFormer + gradient, PnP-BM3D + gradient, SG-ALS + gradient
- Two algorithms changed (Cascade-UNet replaced SpecFormer; PnP-DnCNN replaced PnP-BM3D) without version tracking
**Fix:** Add leaderboard version history or changelog on the webpage.

**H3. PSNR_norm undefined in scoring formula**
- Scoring formula: 0.4 x PSNR_norm + 0.4 x SSIM + 0.2 x (1 - ||y - Hx_hat|| / ||y||)
- PSNR_norm is not defined anywhere (no min/max bounds specified)
- Cannot reproduce the composite score without this definition
**Fix:** Define PSNR_norm = (PSNR - PSNR_min) / (PSNR_max - PSNR_min) with explicit PSNR_min and PSNR_max.

**H4. Forward model mismatch: webpage vs local config**
- Webpage describes M -> R -> D pipeline with explicit modulation, rotation, detector stages
- Local config `brillouin.yaml` specifies `forward_model_type: nonlinear_operator` with `category_module: microscopy_psf`
- Local learn docs describe signal equation as `y = PSF * x + noise` (convolution)
- Brillouin microscopy is NOT a PSF convolution problem -- it measures inelastic scattering spectra via VIPA etalon
- The `microscopy_psf` category module is incorrect for Brillouin spectroscopy
**Fix:** Implement a dedicated Brillouin forward model (VIPA spectrometer + Lorentzian lineshape + elastic leakage) and update category_module.

**H5. Elastic scattering leakage range physically suspect**
- YAML range: [-30.0, 0.0] (only negative, dimensionless)
- Webpage public range: [-12.0, 6.0] (includes positive values)
- Leakage is typically non-negative (added parasitic signal), so negative values are physically unclear
- No documentation on what the sign convention means
**Fix:** Document whether negative leakage is suppression vs. additive, and verify physical correctness.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Data dimensions [64, 64] are very small for Brillouin microscopy -- real VIPA spectrograms are typically 256x256 to 512x512 |
| M2 | Webpage reports 5 scenes per tier (15 total) while existing auto-check says 4 methods; verify sample-method consistency |
| M3 | Performance drop public->hidden: Cascade-UNet drops 3.45 dB (30.29->26.84), SG-ALS drops 1.05 dB (21.63->20.58); ~3.5 dB gap is moderate but undocumented |
| M4 | Primary metric in YAML is `psnr` but scoring formula weights PSNR and SSIM equally at 40% each; inconsistency |
| M5 | Default solver in YAML is `lorentz_fit` but the solver registry only has `Adjoint` and `PnP-ADMM`; `lorentz_fit` is not registered |
| M6 | `theta.density: 0.5` is documented but its meaning for Brillouin spectroscopy is unexplained |
| M7 | No references cited for any leaderboard algorithm (Cascade-UNet, CDAE, PnP-DnCNN, SG-ALS) |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Placeholder links: /benchmark/brillouin/compete, /benchmark/brillouin/contribute |
| L2 | No gallery images showing Brillouin spectrum reconstructions or VIPA spectrograms |
| L3 | Learn material `01_physics_fundamentals.md` describes generic PSF/diffraction optics, not Brillouin scattering physics |
| L4 | Signal equation `y = PSF * x + noise` is generic deconvolution, not Brillouin-specific (should involve Lorentzian lineshape, VIPA transfer function) |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | Path | Files Present | H5 Data? | Size |
|------|------|---------------|----------|------|
| Public | `datasets/benchmark/brillouin/public/` | **NONE** | **NO** | -- |
| Dev | `datasets/benchmark/brillouin/dev/` | **NONE** | **NO** | -- |
| Hidden | `datasets/benchmark/brillouin/hidden/` | **NONE** | **NO** | -- |

**CRITICAL: The `datasets/benchmark/brillouin/` directory does not exist. No local dataset files of any kind.**

### Config-Based Schema (Expected)

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| x_true | (64, 64) | float | Ground-truth Brillouin spectral map |
| y | (64, 64) | float | Measurement (observed spectrogram) |
| H_ideal | TBD | float | Ideal forward operator |

### Mismatch Parameters (from YAML)

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Brillouin shift calibration | 0.0 | [-50.0, 50.0] | MHz |
| VIPA FSR error | 0.0 | [-0.5, 0.5] | - |
| Elastic scattering leakage | 0.0 | [-30.0, 0.0] | - |

### Webpage Mismatch Parameters (per-tier)

| Parameter | Symbol | Public | Dev | Hidden | Unit |
|-----------|--------|--------|-----|--------|------|
| Brillouin shift calibration | b_s | [-10.0, 20.0] | [-12.0, 18.0] | [-7.0, 23.0] | MHz |
| VIPA FSR error | v_f | [-0.1, 0.2] | [-0.12, 0.18] | [-0.07, 0.23] | GHz |
| Elastic scattering leakage | e_s | [-12.0, 6.0] | [-10.8, 7.2] | [-13.8, 4.2] | - |

**Note:** The hidden tier ranges are not strictly wider than dev/public for all parameters (e.g., b_s hidden [-7.0, 23.0] is narrower on the low end than dev [-12.0, 18.0]), which contradicts the "increasing severity" philosophy.

### Dataset Integrity Assessment: **FAIL** -- No dataset files exist locally

---

## 3. Public Dataset Source Assessment

### Current Source: RRUFF Raman Database

| Property | Value |
|----------|-------|
| Dataset ID | `raman_dataset` |
| URL | https://rruff.info/ |
| Citation | RRUFF Project, University of Arizona |
| License | Public domain |
| Fallback | `generated` (Shepp-Logan phantom) |

### RRUFF Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Well-known? | GOOD | RRUFF is a well-established mineral spectroscopy database with >20,000 spectra |
| Relevant to Brillouin? | **POOR** | RRUFF contains Raman spectra, NOT Brillouin spectra |
| Accepted by professors? | MIXED | Excellent for Raman; inappropriate as ground truth for Brillouin |
| Size adequate? | ADEQUATE | >8,000 unoriented high-resolution Raman spectra available |

### Critical Concern: Raman vs. Brillouin Mismatch

RRUFF is a **Raman spectroscopy** database. Brillouin scattering probes acoustic phonons (GHz frequency shifts, 1-30 GHz typical), while Raman scattering probes optical phonons (THz frequency shifts, 100-3000 cm^-1). These are fundamentally different physical phenomena:

- **Brillouin**: GHz shifts, Lorentzian lineshape, sensitive to mechanical/viscoelastic properties
- **Raman**: THz shifts, varied lineshapes, sensitive to molecular bonds and crystal structure

Using Raman spectra as ground truth for a Brillouin benchmark is physically incorrect. The Shepp-Logan fallback is even less appropriate (a 2D spatial phantom, not a spectral dataset).

### Recommended Alternatives

| Source | Year | Type | Why Better |
|--------|------|------|------------|
| Zhang Lab (Scarcelli group) Brillouin datasets | 2019-2024 | Experimental VIPA spectrograms | Actual Brillouin microscopy data from leading group |
| Synthetic Lorentzian BGS generator | -- | Procedural | Can produce realistic Brillouin gain spectra with known ground truth |
| Distributed fiber-optic BOTDA datasets | 2020+ | Experimental | Brillouin frequency shift maps from optical fiber sensing |

### Protection Assessment

| Tier | Protection Level | Method |
|------|-----------------|--------|
| Public | POOR | RRUFF is fully public, but wrong modality |
| Dev | FAIR | Would use `generated` fallback (Shepp-Logan) |
| Hidden | FAIR | Would use `generated` fallback (Shepp-Logan) |

### Overall Source Quality: **POOR** -- Wrong spectroscopy modality (Raman instead of Brillouin)

---

## 4. Algorithm Coverage Assessment

### Currently Tested (Webpage Leaderboard): 4 Algorithms

| Rank | Algorithm | Type | Public PSNR | Dev PSNR | Hidden PSNR | Public SSIM | Dev SSIM | Hidden SSIM |
|------|-----------|------|-------------|----------|-------------|-------------|----------|-------------|
| 1 | Cascade-UNet + gradient | Deep learning (CNN) | 30.29 dB | 29.18 dB | 26.84 dB | 0.921 | 0.903 | 0.854 |
| 2 | CDAE + gradient | Convolutional denoising autoencoder | 29.05 dB | 24.38 dB | 24.87 dB | 0.901 | 0.782 | 0.798 |
| 3 | PnP-DnCNN + gradient | Plug-and-play denoiser | 25.29 dB | 23.65 dB | 22.06 dB | 0.811 | 0.756 | 0.693 |
| 4 | SG-ALS + gradient | Classical (Savitzky-Golay + ALS) | 21.63 dB | 21.60 dB | 20.58 dB | 0.674 | 0.673 | 0.626 |

### Composite Scores (from webpage)

| Algorithm | Public | Dev | Hidden |
|-----------|--------|-----|--------|
| Cascade-UNet + gradient | 0.751 | 0.713 | 0.686 |
| CDAE + gradient | 0.729 | 0.631 | 0.632 |
| PnP-DnCNN + gradient | 0.657 | 0.597 | 0.570 |
| SG-ALS + gradient | 0.566 | 0.541 | 0.528 |

### PWM Solver Registry

| Solver | Module | Status |
|--------|--------|--------|
| Adjoint | `pwm_core.recon.adjoint` | Registered (traditional_cpu) |
| PnP-ADMM | `pwm_core.recon.pnp_admm` | Registered (best_quality) |
| lorentz_fit | -- | **NOT registered** (listed as default_solver in YAML) |

**Neither registered solver appears on the webpage leaderboard. Complete disconnect between registry and benchmark results.**

### Missing Famous/Recent Algorithms (MUST ADD)

| Priority | Algorithm | Year | Citation Count | Why Important |
|----------|-----------|------|----------------|---------------|
| **CRITICAL** | Lorentzian least-squares fit | Classical | Foundational | THE standard Brillouin peak fitting method; listed as default solver but not benchmarked |
| **CRITICAL** | Maximum Entropy Reconstruction (MER) | 2020 | ~60 | Standard Brillouin denoising (Fiore et al., Biomed. Opt. Express 2020) |
| **CRITICAL** | Wavelet Analysis (WA) denoising | 2020 | ~60 | Paired with MER in the first Brillouin SNR benchmark paper |
| **HIGH** | PSRN (Physics-enhanced SR neural network) | 2025 | New | Unsupervised super-spatial-resolution for Brillouin frequency shift extraction (arXiv:2503.00506) |
| **HIGH** | Physics-informed denoising diffusion (PI-DDPM) | 2024 | ~30 | Physics-informed diffusion model for microscopy reconstruction (Nature Commun. Eng.) |
| **HIGH** | Pseudo-Voigt fitting | Classical | >1,000 | Standard alternative to Lorentzian for asymmetric Brillouin peaks |
| **HIGH** | Damped Harmonic Oscillator (DHO) model | Classical | >500 | Required for viscoelastic samples where Lorentzian fails |
| **MEDIUM** | DNN-assisted BOTDA reconstruction | 2023 | ~40 | Neural network for distributed Brillouin sensing |
| **MEDIUM** | Convolutional denoising autoencoder (1D spectral) | 2022 | ~80 | Specific to spectral denoising, not generic image CDAE |
| **MEDIUM** | PnP-BM3D + gradient | Classical+PnP | -- | Was on prior leaderboard version; should be retained for comparison |
| **LOW** | SpecFormer + gradient | Transformer | -- | Was on prior leaderboard version; should be retained |
| **LOW** | U-Net for Raman/Brillouin spectral denoising | 2024 | ~20 | Comparative study shows robustness advantages |

### Algorithm Gap Analysis

| Category | Have | Missing | Gap |
|----------|------|---------|-----|
| Classical peak fitting | SG-ALS | Lorentzian fit, Pseudo-Voigt, DHO model | 3 |
| Classical denoising | -- | MER, Wavelet Analysis | 2 |
| CNN-based | Cascade-UNet, CDAE | 1D spectral CDAE, U-Net denoiser | 2 |
| Plug-and-play | PnP-DnCNN | PnP-BM3D (removed from leaderboard) | 1 |
| Transformer | -- | SpecFormer (removed from leaderboard) | 1 |
| Diffusion models | -- | PI-DDPM | 1 |
| Physics-informed DL | -- | PSRN (unsupervised, physics-enhanced) | 1 |
| Distributed sensing DL | -- | DNN-BOTDA | 1 |

**Total gap: 12 algorithms missing across 8 categories**

---

## 5. Improvement Suggestions

### 5.1 Dataset Improvements

1. **REPLACE RRUFF WITH ACTUAL BRILLOUIN DATA (CRITICAL)**
   - RRUFF is a Raman database, not Brillouin
   - Raman and Brillouin probe different phonon branches at different frequency scales
   - Using Raman as ground truth for Brillouin reconstruction is physically incorrect
   - Options: (a) contact Zhang Lab / Scarcelli group for experimental VIPA data, (b) generate synthetic Brillouin gain spectra with known Lorentzian parameters, (c) use published distributed fiber BOTDA datasets

2. **BUILD THE DATASET (CRITICAL)**
   - No local data files exist at all
   - Need to generate/download HDF5 files for all 3 tiers
   - Minimum: synthetic Brillouin spectral maps with controlled Lorentzian parameters

3. **Increase spatial resolution from 64x64**
   - Real Brillouin VIPA spectrograms are typically 200-500 pixels across
   - 64x64 is unrealistically small and may not capture spectral fine structure
   - Suggest 256x256 minimum for meaningful spectral features

4. **Add spectral dimension**
   - Current data is 2D spatial [64, 64] but Brillouin microscopy is inherently spectral
   - Should include spectral axis (e.g., [64, 64, N_spectral] or [N_spatial, N_spectral])
   - The VIPA spectrometer produces 2D patterns where one axis encodes frequency

5. **Tier-specific ranges need consistency**
   - Hidden tier b_s [-7.0, 23.0] has a narrower lower bound than dev [-12.0, 18.0]
   - Should follow monotonic widening: public < dev < hidden

### 5.2 Algorithm Testing Improvements

6. **Add Lorentzian least-squares fitting as primary baseline**
   - Listed as `default_solver` in YAML but never benchmarked
   - This is THE foundational method in Brillouin spectroscopy
   - Every Brillouin paper compares against Lorentzian fit

7. **Add MER and Wavelet Analysis (Fiore et al. 2020)**
   - First dedicated Brillouin SNR enhancement study
   - ~60 citations, specific to Brillouin microspectroscopy
   - Reference implementation should be straightforward

8. **Add PSRN physics-enhanced neural network (2025)**
   - State-of-the-art unsupervised approach
   - Embeds Brillouin gain spectrum physics as a prior
   - Directly applicable to this benchmark

9. **Restore removed algorithms**
   - SpecFormer + gradient and PnP-BM3D + gradient were on previous leaderboard
   - Should retain for longitudinal comparison, not silently remove

10. **Add pseudo-Voigt and DHO fitting**
    - Standard spectral analysis methods used in every Brillouin lab
    - Essential for viscoelastic and heterogeneous biological samples

### 5.3 Benchmark Infrastructure Improvements

11. **Replace microscopy_psf forward model with Brillouin-specific model**
    - PSF convolution is incorrect for Brillouin spectroscopy
    - Need: VIPA etalon transfer function, Lorentzian lineshape, elastic scattering leakage
    - The `has_dedicated_operator: true` flag is set but the operator uses generic PSF

12. **Sync YAML ranges with webpage ranges**
    - YAML: single global range
    - Webpage: per-tier ranges
    - Must match to allow reproducible benchmarking

13. **Define PSNR_norm on webpage**
    - Scoring formula uses PSNR_norm but never defines it
    - Composite score cannot be reproduced

14. **Update learning materials**
    - 01_physics_fundamentals.md describes generic photon PSF optics
    - Should cover: Brillouin scattering physics, phonon interaction, VIPA etalon principle, frequency shift measurement
    - 02_forward_model.md should derive the Brillouin spectral forward model, not generic convolution

15. **Add Brillouin-specific metrics**
    - Brillouin shift error (MHz) -- the primary quantity of interest
    - Linewidth error (MHz) -- related to viscoelastic properties
    - Elastic-to-Brillouin ratio -- measures leakage suppression quality

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Replace RRUFF Raman source with actual Brillouin data | Dataset team | TODO |
| CRITICAL | Build HDF5 dataset files for all 3 tiers | Dataset team | TODO |
| CRITICAL | Implement Brillouin-specific forward model (replace microscopy_psf) | Physics team | TODO |
| CRITICAL | Add Lorentzian least-squares fitting baseline | Algorithm team | TODO |
| CRITICAL | Define PSNR_norm formula on webpage | Main server | TODO |
| HIGH | Add MER + Wavelet Analysis (Fiore et al. 2020) | Algorithm team | TODO |
| HIGH | Add PSRN physics-enhanced neural network (2025) | Algorithm team | TODO |
| HIGH | Sync YAML mismatch ranges with webpage per-tier ranges | Config team | TODO |
| HIGH | Restore SpecFormer and PnP-BM3D to leaderboard | Main server | TODO |
| HIGH | Rewrite 01_physics_fundamentals.md for Brillouin physics | Learn team | TODO |
| HIGH | Add pseudo-Voigt and DHO fitting methods | Algorithm team | TODO |
| MEDIUM | Increase data resolution from 64x64 to 256x256 | Dataset team | TODO |
| MEDIUM | Add spectral dimension to data format | Dataset team | TODO |
| MEDIUM | Add Brillouin-specific metrics (shift error, linewidth error) | Metrics team | TODO |
| MEDIUM | Fix tier mismatch range monotonicity | Config team | TODO |
| MEDIUM | Register lorentz_fit in solver registry (listed as default but missing) | Infrastructure | TODO |
| LOW | Add gallery with VIPA spectrogram reconstructions | Main server | TODO |
| LOW | Fix placeholder links (/compete, /contribute) | Main server | TODO |
| LOW | Add algorithm references and citations | Main server | TODO |

---

## Appendix: Key References

- Scarcelli, G., Yun, S.H. "Confocal Brillouin microscopy for three-dimensional mechanical imaging." Nature Photonics 2.1 (2008): 39-43.
- Scarcelli, G., et al. "Noncontact three-dimensional mapping of intracellular hydromechanical properties by Brillouin microscopy." Nature Methods 12.12 (2015): 1132-1134.
- Fiore, A., Zhang, J., Shao, P., Yun, S.H., Scarcelli, G. "SNR enhancement in Brillouin microspectroscopy using spectrum reconstruction." Biomed. Opt. Express 11.2 (2020): 1020-1029.
- Prevedel, R., Diz-Munoz, A., Ruber, G., Grec, K. "Brillouin microscopy: an emerging tool for mechanobiology." Nature Methods 16.10 (2019): 969-977.
- Zhang, J., Scarcelli, G. "Mapping mechanical properties of biological materials via an add-on Brillouin module to confocal microscopes." Nature Protocols 16.2 (2021): 1251-1275.
- Lafuente, B., Downs, R.T., Yang, H., Stone, N. "The power of databases: the RRUFF project." Highlights in Mineralogical Crystallography (2016): 1-30.
- Wu, H., et al. "Unsupervised super-spatial-resolution Brillouin frequency shift extraction based on physical enhanced spatial resolution neural network." arXiv:2503.00506 (2025).
- Badon, A., et al. "Microscopy image reconstruction with physics-informed denoising diffusion probabilistic model." Commun. Eng. 3 (2024): 45.
- Nikolova, L., et al. "Brillouin microscopy." Nature Rev. Methods Primers 4 (2024): 8.
- Mattana, S., et al. "Label-free Brillouin endo-microscopy for quantitative 3D imaging." Commun. Biol. 7 (2024): 479.

---

*Comprehensive 6-point review on 2026-03-03. Covers: page errors, local dataset verification, source quality, algorithm coverage, improvement suggestions, and action items. Key finding: the benchmark uses a Raman database (RRUFF) as ground truth for a Brillouin modality, has no local dataset files, and employs a generic PSF convolution forward model instead of a Brillouin-specific spectral model. 12 domain-specific algorithms are missing from the leaderboard.*