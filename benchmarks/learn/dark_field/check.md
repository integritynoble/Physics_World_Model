# Comprehensive Benchmark QA Check — dark_field

**URL:** https://pwm.platformai.org/benchmark/dark_field
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
| HIGH     | 7     |
| MEDIUM   | 10    |
| LOW      | 8     |

### HIGH Severity

**H1. Negative values for stray light mismatch parameter**
The webpage spec ranges show negative stray-light values across tiers:
- `stray_light`: -1.0 to 2.0 (Public), -1.2 to 1.8 (Dev), -0.7 to 2.3 (Hidden)
Stray light is an additive intensity, and negative stray light is physically meaningless in dark-field microscopy. The local YAML config shows range [0.0, 5.0] which is physically correct.
**Fix:** Update webpage spec ranges to non-negative values matching the local config or provide physical justification for negative values (e.g., dark-current subtraction).

**H2. PSNR_norm undefined in scoring formula**
Scoring formula is `0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * (1 - ||y - Hx_hat|| / ||y||)` but:
- Normalization method for PSNR not defined (min-max? reference baseline? dynamic range?)
- Data range for SSIM not specified (uint8 [0,255] vs float [0,1])
- Norm type in data-fidelity term not stated (L2? Frobenius?)
**Fix:** Define PSNR_norm = (PSNR - PSNR_min) / (PSNR_max - PSNR_min) with explicit bounds, specify L2 norm and SSIM data_range.

**H3. Scattering angle range is zero in YAML but +/-10% on webpage**
- Local YAML config: `scattering_angle_range` range [0, 0] (fixed, no mismatch)
- Webpage benchmark doc: "scattering_angle_range: fixed -0.15 to 0.15 across all tiers"
- Modality benchmark doc (docs/modality_benchmarks/dark_field.md): "+/- 10%"
Three conflicting definitions for the same parameter across three locations.
**Fix:** Reconcile all three sources. If scattering angle is actively perturbed, update YAML; if fixed at zero, update webpage and docs.

**H4. Spec ranges do not monotonically increase in difficulty across tiers**
Webpage shows non-nesting condenser NA ratio ranges:
- `condenser_na_vs_objective_na_ratio`: 1.14-1.32 (Public), 1.128-1.308 (Dev), 1.158-1.338 (Hidden)
- Dev range (1.128-1.308) is wider than Public (1.14-1.32) on the low end but narrower on the high end
- Hidden range (1.158-1.338) is narrower than Dev on the low end
These non-nesting ranges make tier difficulty ambiguous.
**Fix:** Ensure strict nesting: Public subset of Dev subset of Hidden, with Hidden having the widest and most extreme ranges.

**H5. HDF5 submission format undocumented**
Page says "submit Reconstructed signals and corrected spec as HDF5" but specifies:
- No HDF5 key/group structure
- No data types (float32 vs float64)
- No array shapes or expected dimensions
- No example file provided
**Fix:** Add a "Submission Format" section with exact HDF5 schema, including key names, dtypes, and shapes.

**H6. Forward operator H delivery format unspecified**
States input includes "ideal forward operator (H)" but does not specify whether H is:
- Dense matrix (infeasible for large images)
- Sparse matrix representation
- Functional operator (Python callable)
- Implicit via PSF kernel + detector model
**Fix:** Specify that H is delivered as an implicit operator defined by the C-->D DAG (convolution + detector).

**H7. Leaderboard method names inconsistent with literature**
Leaderboard lists "PolScope-Former + gradient" and "FLIM-Net + gradient" but:
- PolScope-Former is a polarization microscopy method, not dark-field
- FLIM-Net is a fluorescence lifetime imaging method, not dark-field
- These appear to be auto-generated placeholder names from a template
**Fix:** Replace with actual dark-field reconstruction methods or clarify that these are adapted architectures.

### MEDIUM Severity

| ID  | Issue |
|-----|-------|
| M1  | Noise model incomplete -- no SNR/sigma specification; YAML has read_noise_e=5.0 but webpage omits it |
| M2  | Condenser NA nominal value (1.2) outside Public tier range (1.14-1.32) -- close but asymmetric |
| M3  | Pixel size (6.5 um from YAML) not mentioned on webpage; critical for resolution context |
| M4  | PSF sigma (2.0 from YAML) not mentioned on webpage; affects expected resolution |
| M5  | 11 spec primitives listed on webpage but DAG only uses 2 (C, D); unclear which primitives apply |
| M6  | BioSR dataset reference (Qiao et al., Nat. Methods 2024) needs DOI and direct download link |
| M7  | "3 evaluation tiers with 5 scenes each" = 15 scenes total, but expanded config shows 252 total cases across B1-B4 |
| M8  | Mismatch levels M0-M4 defined in expanded config but absent from webpage |
| M9  | Cell phantom generator referenced in YAML (synthetic_generator: cell_phantom) but not described on webpage |
| M10 | Image sizes range from 128x128 to 1024x1024 in expanded config but webpage only shows 64x64 |

### LOW Severity

| ID  | Issue |
|-----|-------|
| L1  | No alt-text on gallery images |
| L2  | Mixed notation: "H" vs "forward operator" used interchangeably |
| L3  | No figure caption on DAG diagram (C --> D) |
| L4  | Compete and Contribute pages are placeholder stubs |
| L5  | Best-quality solver listed as CARE (Weigert et al. 2018) but no pre-trained weights path |
| L6  | Richardson-Lucy listed with params="0" in YAML (should be iteration count or regularization) |
| L7  | No explicit wavelength or illumination spectrum specified for the dark-field modality |
| L8  | maturity: M0 in YAML but webpage shows evaluation across all difficulty tiers |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier   | Directory                                          | Status |
|--------|----------------------------------------------------|--------|
| Public | datasets/benchmark/dark_field/public/            | NOT FOUND |
| Dev    | datasets/benchmark/dark_field/dev/               | NOT FOUND |
| Hidden | datasets/benchmark/dark_field/hidden/            | NOT FOUND |

**The entire datasets/benchmark/dark_field/ directory does not exist locally.**

### Configuration Files Found

| File | Size | Status |
|------|------|--------|
| benchmarks/configs/dark_field.yaml | 91 lines | Present, well-formed |
| benchmarks/expanded_configs/dark_field_expanded.yaml | 94 lines | Present, well-formed |
| docs/modality_benchmarks/dark_field.md | 73 lines | Present, well-formed |

### YAML Config Key Parameters

| Parameter | Value | Consistency |
|-----------|-------|-------------|
| x_shape | [64, 64] | Conflicts with expanded config (128-1024) |
| y_shape | [64, 64] | Consistent with x_shape (same space for dark-field) |
| sigma | 2.0 | PSF Gaussian blur |
| read_noise_e | 5.0 electrons | Detector noise model |
| pixel_size_um | 6.5 um | Typical for sCMOS cameras |
| forward_model_type | linear_operator | Correct for dark-field convolution |
| has_dedicated_operator | true | Dedicated forward operator registered |
| data_source.fallback | generated | Falls back to synthetic cell phantoms |
| data_source.dataset_id | '' (empty) | No real dataset configured |
| data_source.citation | '' (empty) | No citation configured |

### Expanded Config Key Parameters

| Parameter | Value |
|-----------|-------|
| Noise levels | Clean (60 dB), Low (40 dB), Medium (30 dB), High (20 dB) |
| Mismatch levels | M0 (nominal), M1 (single param), M2 (3+ params), M3 (real), M4 (adversarial) |
| Total benchmark cases | 252 (B1: 12, B2: 80, B3: 80, B4: 80) |
| Image size variants | 128x128, 256x256, 512x512, 1024x1024 |

### Dataset Integrity Assessment: **FAIL -- No local dataset files exist**

---

## 3. Public Dataset Source Assessment

### Current Source

**Webpage claims:** BioSR dataset (Qiao et al., Nat. Methods 2024)
- Biological super-resolution dataset with paired low/high-resolution fluorescence images
- Originally designed for structured illumination microscopy (SIM) and other super-resolution modalities
- Contains 12 biological structures (CCPs, ER, F-actin, microtubules, etc.)

**Local config claims:** No dataset configured
- data_source.dataset_id: '' (empty)
- data_source.dataset_url: '' (empty)
- data_source.citation: '' (empty)
- data_source.fallback: generated (uses synthetic cell phantoms)

### Assessment of Source Quality

| Criterion | Rating | Notes |
|-----------|--------|-------|
| **Webpage dataset matches modality?** | NEEDS REVIEW | BioSR is primarily fluorescence/SIM, not dark-field microscopy. The samples may serve as generic biological phantoms, but dark-field contrast (scattering) differs fundamentally from fluorescence contrast (emission). |
| **BioSR well-known?** | GOOD | Published in Nature Methods 2024, widely cited in computational microscopy community |
| **BioSR accepted by professors/PhDs?** | GOOD for fluorescence | Well-established for SIM reconstruction; less established for dark-field specifically |
| **Local dataset available?** | FAIL | No HDF5 files or any data files present locally |
| **Config-webpage alignment?** | FAIL | Webpage references BioSR; local config has no dataset configured, falls back to synthetic generation |
| **Dev tier protection?** | GOOD (if synthetic) | Cell phantom synthetic generator prevents reverse-engineering |
| **Hidden tier protection?** | GOOD (if synthetic) | Server-side evaluation prevents data access |

### Recommendations for Public Tier

1. **Use a dedicated dark-field microscopy dataset:**
   - **OpenDarkField** (if available): Native dark-field microscopy images with known illumination parameters
   - **BioSR with dark-field simulation**: Apply dark-field forward model to BioSR biological structures (justified if scattering properties are realistic)

2. **Alternative established datasets for dark-field-like imaging:**
   - **BBBC (Broad Bioimage Benchmark Collection)**: Standardized biological image sets with various contrast mechanisms
   - **Cell Image Library**: Curated microscopy images including dark-field examples
   - **MNIST-like phantoms**: For initial synthetic validation (already planned via cell_phantom generator)

3. **For real dark-field credibility:**
   - Partner with a microscopy lab to acquire real dark-field images with calibrated forward models
   - Nikon/Olympus/Zeiss dark-field documentation often includes sample images with known NA configurations
   - Published dark-field datasets from materials science (nanoparticle imaging) could supplement biological samples

4. **Resolve webpage vs local config discrepancy:**
   - Either download and integrate BioSR locally, or update the webpage to reflect the actual synthetic data source
   - Current state creates confusion about what participants actually receive

---

## 4. Algorithm Coverage Assessment

### Currently Tested (Webpage Leaderboard)

| # | Algorithm | Score | Type | Notes |
|---|-----------|-------|------|-------|
| 1 | Restormer + gradient | 0.734 | Transformer-based DL | Zamir et al., CVPR 2022; top performer |
| 2 | CARE + gradient | 0.711 | U-Net DL | Weigert et al., Nat. Methods 2018 |
| 3 | Richardson-Lucy + gradient | 0.641 | Classical iterative | Richardson 1972 / Lucy 1974 |
| 4 | PnP-BM3D + gradient | -- | Plug-and-play | Listed in earlier check.md but score not on webpage |

Note: The earlier auto-generated check.md listed different method names (PolScope-Former, FLIM-Net, PnP-BM3D, Phasor-FLIM), suggesting the leaderboard may have been updated between checks.

### PWM Solver Registry

| Solver | Function | GPU | Status |
|--------|----------|-----|--------|
| Richardson-Lucy | pwm_core.recon.richardson_lucy.run_richardson_lucy | No | Registered (traditional_cpu) |
| CARE | pwm_core.recon.care_unet.care_restore_2d | Yes | Registered (best_quality) |

### Missing Famous/Recent Algorithms (MUST ADD)

| Priority | Algorithm | Year | Citation | Why Important |
|----------|-----------|------|----------|---------------|
| **CRITICAL** | Wiener Deconvolution | 1949 | Wiener, Extrapolation (~10,000+ citations) | Foundational linear deconvolution baseline; trivial to implement |
| **CRITICAL** | Total Variation Deconvolution | 2004 | Chambolle, J. Math. Imaging Vision (~4,000 citations) | Standard non-blind deconvolution, sharp edge preservation |
| **CRITICAL** | Deep Learning-Assisted Dark-Field (DL-DF) | 2024 | PMC:11638943 | Super-resolution label-free imaging via plasmonic dark-field with DL |
| **HIGH** | ADMM / Split-Bregman | 2009 | Boyd et al., Found. Trends ML (~15,000 citations) | General-purpose proximal splitting; handles complex forward models |
| **HIGH** | U-Net (generic) | 2015 | Ronneberger et al., MICCAI (~90,000 citations) | Standard DL baseline for image-to-image regression |
| **HIGH** | NAFNet | 2022 | Chen et al., ECCV (~1,000 citations) | Simple yet powerful baseline, nonlinear activation free |
| **HIGH** | Dark-Field X-ray with AI | 2025 | MRS Comm. (Springer, 2025) | Physics-informed AI for dislocation characterization in DFXM |
| **MEDIUM** | Structured Illumination DFXM | 2025 | Comm. Physics (Nature, 2025) | 3D dark-field imaging without sample rotation |
| **MEDIUM** | DRUNet / PnP-DRUNet | 2021 | Zhang et al., TPAMI (~2,000 citations) | Plug-and-play with deep denoiser, no retraining needed |
| **MEDIUM** | DPIR (Deep Plug-and-Play) | 2021 | Zhang et al., CVPR (~1,500 citations) | Half-quadratic splitting with DRUNet denoiser |
| **MEDIUM** | Score-based Diffusion | 2022 | Song et al., ICLR (~3,000 citations) | Generative prior for inverse problems, state-of-the-art |
| **LOW** | Lucy-Richardson with TV regularization | -- | -- | Enhanced classical baseline with edge-preserving regularization |
| **LOW** | Sparse deconvolution (L1) | -- | -- | Classical CS-inspired approach for sparse specimen features |

### Algorithm Gap Analysis

| Category | Have | Missing | Gap |
|----------|------|---------|-----|
| Classical deconvolution | Richardson-Lucy | Wiener, TV deconvolution | 2 methods |
| Optimization-based | -- | ADMM, Split-Bregman | 2 methods |
| U-Net architectures | CARE | vanilla U-Net, NAFNet | 2 methods |
| Transformer-based | Restormer | SwinIR, Uformer | 2 methods |
| Plug-and-play | PnP-BM3D (maybe) | PnP-DRUNet, DPIR | 2 methods |
| Dark-field specific DL | -- | DL-DF, AI-DFXM | 2 methods |
| Generative/diffusion | -- | Score-based diffusion | 1 method |
| Blind deconvolution | -- | Any blind method | 1+ methods |

**Total gap: 14+ algorithms missing across 8 categories**

---

## 5. Improvement Suggestions

### 5.1 Dataset Improvements

1. **Create local dataset directory and populate HDF5 files**
   - datasets/benchmark/dark_field/ does not exist
   - This is the most critical gap: the benchmark has configs and webpage but no actual data locally
   - Generate synthetic data using the cell_phantom generator defined in YAML as immediate fallback
   - Target at least 5 scenes per tier (public/dev/hidden) matching the webpage claim

2. **Resolve BioSR vs synthetic data source conflict**
   - Webpage references BioSR (Qiao et al., 2024) but local config has no dataset configured
   - Either: (a) download BioSR and apply dark-field forward model, or (b) update webpage to state "synthetic cell phantoms"
   - If using BioSR, justify the fluorescence-to-dark-field domain transfer

3. **Acquire real dark-field microscopy data for public tier**
   - Contact microscopy labs for calibrated dark-field images with known condenser/objective NA configurations
   - Even 5-10 real images would significantly boost benchmark credibility
   - Include samples at different NA ratios to validate the mismatch parameter

4. **Reconcile image size discrepancies**
   - Base YAML: 64x64 (very small, may not capture relevant structures)
   - Expanded config: 128x128 to 1024x1024 (more realistic)
   - Webpage: unclear
   - Recommend: standardize on 256x256 for dev/hidden, allow 512x512 for advanced tier

5. **Add biological diversity to phantom generator**
   - Current: generic cell_phantom generator
   - Suggested: include nanoparticle suspensions, fibrous structures, and cell clusters
   - Dark-field excels at scattering-dominant samples; phantoms should reflect this

### 5.2 Algorithm Testing Improvements

6. **Add Wiener deconvolution immediately**
   - Optimal linear estimator, trivial to implement
   - Essential lower-bound analytical baseline for all comparisons
   - Can be computed in under 1 second for 256x256 images

7. **Add Total Variation deconvolution**
   - Standard convex optimization baseline with edge-preserving properties
   - Well-suited for dark-field where scatterers have sharp boundaries
   - Available in scikit-image (skimage.restoration.denoise_tv_chambolle)

8. **Add a blind deconvolution method**
   - Current benchmark assumes known forward operator H
   - Real dark-field systems have PSF uncertainty; blind methods test robustness
   - Consider: BDRL (Ren et al., 2020) or SelfDeblur (Ren et al., 2020)

9. **Test dark-field-specific deep learning methods**
   - Recent work (PMC:11638943, 2024) demonstrates DL-assisted plasmonic dark-field for super-resolution
   - AI-enhanced DFXM (MRS Comm., 2025) shows physics-informed approaches for dark-field analysis
   - These domain-specific methods may outperform generic restoration networks

10. **Run all solvers consistently across all tiers**
    - Current leaderboard has only 3-4 methods
    - All methods should report scores on public, dev, and hidden tiers
    - Show per-tier degradation to validate that mismatch difficulty scales correctly

### 5.3 Benchmark Infrastructure Improvements

11. **Fix leaderboard method names**
    - Earlier check showed PolScope-Former, FLIM-Net (wrong modality); current page shows Restormer, CARE, R-L
    - Ensure all method names are accurate and consistent across checks

12. **Add explicit forward model equation to webpage**
    - Current: only DAG notation "C --> D"
    - Needed: y = D(C(x; PSF(NA_c, NA_o)) + s) + n where s=stray light, n=read noise
    - Include PSF model: Airy disk or Gaussian with NA-dependent width

13. **Define all mismatch levels on webpage**
    - Expanded config defines M0-M4 mismatch hierarchy but webpage only shows tier-specific ranges
    - Users need to understand the systematic difficulty scaling

14. **Add per-sample breakdown in leaderboard**
    - Current: aggregate scores only
    - Suggested: show per-sample PSNR/SSIM or at least mean +/- std
    - Enables statistical significance testing between algorithms

15. **Increase total benchmark cases**
    - Current expanded config: 252 total (B1:12, B2:80, B3:80, B4:80)
    - This is a reasonable total, but actual data needs to be generated and stored locally

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Create datasets/benchmark/dark_field/ and generate HDF5 data for all tiers | Dataset team | TODO |
| CRITICAL | Resolve BioSR vs synthetic source conflict between webpage and config | Main server | TODO |
| CRITICAL | Fix scattering_angle_range: reconcile YAML (0,0), webpage (0.15), and docs (+/-10%) | Main server | TODO |
| CRITICAL | Fix stray_light spec ranges: remove physically meaningless negative values | Main server | TODO |
| CRITICAL | Define PSNR_norm formula explicitly on webpage | Main server | TODO |
| HIGH | Add Wiener deconvolution baseline to leaderboard | Algorithm team | TODO |
| HIGH | Add Total Variation deconvolution to leaderboard | Algorithm team | TODO |
| HIGH | Add explicit forward model equation to webpage | Main server | TODO |
| HIGH | Ensure spec ranges nest properly across tiers (Public < Dev < Hidden) | Main server | TODO |
| HIGH | Add HDF5 submission format specification | Main server | TODO |
| HIGH | Reconcile x_shape discrepancy: 64x64 (YAML) vs 128-1024 (expanded) vs webpage | Main server | TODO |
| HIGH | Fix leaderboard method names (remove PolScope-Former/FLIM-Net if wrong modality) | Main server | TODO |
| MEDIUM | Acquire real dark-field microscopy data for public tier credibility | Dataset team | TODO |
| MEDIUM | Add dark-field-specific DL methods (DL-DF, AI-DFXM) to algorithm suite | Algorithm team | TODO |
| MEDIUM | Add blind deconvolution method to test PSF-uncertainty robustness | Algorithm team | TODO |
| MEDIUM | Define all mismatch levels (M0-M4) on webpage | Main server | TODO |
| MEDIUM | Add per-sample metric breakdown to leaderboard | Main server | TODO |
| MEDIUM | Add PnP-DRUNet / DPIR to algorithm suite (no retraining needed) | Algorithm team | TODO |
| LOW | Add diffusion-based reconstruction (score-based prior) | Algorithm team | TODO |
| LOW | Add biological diversity to cell phantom generator (nanoparticles, fibers) | Dataset team | TODO |
| LOW | Specify illumination wavelength/spectrum on webpage | Main server | TODO |
| LOW | Fix placeholder links (/compete, /contribute) | Main server | TODO |
| LOW | Add gallery image alt-text and figure captions | Main server | TODO |

---

## Appendix: Key References

- Richardson, W.H. "Bayesian-based iterative method of image restoration." JOSA 62.1 (1972): 55-59.
- Lucy, L.B. "An iterative technique for the rectification of observed distributions." AJ 79 (1974): 745.
- Wiener, N. Extrapolation, Interpolation, and Smoothing of Stationary Time Series. MIT Press (1949).
- Chambolle, A. "An algorithm for total variation minimization and applications." J. Math. Imaging Vision 20.1-2 (2004): 89-97.
- Ronneberger, O., Fischer, P., Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI (2015).
- Weigert, M., et al. "Content-aware image restoration: pushing the limits of fluorescence microscopy." Nature Methods 15.12 (2018): 1090-1097.
- Boyd, S., et al. "Distributed Optimization and Statistical Learning via the ADMM." Found. Trends ML 3.1 (2011): 1-122.
- Zhang, K., Li, Y., Zuo, W., et al. "Plug-and-Play Image Restoration with Deep Denoiser Prior." IEEE TPAMI (2021).
- Zamir, S.W., et al. "Restormer: Efficient Transformer for High-Resolution Image Restoration." CVPR (2022).
- Chen, L., et al. "Simple Baselines for Image Restoration (NAFNet)." ECCV (2022).
- Qiao, C., et al. "BioSR: Evaluation and benchmarking of biological image super-resolution." Nature Methods (2024).
- PMC:11638943. "Deep Learning Assisted Plasmonic Dark-Field Microscopy for Super-Resolution Label-Free Imaging." (2024).
- MRS Communications (Springer, 2025). "Advances in artificial intelligence-based approaches to enhance dark field X-ray microscopy analysis."
- Nature Communications Physics (2025). "Dark-field X-ray microscopy with structured illumination for three-dimensional imaging."
- Song, Y., et al. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR (2021).

### Recent Literature (2024-2025 Web Search Results)

- [Advances in AI-based approaches for dark field X-ray microscopy](https://link.springer.com/article/10.1557/s43579-025-00860-4) -- MRS Comm., 2025
- [Trusting deep learning-based reconstruction for X-ray microscopy](https://analyticalscience.wiley.com/content/article-do/trusting-deep-learning-based-reconstruction-x-ray-microscopy) -- Wiley, 2025
- [Deep Learning-Assisted Weak Beam Identification in DFXM](https://arxiv.org/abs/2509.05017) -- arXiv, 2025
- [Dark-field X-ray microscopy with structured illumination for 3D imaging](https://www.nature.com/articles/s42005-025-01952-2) -- Comm. Physics, 2025
- [DL-assisted plasmonic dark-field for super-resolution](https://pmc.ncbi.nlm.nih.gov/articles/PMC11638943/) -- PMC, 2024
- [Virtual differential phase-contrast and dark-field via deep learning](https://aiche.onlinelibrary.wiley.com/doi/full/10.1002/btm2.10494) -- Bioeng. & Transl. Med., 2023
- [Revolutionizing optical imaging: computational imaging via deep learning](https://www.spiedigitallibrary.org/journals/photonics-insights/volume-4/issue-2/R03/Revolutionizing-optical-imaging-computational-imaging-via-deep-learning/10.3788/PI.2025.R03.full) -- SPIE Photonics Insights, 2025
- [Enhanced detection of threat materials by dark-field X-ray imaging + DNN](https://www.nature.com/articles/s41467-022-32402-0) -- Nature Comm., 2022

---

*Comprehensive 6-point review on 2026-03-03. Covers: page errors, local dataset verification, source quality, algorithm coverage, improvement suggestions, and action items.*