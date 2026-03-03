# Comprehensive Benchmark QA Check -- Active Thermography

**URL:** https://pwm.platformai.org/benchmark/active_thermography
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
| MEDIUM   | 8     |
| LOW      | 4     |

### HIGH Severity

**H1. DAGM 2007 is an OPTICAL INSPECTION dataset, not a thermography dataset**
- Webpage cites "DAGM 2007 (Wieler & Hahn, DAGM 2007)" as the dataset source
- DAGM 2007 is a synthetic benchmark for weakly supervised **optical** defect detection on textured surfaces (visible-light images, 10 classes of textured backgrounds with defects)
- It was created by Bosch Research for the DAGM symposium competition on industrial **optical** inspection
- It has **no infrared, thermal, or thermographic content whatsoever**
- Using an optical inspection dataset for an active thermography benchmark is a fundamental domain mismatch
**Fix:** Replace DAGM 2007 with a genuine thermography dataset. Options: (a) CFRP pulsed thermography dataset (Marani et al., MDPI Applied Sciences 2023), (b) the publicly available PVC specimen pulsed thermography dataset (Bang et al., Applied Sciences 2023), (c) synthetic thermal diffusion data from a proper heat equation PDE solver.

**H2. Forward model DAG "P --> D" missing thermal diffusion physics**
- Active thermography requires solving the time-domain heat diffusion equation (thermal PDE)
- The DAG shows only generic "P (Propagation) --> D (Detector)" with Fresnel/Rayleigh-Sommerfeld kernels listed
- Fresnel and Rayleigh-Sommerfeld are **wave optics** propagation kernels, completely irrelevant to thermal diffusion
- No temporal integration primitive, no thermal diffusivity (k, rho, c_p) parameters
- No excitation waveform specification (pulsed, lock-in, step-heating)
- The category_module is listed as `microscopy_psf` which is incorrect for thermography
**Fix:** Replace DAG with thermal-physics-specific pipeline: Excitation --> Heat PDE (thermal diffusivity) --> Surface T(x,y,t) --> Planck emission --> IR Detector. Use a heat-equation solver, not a PSF convolution module.

**H3. PSNR_norm undefined and scoring formula notation inconsistent**
- Scoring: `0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * (1 - ||y - H_hat * x_hat|| / ||y||)`
- PSNR_norm normalization method and range not specified (what is the max PSNR used for normalization?)
- Formula uses H_hat (estimated operator) but description says "ideal forward operator H"
- No justification for the 40/40/20 weighting split
- Consistency term `(1 - ||y - H_hat * x_hat|| / ||y||)` depends on forward model fidelity, but the forward model itself is incorrect (see H2)
**Fix:** Define PSNR_norm bounds explicitly, use H consistently, justify weights, and ensure consistency term uses the correct thermal forward model.

**H4. References incomplete -- no DOIs or verifiable citations**
- "Wieler & Hahn, DAGM 2007" has no DOI, URL, or full title
- LSTM-NDT and DefectNet papers on leaderboard have no arXiv IDs, venue, or DOI
- PnP-ADMM and TSR references are also missing
**Fix:** Provide full citations with DOIs for all referenced works.

**H5. Wavelength range listed as "0 -- 0 nm" in physics fundamentals**
- The learn docs (01_physics_fundamentals.md) lists wavelength range as "0 -- 0 nm"
- Active thermography operates in the mid-wave IR (3--5 um) or long-wave IR (8--14 um) bands
- This is a placeholder that was never filled in
**Fix:** Set wavelength range to 3000--14000 nm (3--14 um) or specify MWIR/LWIR band explicitly.

### MEDIUM Severity

| ID | Issue | Fix |
|----|-------|-----|
| M1 | Cross-tier ranking reversal unexplained: PnP-ADMM surpasses DefectNet on Dev but ranks lower on Public. No analysis of why. | Add robustness discussion for each algorithm. |
| M2 | Dev tier says "Blind -- no ground truth" but PSNR/SSIM require ground truth. Server-side scoring not explicitly documented. | State that scoring is server-side with held-back ground truth. |
| M3 | Mismatch parameter ranges on webpage (e.g., emissivity 0.943--0.97) differ from YAML config (0.85--1.0). Webpage shows the per-tier actual drawn values, but this is not clarified. | Add a note explaining that webpage shows realized per-tier ranges and YAML shows the full prior range. |
| M4 | Only 3 scenes per tier (9 total) is extremely small for a robust benchmark. Statistical significance of PSNR/SSIM differences between methods is questionable. | Increase to at least 20 scenes per tier, or add confidence intervals to leaderboard scores. |
| M5 | HDF5 dataset schema undocumented: no key names, dimensions, dtypes, noise model, or example loading code on the benchmark page. | Add schema table and Python loading snippet to the benchmark page. |
| M6 | Hidden tier Docker submission specs missing: no base image, entrypoint signature, runtime limits, or output format. | Document submission API and constraints. |
| M7 | Missing defect-level evaluation metrics: PSNR/SSIM measure pixel fidelity but not defect detection accuracy. Active thermography is fundamentally about finding defects. | Add precision, recall, IoU, or Dice coefficient for defect regions. |
| M8 | Config says `forward_model_type: nonlinear_operator` with `category_module: microscopy_psf` and `default_solver: thermal_diffusivity_inversion` but the solver name implies thermal physics that is not reflected in the forward model module. This is internally inconsistent. | Align forward model module, solver, and physics category. |

### LOW Severity

| ID | Issue | Fix |
|----|-------|-----|
| L1 | Gallery JavaScript: `selectGalleryScene()` panels do not render. | Debug JS or provide static fallback images. |
| L2 | Phase information ignored: lock-in thermography uses phase images for depth estimation, but only intensity-based metrics (PSNR/SSIM) are evaluated. | Add phase-based evaluation or clarify that only pulsed (transient) thermography is targeted. |
| L3 | No specification of whether this is pulsed thermography, lock-in thermography, or step-heating. The excitation type fundamentally changes the physics and algorithms. | Specify excitation type in the benchmark description. |
| L4 | `theta: {}` in config means no physics parameters are specified for the operator -- thermal conductivity, diffusivity, emissivity, and detector NETD are all absent. | Populate theta with physical parameters. |

---

## 2. Local Dataset Inspection

### File Inventory

**No local dataset files found.**

```
datasets/benchmark/active_thermography/   --> DOES NOT EXIST
```

The directory `datasets/benchmark/active_thermography/` is absent from the repository. No HDF5, CSV, JSON, or any other data files are present locally.

### Config Files Found

| File | Location | Status |
|------|----------|--------|
| Benchmark config | `benchmarks/configs/active_thermography.yaml` | EXISTS |
| Expanded config | `benchmarks/expanded_configs/active_thermography_expanded.yaml` | EXISTS |
| Learn docs (6 files) | `benchmarks/learn/active_thermography/` | EXISTS |

### Config Key Details

| Property | Value | Issue |
|----------|-------|-------|
| maturity | M0 | Lowest maturity level |
| data_source.dataset_id | (empty string) | No dataset linked |
| data_source.dataset_url | (empty string) | No URL |
| data_source.fallback | `generated` | Falls back to synthetic |
| data_source.synthetic_generator | `shepp_logan` | Shepp-Logan is a CT phantom, NOT a thermography phantom |
| reference_psnr | null | No reference baseline |
| expected_psnr_range | null | No expected range |
| theta | {} | Empty physics parameters |

### Expanded Config Details

| Property | Value |
|----------|-------|
| Image sizes | 128x128 (small), 256x256 (standard), 512x512 (large) |
| Noise levels | Clean (60 dB), Low (40 dB), Medium (30 dB), High (20 dB) |
| mismatch_params | [] (empty in expanded config!) |
| Total cases | B1: 12, B2: 60, B3: 60, B4: 60 = 192 grand total |
| Data sources | Generated only |

### Schema (from config, not verified from data)

| Dimension | Shape | Notes |
|-----------|-------|-------|
| Object (x) | [64, 64] | Base config; expanded config says 128/256/512 |
| Measurements (y) | [64, 64] | Same shape as object (no compression) |

### Dataset Integrity Assessment: **FAIL**

- No local data files exist
- Synthetic generator is `shepp_logan` (CT phantom, wrong domain)
- Dataset ID and URL are empty
- mismatch_params are empty in the expanded config (inconsistent with base config)
- Physics parameters (theta) are empty
- x_shape == y_shape (no measurement compression, which is unusual for an inverse problem)
- Maturity M0 indicates this modality has not progressed beyond initial scaffold

---

## 3. Public Dataset Source Assessment

### DAGM 2007: **POOR -- WRONG DOMAIN**

- Wieler & Hahn (2007), "Weakly Supervised Learning for Industrial Optical Inspection"
- Created by Bosch Research for the DAGM symposium 2007
- Contains 10 classes of synthetically textured surfaces with artificially generated defects
- Images are **visible-light optical** inspection images, not infrared thermograms
- 575 training + 575 test images per class (5,750 training + 5,750 test total)
- Available on Kaggle and Zenodo
- Widely cited (~500+ citations) for surface defect detection -- but in **optical inspection**, not thermography

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Well-known? | YES | DAGM 2007 is a standard benchmark in defect detection |
| Accepted by professors? | YES | Widely used in published research |
| Relevant to active thermography? | **NO** | Optical inspection, not IR/thermal imaging |
| Contains thermal physics? | **NO** | No heat equation, no IR emission, no thermal diffusion |
| Suitable replacement exists? | YES | See recommendations below |

### Recommended Replacement Datasets

| Dataset | Year | Description | Availability |
|---------|------|-------------|--------------|
| CFRP PT dataset (Marani et al.) | 2023 | Pulsed thermography on carbon fiber composites with calibrated defects | MDPI open data |
| PVC PT dataset (Bang et al.) | 2023 | Pulsed thermography with flat-bottom holes at known depths | MDPI Applied Sciences |
| Steel PT dataset (Vavilov & Burleigh) | 2015 | Standard steel specimens with known subsurface defects | Published with textbook |
| Synthetic thermal PDE | N/A | Generate via 3D heat equation solver (FEM) with known defects | Self-generated |

---

## 4. Algorithm Coverage Assessment

### Currently on Leaderboard: 4 algorithms

| Rank | Algorithm | Type | Overall | Public PSNR/SSIM | Dev PSNR/SSIM | Hidden PSNR/SSIM |
|------|-----------|------|---------|------------------|---------------|------------------|
| 1 | LSTM-NDT + gradient | DL temporal | 0.697 | 33.78 dB / 0.959 | 26.22 dB / 0.838 | 24.66 dB / 0.791 |
| 2 | DefectNet + gradient | DL spatial | 0.696 | 30.53 dB / 0.925 | 27.03 dB / 0.859 | 25.76 dB / 0.825 |
| 3 | PnP-ADMM + gradient | Iterative plug-and-play | 0.655 | 27.82 dB / 0.877 | 25.08 dB / 0.805 | 24.79 dB / 0.795 |
| 4 | TSR + gradient | Classical signal processing | 0.618 | 25.11 dB / 0.806 | 24.29 dB / 0.779 | 23.39 dB / 0.746 |

### Observations on Current Leaderboard

- All 4 methods use "+ gradient" suffix, suggesting a gradient-descent refinement step is appended
- LSTM-NDT ranks 1st overall but DefectNet is more robust (smaller public-to-hidden gap: 4.77 dB vs 9.12 dB)
- PnP-ADMM shows the smallest tier-to-tier degradation (3.03 dB drop from public to hidden)
- Only 4 algorithms is very sparse compared to CT (8 algorithms)
- "LSTM-NDT" and "DefectNet" do not appear to be established, widely-cited algorithm names in the literature

### Missing Famous/Recent Algorithms

| Priority | Algorithm | Year | Why It Should Be Included |
|----------|-----------|------|--------------------------|
| HIGH | TSR (standalone) | 2001 | Shepard et al., foundational thermography signal reconstruction; the benchmark's "TSR + gradient" may differ from standard TSR |
| HIGH | PCT (Principal Component Thermography) | 2002 | Rajic (2002), standard dimensionality-reduction approach, used in nearly every PT study |
| HIGH | PPT (Pulsed Phase Thermography) | 1996 | Maldague & Marinetti (1996), frequency-domain analysis, >1000 citations |
| HIGH | U-Net / Mask R-CNN | 2015/2017 | Universal DL baselines for segmentation; widely applied to thermography in 2023-2025 |
| HIGH | BiLSTM 3D reconstruction | 2023 | Automatic 3D defect reconstruction from PT sequences (Wan et al., 2023) |
| MEDIUM | DeepLabv3 + BiLSTM hybrid | 2025 | Spatial-temporal DL for CFRP thermography (Tandfonline 2025), F1=0.96, IoU=0.83 |
| MEDIUM | Sparse PCT / Sliding-Window PCT | 2021 | Enhanced PCT variants for improved contrast |
| MEDIUM | Matched Filter | Classical | Optimal linear filter for known defect thermal signature |
| MEDIUM | Virtual Wave Transform | 2014 | Transforms thermal data to wave-like propagation for depth estimation |
| LOW | Wiener deconvolution | Classical | Simple linear inverse filter baseline |
| LOW | Tikhonov-regularized inversion | Classical | Standard regularized linear inversion |
| LOW | GAN-based defect enhancement | 2022-2024 | Generative models for thermogram enhancement |

### Algorithm Gap Analysis

The benchmark has **4 algorithms** spanning only 3 categories (DL temporal, DL spatial, iterative, classical). Major gaps:

- **No standalone classical thermography algorithms** (TSR, PCT, PPT without gradient refinement)
- **No universal DL baselines** (U-Net, ResNet, Mask R-CNN)
- **No frequency-domain methods** (PPT, Fourier analysis)
- **No dimensionality-reduction methods** (PCT, sparse PCT)
- **No depth-estimation methods** (Virtual Wave Transform)
- **Missing standard linear baselines** (Wiener, Tikhonov, matched filter)
- **Algorithm names may be non-standard** ("LSTM-NDT" and "DefectNet" are not clearly identifiable from published literature)

**Total gap: 12+ missing algorithms across 6 categories. Coverage is POOR.**

---

## 5. Improvement Suggestions

### 5.1 Dataset (Critical Priority)

1. **Replace DAGM 2007 with a genuine thermography dataset.** DAGM 2007 is optical inspection (visible light), not infrared thermography. This is a domain mismatch that invalidates the entire benchmark. Use a real pulsed thermography dataset or generate synthetic data from a heat equation solver.
2. **Replace shepp_logan synthetic generator** with a thermal-diffusion-based phantom generator that models subsurface defects (voids, delaminations, inclusions) with known thermal properties.
3. **Create local dataset files.** The `datasets/benchmark/active_thermography/` directory does not exist. At minimum, public tier data should be downloadable.
4. **Increase from 3 to 20+ scenes per tier.** Current 3 scenes per tier is statistically insufficient to distinguish algorithm performance reliably.
5. **Fix wavelength range** from "0 -- 0 nm" to "3000 -- 14000 nm" (MWIR/LWIR).

### 5.2 Forward Model (Critical Priority)

6. **Replace microscopy_psf module** with a thermal-PDE-based forward model. The forward model should solve the heat diffusion equation, not perform PSF convolution.
7. **Specify excitation type** (pulsed, lock-in, or step-heating) and include excitation parameters in the forward model.
8. **Populate theta** with thermal physics parameters: conductivity k, density rho, specific heat c_p, emissivity epsilon, detector NETD.
9. **Fix DAG** to reflect thermal physics: Excitation --> Heat PDE --> Surface Temperature T(x,y,t) --> Planck emission --> IR Detector.

### 5.3 Algorithms

10. **Add classical thermography baselines:** standalone TSR, PCT, PPT (without gradient refinement).
11. **Add universal DL baselines:** U-Net, Mask R-CNN.
12. **Add BiLSTM 3D reconstruction** (Wan et al., 2023) and DeepLabv3+BiLSTM hybrid (2025).
13. **Verify LSTM-NDT and DefectNet citations** -- these names do not clearly map to published papers.
14. **Add defect-level metrics** (precision, recall, IoU, Dice) alongside pixel-level PSNR/SSIM.

### 5.4 Infrastructure

15. **Define PSNR_norm** normalization formula explicitly.
16. **Document HDF5 schema** with key names, shapes, dtypes, and loading code.
17. **Document Docker submission API** for hidden tier.
18. **Add confidence intervals** to leaderboard scores (especially with only 3 scenes/tier).
19. **Fix gallery JavaScript** rendering issue.

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Replace DAGM 2007 with a real thermography dataset or proper synthetic thermal data | Dataset team | TODO |
| CRITICAL | Replace microscopy_psf forward model with heat-equation-based module | Physics team | TODO |
| CRITICAL | Replace shepp_logan generator with thermal-defect phantom generator | Physics team | TODO |
| CRITICAL | Populate theta with thermal physics parameters (k, rho, c_p, epsilon, NETD) | Physics team | TODO |
| CRITICAL | Fix wavelength range from "0--0 nm" to "3000--14000 nm" | Config team | TODO |
| CRITICAL | Create local dataset directory and populate with data | Dataset team | TODO |
| HIGH | Specify excitation type (pulsed/lock-in/step-heating) | Physics team | TODO |
| HIGH | Fix DAG to thermal physics pipeline | Physics team | TODO |
| HIGH | Add PCT, PPT, standalone TSR baselines | Algorithm team | TODO |
| HIGH | Add U-Net / Mask R-CNN DL baselines | Algorithm team | TODO |
| HIGH | Verify LSTM-NDT and DefectNet citations | Algorithm team | TODO |
| HIGH | Define PSNR_norm formula | Scoring team | TODO |
| HIGH | Increase to 20+ scenes per tier | Dataset team | TODO |
| MEDIUM | Add BiLSTM 3D and DeepLabv3+BiLSTM algorithms | Algorithm team | TODO |
| MEDIUM | Add defect-level metrics (IoU, Dice, precision, recall) | Scoring team | TODO |
| MEDIUM | Document HDF5 schema and Docker submission specs | Docs team | TODO |
| MEDIUM | Resolve mismatch_params inconsistency between base and expanded configs | Config team | TODO |
| MEDIUM | Add confidence intervals to leaderboard | Scoring team | TODO |
| LOW | Fix gallery JavaScript rendering | Frontend team | TODO |
| LOW | Add phase-based evaluation for lock-in mode | Scoring team | TODO |
| LOW | Add Wiener and Tikhonov baselines | Algorithm team | TODO |

---

## Appendix: Key References

### Thermography Processing Algorithms
- Shepard et al. "Reconstruction and enhancement of active thermographic image sequences." Optical Engineering 42(5):1337-1342 (2003). [TSR foundational paper]
- Rajic. "Principal component thermography for flaw contrast enhancement and flaw depth characterisation in composite structures." Composite Structures 58(4):521-528 (2002).
- Maldague & Marinetti. "Pulse phase infrared thermography." J. Applied Physics 79(5):2694-2698 (1996). [PPT foundational paper]
- Vavilov & Burleigh. "Review of pulsed thermal NDT." QIRT Journal 12(2):147-180 (2015).

### Deep Learning for Thermography
- Bang et al. "Defect shape detection and defect reconstruction in active thermography by means of 2D CNN as well as spatiotemporal ConvLSTM network." QIRT Journal 19(2):103-114 (2022).
- Wan et al. "Automatic defect detection and 3D reconstruction from pulsed thermography images based on BiLSTM." Engineering Applications of AI 126:106574 (2023).
- Fleuret et al. "Spatial and temporal deep learning algorithms for defect segmentation in infrared thermographic imaging of CFRP." NDT&E International (2025).
- Saeed et al. "Machine learning in thermography NDT: A systematic review." Applied Sciences 15(17):9624 (2025).

### Dataset Sources (DAGM 2007 -- current, incorrect)
- Wieler & Hahn. "Weakly supervised learning for industrial optical inspection." DAGM (2007).

### Recommended Replacement Dataset Sources
- Marani et al. "Advanced Thermal Imaging Processing and Deep Learning Integration for Enhanced Defect Detection in CFRP Laminates." PMC (2024).
- Bang et al. "A Dataset of Pulsed Thermography for Automated Defect Depth Estimation." Applied Sciences 13(24):13093 (2023).

---

*Comprehensive 6-point review on 2026-03-03. Active Thermography benchmark has CRITICAL issues: wrong dataset domain (DAGM 2007 is optical, not thermal), wrong forward model physics (microscopy PSF instead of heat equation), no local data, and only 4 algorithms with poor coverage. This modality requires a ground-up rebuild before it can be considered a credible benchmark.*
