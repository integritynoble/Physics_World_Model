# Comprehensive Benchmark QA Check — confocal_endomicroscopy

**URL:** https://pwm.platformai.org/benchmark/confocal_endomicroscopy
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

**H1. Wrong algorithms on leaderboard -- OCT algorithms served for CLE modality**
- Webpage leaderboard previously showed OCT-specific algorithms (FFT-OCT, Speckle-DenoiseNet, OCTA-Net) routed via the `clinical_optics` pool.
- Current leaderboard baselines (EndoL2H + gradient, FiberNet + gradient, PnP-BM3D + gradient, Interpolation + gradient) are CLE-appropriate.
- The `modify_plan.md` documents this routing error. Verify the fix is fully deployed.
**Fix:** Confirm `_algorithm_catalog.py` CLE-specific override is live; remove any residual OCT entries.

**H2. Config uses `medical_ct_radon` category module -- wrong physics engine for CLE**
- `confocal_endomicroscopy.yaml` sets `category_module: medical_ct_radon` (Radon transform / projection-based sensing).
- CLE is a fiber-bundle confocal fluorescence modality. Its physics is PSF convolution + fiber bundle sampling, not Radon projection.
- Learning materials (`02_forward_model.md`) propagate this error: "This modality uses the `medical_ct_radon` category module."
**Fix:** Change `category_module` to `microscopy_psf` or create a dedicated `confocal_fiber_bundle` module.

**H3. Data source is OCT retinal dataset, not CLE tissue**
- Config: `dataset_id: oct_retinal`, `dataset_url: https://www.kaggle.com/datasets/paultimothymooney/kermany2018` (Kermany et al., Cell 2018).
- This is an OCT retinal dataset (optical coherence tomography), not confocal endomicroscopy tissue.
- CLE images show mucosal microarchitecture (crypts, villi, vessels) with fluorescein, not OCT cross-sections.
**Fix:** Replace with a proper CLE dataset (e.g., EndoData CLE, Mauna Kea Cellvizio datasets, or university pCLE research datasets).

**H4. Fiber bundle honeycomb mismatch range is [0, 0] -- effectively disabled**
- Config: `Fiber bundle honeycomb pattern: range [0, 0]`.
- The honeycomb artifact is the single most characteristic challenge of fiber-bundle endomicroscopy.
- With range [0, 0], this defining mismatch parameter never activates.
- Webpage shows perturbed value 0.0 with range +/-0.15, contradicting the YAML.
**Fix:** Set meaningful range (e.g., [0.0, 0.3]) to enable the primary CLE artifact.

**H5. Webpage scoring formula differs from config**
- Webpage: `0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * (1 - ||y - Hx||/||y||)` (3-component composite).
- Config: `metrics: [psnr, ssim]`, `primary: psnr` (no consistency term).
- The `PSNR_norm` normalization method is undefined.
**Fix:** Align config metrics with webpage composite formula. Define PSNR_norm explicitly.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Webpage says 3 scenes per tier but config has no sample count field; verify actual HDF5 contents |
| M2 | Motion artifact range on webpage (2.0 px/frame perturbed) differs from config (0.0-10.0 px/frame full range) |
| M3 | Fluorescein concentration variation: webpage says 0.52-1.92x but config says 0.3-3.0x -- significant discrepancy |
| M4 | Forward model type listed as `nonlinear_operator` but signal equation is linear (PSF convolution + additive noise) |
| M5 | Default solver `fiber_deconvolution` not listed in solver registry; only `FBP` and `DL-Recon` registered |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Learning materials are generic templates with minimal CLE-specific content |
| L2 | No alt-text on gallery/preview images |
| L3 | Config image shape 64x64 is very small for CLE (clinical pCLE is typically 576x576 or 1024x1024) |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | Directory | Status |
|------|-----------|--------|
| datasets/benchmark/confocal_endomicroscopy/ | **DOES NOT EXIST** | No local data |

No local HDF5 files, spec.json, or ground truth are available for inspection.
The benchmark currently has no locally generated or downloaded dataset.

### Config-Declared Data

| Property | Value | Issue |
|----------|-------|-------|
| dataset_id | `oct_retinal` | WRONG: OCT, not CLE |
| dataset_url | kaggle/kermany2018 | OCT retinal images |
| fallback | `generated` | Uses `shepp_logan` phantom |
| synthetic_generator | `shepp_logan` | CT/MRI phantom, not CLE tissue |
| x_shape | [64, 64] | Very small |
| y_shape | [64, 64] | Very small |

### Dataset Integrity Assessment: **FAIL**

The benchmark has no local dataset. The configured data source (OCT retinal) is the wrong modality entirely. The fallback synthetic generator (Shepp-Logan phantom) produces CT-like ellipse phantoms, not CLE-like mucosal tissue patterns.

---

## 3. Public Dataset Source Assessment

### Current Source: Kermany et al. OCT Retinal -- **WRONG MODALITY**

- Kermany et al. (2018), Cell, doi:10.1016/j.cell.2018.02.010
- This is an **optical coherence tomography** dataset of retinal cross-sections
- CLE produces en face fluorescence images of mucosal tissue, not OCT B-scans
- Completely inappropriate for a CLE benchmark

### Recommended CLE Data Sources

| Dataset | Description | Reference | License |
|---------|-------------|-----------|---------|
| Cellvizio pCLE | Clinical pCLE images from Mauna Kea Technologies systems | Le Goualher et al., MICCAI 2004 | Restricted |
| Fiber-bundle endomicroscopy | University research datasets (GI tract, brain) | Ravi et al., Medical Image Analysis 2019 | Varies |
| ETIS-Larib polyp + CLE | Paired endoscopy + endomicroscopy | Various clinical studies | Research use |
| Synthetic CLE | Generated from histology with fiber bundle + PSF model | Shao et al., Med. Image Anal. 2019 | Open |

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Public: Well-known? | FAIL | Wrong modality (OCT, not CLE) |
| Public: Accepted by professors? | FAIL | CLE experts would reject OCT data |
| Dev: Protected? | N/A | No local data exists |
| Hidden: Protected? | N/A | No local data exists |

---

## 4. Algorithm Coverage Assessment

### Currently on Webpage Leaderboard: 4 algorithms

| # | Algorithm | Type | Public PSNR | Hidden PSNR | Notes |
|---|-----------|------|-------------|-------------|-------|
| 1 | EndoL2H + gradient | Deep learning | 30.31 dB | 22.50 dB | CLE super-resolution (appropriate) |
| 2 | FiberNet + gradient | Deep learning | 28.88 dB | 21.46 dB | Fiber bundle reconstruction CNN (appropriate) |
| 3 | PnP-BM3D + gradient | Plug-and-play | 25.03 dB | 18.51 dB | General-purpose PnP (acceptable) |
| 4 | Interpolation + gradient | Classical | 21.47 dB | 21.02 dB | Baseline (appropriate) |

### Config-Registered Solvers: 2 only

| # | Solver | Module | Appropriate? |
|---|--------|--------|-------------|
| 1 | FBP | `pwm_core.recon.fbp` | NO -- FBP is for Radon/CT, not CLE |
| 2 | DL-Recon | `pwm_core.recon.dl_recon` | Generic -- needs CLE-specific training |

The config solvers do not match the webpage leaderboard entries at all.

### Missing Famous/Recent CLE Algorithms

| Priority | Algorithm | Year | Why |
|----------|-----------|------|-----|
| HIGH | Interpolation + Wiener deconvolution | Classical | Standard CLE processing pipeline (Elahi et al., J. Biomed. Opt. 2014) |
| HIGH | Restormer | 2022 | SOTA image restoration transformer (Zamir et al., CVPR 2022) |
| HIGH | Multistage Neural Network + CAM | 2024 | Cross-channel attention for pCLE deblurring (Photonics 2024) |
| MEDIUM | Bundle-shifting super-resolution | 2018 | PZT-actuated multi-frame SR (Lee et al., BOE 2018) |
| MEDIUM | CNN + Bundle rotation SR | 2023 | Multi-frame SR exploiting rotation (MDPI Sensors 2023, SSIM 1.97x improvement) |
| MEDIUM | Self-supervised CLE filtering | 2025 | SSL on unlabeled CLE video (arXiv 2511.00098) |
| LOW | Richardson-Lucy deconvolution | Classical | Standard iterative deconvolution baseline |
| LOW | CARE / Content-Aware Image Restoration | 2018 | Weigert et al., Nature Methods -- general microscopy restoration |

### Algorithm Gap Analysis

The webpage leaderboard shows 4 appropriate CLE algorithms, which is a reasonable starting set. However, the config file registers completely different (and wrong) solvers. Key gaps:
- No classical CLE-specific baseline in config (Wiener deconvolution, triangular interpolation)
- No modern transformer architectures (Restormer, SwinIR)
- No multi-frame / video-based methods (critical for clinical pCLE)
- FBP solver in config is CT-specific and nonsensical for CLE

**Total gap: 8+ algorithms missing from config; webpage leaderboard and config are desynchronized**

---

## 5. Improvement Suggestions

### 5.1 Dataset (CRITICAL -- nothing works without this)

1. **Replace OCT data source with actual CLE data** -- use synthetic CLE images generated from histology patches passed through a fiber bundle + PSF forward model, or acquire a real pCLE dataset
2. **Replace Shepp-Logan fallback** with a CLE-appropriate synthetic generator (e.g., Voronoi-based mucosal texture + fiber bundle sampling)
3. **Increase image resolution** from 64x64 to at least 256x256 (clinical pCLE produces 576x576+)
4. **Create local HDF5 files** for all three tiers with proper CLE physics

### 5.2 Physics Model

5. **Change `category_module`** from `medical_ct_radon` to `microscopy_psf` or a new CLE-specific module
6. **Enable fiber bundle honeycomb mismatch** -- set range to [0.0, 0.3] instead of [0, 0]
7. **Add fiber core spacing parameter** as a mismatch dimension (core pitch varies 2-10 um across probes)
8. **Correct forward model type** -- PSF convolution + fiber sampling is linear (not `nonlinear_operator`)

### 5.3 Algorithms

9. **Sync config solvers with webpage leaderboard** -- register EndoL2H, FiberNet, PnP-BM3D, Interpolation
10. **Remove FBP solver** from config (Radon-based, irrelevant to CLE)
11. **Add Restormer** -- SOTA transformer for image restoration
12. **Add classical Wiener deconvolution baseline** -- standard CLE processing

### 5.4 Webpage/Config Consistency

13. **Sync mismatch ranges** between webpage and YAML config
14. **Define PSNR_norm** explicitly in scoring formula
15. **Add consistency metric** to config to match webpage composite score

### 5.5 Learning Materials

16. **Rewrite `01_physics_fundamentals.md`** with CLE-specific content (fiber bundles, confocal pinhole, fluorescein, mucosal imaging)
17. **Rewrite `02_forward_model.md`** -- remove Radon references, add fiber bundle sampling + PSF convolution
18. **Add CLE-specific key references** (Ravi et al. 2019 review, Le Goualher et al. MICCAI 2004)

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Replace OCT data source with CLE-appropriate dataset | Dataset team | TODO |
| CRITICAL | Change `category_module` from `medical_ct_radon` to `microscopy_psf` | Config team | TODO |
| CRITICAL | Enable honeycomb mismatch range (currently [0,0]) | Config team | TODO |
| CRITICAL | Replace Shepp-Logan fallback with CLE tissue generator | Dataset team | TODO |
| CRITICAL | Create local HDF5 files for public/dev/hidden tiers | Dataset team | TODO |
| HIGH | Sync config solvers with webpage leaderboard (EndoL2H, FiberNet, etc.) | Algorithm team | TODO |
| HIGH | Remove FBP solver (CT-specific) from CLE config | Config team | TODO |
| HIGH | Sync mismatch ranges between webpage and YAML | Config team | TODO |
| HIGH | Define PSNR_norm and add consistency metric to config | Metrics team | TODO |
| HIGH | Increase image resolution from 64x64 to 256x256+ | Config team | TODO |
| MEDIUM | Add Restormer and classical Wiener baseline | Algorithm team | TODO |
| MEDIUM | Rewrite learning materials with CLE-specific physics | Docs team | TODO |
| MEDIUM | Correct forward model type (linear, not nonlinear) | Config team | TODO |
| MEDIUM | Add fiber core spacing as mismatch parameter | Config team | TODO |
| LOW | Add multi-frame / video-based CLE algorithms | Algorithm team | TODO |
| LOW | Add alt-text to gallery images | Web team | TODO |

---

## Appendix: Key References

- Ravi et al. "Image computing for fibre-bundle endomicroscopy: A review." Medical Image Analysis 62:101620 (2019). doi:10.1016/j.media.2019.101620
- Shao et al. "Fiber bundle image restoration using deep learning." Medical Image Analysis 56:78-92 (2019).
- Elahi et al. "Image processing and analysis for pCLE." J. Biomedical Optics 19(4):046014 (2014).
- Zamir et al. "Restormer: Efficient Transformer for High-Resolution Image Restoration." CVPR (2022).
- Lee et al. "Fiber bundle shifting endomicroscopy for high-resolution imaging." Biomedical Optics Express 9(10):4649-4664 (2018).
- MDPI Sensors 2023. "Fiber Bundle Image Reconstruction Using CNNs and Bundle Rotation in Endomicroscopy." Sensors 23(5):2469.
- Kermany et al. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." Cell 172(5):1122-1131 (2018). [Current data source -- WRONG MODALITY]
- Photonics 2024. "High-Resolution Image Processing of pCLE Based on Multistage Neural Networks and Cross-Channel Attention Module." Photonics 11(2):106.
- arXiv 2511.00098. "A filtering scheme for CLE-video sequences for self-supervised learning." (2025).

---

*Comprehensive 6-point review on 2026-03-03. confocal_endomicroscopy is the most critically misconfigured benchmark reviewed: wrong data source (OCT instead of CLE), wrong physics engine (Radon instead of PSF+fiber), disabled primary mismatch (honeycomb [0,0]), and desynchronized config vs. webpage. Five CRITICAL action items must be resolved before this benchmark produces meaningful results.*