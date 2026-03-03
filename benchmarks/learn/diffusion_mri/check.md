# Comprehensive Benchmark QA Check — diffusion_mri

**URL:** https://pwm.platformai.org/benchmark/diffusion_mri
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

**H1. Webpage reports 128x128 voxels but page also states 64 diffusion directions; local YAML says 30 directions**
- Webpage acquisition parameters: 64 diffusion directions
- Local `diffusion_mri.yaml`: `n_directions: 30`
- This is a critical discrepancy -- algorithms designed for 64 directions will fail on 30-direction data
**Fix:** Sync webpage diffusion directions count with YAML (30) or update YAML to match webpage (64).

**H2. Webpage lists 4 mismatch parameters (b-value error, eddy-current, gradient direction, susceptibility) but local YAML has `mismatch_params: []`**
- Webpage defines perturbed ranges: b-value error (-3.0% to 6.9%), eddy-current distortion (-0.6 to 1.15 voxels), gradient direction error (-1.2 to 2.3 deg), susceptibility distortion (-1.2 to 2.3 voxels)
- Local config: `mismatch_params: []` (empty)
- Local learn material (02_forward_model.md, 04_pwm_benchmark.md): "No mismatch parameters defined for this modality"
- The entire mismatch challenge is disconnected from the local implementation
**Fix:** Populate `mismatch_params` in `diffusion_mri.yaml` with the 4 parameters and their ranges, or clarify that webpage describes a future version.

**H3. Webpage composite score uses 3-tier weighted formula (40% PSNR_norm + 40% SSIM + 20% Consistency) but local metrics only define PSNR (primary) and SSIM**
- Webpage: score = 0.4*PSNR_norm + 0.4*SSIM + 0.2*(1 - ||y - Hx||/||y||)
- Local config: `metrics: names: [psnr, ssim], primary: psnr`
- Consistency metric (data-fidelity term) is entirely absent from local implementation
- PSNR_norm normalization method is undefined
**Fix:** Add consistency metric to local scoring pipeline. Define PSNR_norm formula.

**H4. Webpage shows 12 scenes (3 per tier x 4 tiers: public/dev/hidden + overall) but local data directory `datasets/benchmark/diffusion_mri` does not exist**
- No local dataset directory found at `datasets/benchmark/diffusion_mri`
- Existing benchmarks with data: cacti, cbct, cryo_em, ct, mri, sd_cassi, spc_kronecker, ultrasound
- Diffusion MRI benchmark has no downloadable or generated data locally
**Fix:** Build diffusion MRI dataset or link to existing MRI data with diffusion-specific augmentation.

**H5. Webpage forward model states S = S0 * exp(-b * D_eff) but local learn material uses MRI k-space signal equation s(t) = integral of rho * S_c * exp(-i2pi k.r)**
- Webpage: diffusion signal attenuation model (Stejskal-Tanner)
- Local: generic MRI k-space acquisition model
- These are fundamentally different formulations -- one describes diffusion contrast, the other describes spatial encoding
- The local forward model is the general MRI model, not a diffusion-specific model
**Fix:** Update local forward model documentation to incorporate both the Stejskal-Tanner diffusion weighting AND the k-space spatial encoding.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Webpage says "Siemens MAGNETOM Prisma" scanner but YAML has no scanner field |
| M2 | Webpage b-values [0, 1000] s/mm^2 matches YAML but webpage says 64 directions (YAML says 30) |
| M3 | Webpage TR=4000 ms but YAML says TR=8000 ms -- factor of 2 discrepancy |
| M4 | Webpage says "single-shot spin-echo EPI DWI" but YAML category_module is `medical_ct_radon` -- wrong physics engine |
| M5 | Learn material `02_forward_model.md` claims physics engine is `medical_ct_radon` (Radon transform); diffusion MRI does NOT use Radon projection |
| M6 | 8 baselines on webpage leaderboard (PromptMR, E2E-VarNet, U-Net, PnP-DnCNN, ReconFormer, L1-Wavelet, Score-MRI, Zero-Filled) but local config lists only 1 solver (SENSE WLS tensor fit) |
| M7 | Webpage leaderboard shows "+ gradient" suffix on all methods -- gradient-based mismatch correction is undocumented locally |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Local config `category_module: medical_ct_radon` is incorrect for diffusion MRI (should be `medical_mri_kspace` or a new `medical_diffusion_mri`) |
| L2 | Webpage uses voxel size "2x2x2 mm isotropic" but local YAML has no voxel_size field |
| L3 | No alt-text on gallery images (accessibility) |
| L4 | Placeholder links: /benchmark/diffusion_mri/compete, /benchmark/diffusion_mri/contribute |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | File | Size | Samples |
|------|------|------|---------|
| Public | -- | MISSING | -- |
| Dev | -- | MISSING | -- |
| Hidden | -- | MISSING | -- |

**No local data directory exists** at `datasets/benchmark/diffusion_mri/`.

### Existing Related Dataset: MRI benchmark

The `datasets/benchmark/mri/` directory exists with:
- `public/mri_challenge_public.h5` (~170 MB, 11 samples, 320x320, 15 coils)
- `dev/mri_challenge_dev.h5` (~308 MB, 20 samples, IXI T2w)
- `hidden/mri_challenge_hidden.h5` (~307 MB, 20 samples, BraTS T2w)

These are standard structural MRI (T1/T2-weighted) datasets, NOT diffusion-weighted.

### Local Config vs Webpage Comparison

| Parameter | Local YAML | Webpage | Match |
|-----------|-----------|---------|-------|
| Image shape | 128 x 128 | 128 x 128 | YES |
| b-values | [0, 1000] | [0, 1000] s/mm^2 | YES |
| n_directions | 30 | 64 | NO |
| TR | 8000 ms | 4000 ms | NO |
| TE | 80 ms | 80 ms | YES |
| n_coils | 32 | not specified | -- |
| Field strength | 3.0 T | 3.0 T | YES |
| Mismatch params | [] (empty) | 4 params defined | NO |
| Solvers | 1 (SENSE WLS) | 8 methods | NO |
| category_module | medical_ct_radon | (not shown) | WRONG |

### Dataset Integrity Assessment: **FAIL -- no data exists**

---

## 3. Public Dataset Source Assessment

### Declared Source: Human Connectome Project (HCP)

- Van Essen et al. (2013), "The WU-Minn Human Connectome Project: An overview," NeuroImage
- URL: https://db.humanconnectome.org/
- License: HCP Open Access Data Use Terms
- ~1,200 subjects with diffusion MRI (90 directions, b=1000/2000/3000)
- One of the most widely used diffusion MRI datasets worldwide
- Webpage also mentions UK Biobank as secondary source

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Public: Well-known? | EXCELLENT | HCP is the gold standard for diffusion MRI |
| Public: Accepted by professors? | EXCELLENT | >10,000 citations, used in hundreds of DTI studies |
| Dev: Protected? | UNKNOWN | No dev-tier data source specified locally |
| Hidden: Protected? | UNKNOWN | No hidden-tier data source specified locally |

### Data Source Concerns

1. **HCP data requires registration** -- cannot be auto-downloaded like LoDoPaB-CT for CT benchmark
2. **Fallback is `shepp_logan` phantom** -- the Shepp-Logan phantom is a CT phantom, not appropriate for diffusion MRI
3. **No diffusion-specific synthetic generator** -- unlike the MRI benchmark which has IXI/BraTS loaders
4. **UK Biobank requires separate application** -- months-long approval process

### Source Rating: **FAIR** (excellent declared source, but no practical data pipeline)

---

## 4. Algorithm Coverage Assessment

### Currently Tested Locally: 1 algorithm -- **WORST COVERAGE among all modalities**

| # | Algorithm | Type | Notes |
|---|-----------|------|-------|
| 1 | SENSE (WLS tensor fit) | Classical parallel imaging | Only local solver |

### Webpage Leaderboard: 8 algorithms (not implemented locally)

| Rank | Method | Overall Score | Type |
|------|--------|:------------:|------|
| 1 | PromptMR + gradient | 0.777 | Unrolled DL (prompt-based) |
| 2 | E2E-VarNet + gradient | 0.763 | Unrolled DL (variational) |
| 3 | U-Net + gradient | 0.740 | CNN post-processing |
| 4 | PnP-DnCNN + gradient | 0.694 | Plug-and-play |
| 5 | ReconFormer + gradient | 0.664 | Transformer-based |
| 6 | L1-Wavelet (ESPIRiT) + gradient | 0.619 | Compressed sensing |
| 7 | Score-MRI + gradient | 0.618 | Score-based diffusion model |
| 8 | Zero-Filled IFFT + gradient | 0.570 | Naive baseline |

### Missing Famous/Recent Algorithms (from Literature Review)

| Priority | Algorithm | Year | Why Important |
|----------|-----------|------|---------------|
| HIGH | SDnDTI (Self-supervised denoising DTI) | 2022 | Self-supervised, no clean targets needed; specifically designed for DTI |
| HIGH | Patch2Self | 2020 | Self-supervised denoiser for dMRI, Fadnavis et al.; widely used in DIPY |
| HIGH | MPPCA (Marchenko-Pastur denoising) | 2017 | Standard dMRI denoising in MRtrix3, Veraart et al. |
| HIGH | q-DL (q-space deep learning) | 2021 | Joint k-q undersampled reconstruction, Mani et al. |
| MEDIUM | SSDiffRecon | 2024 | Self-supervised diffusion model MRI reconstruction |
| MEDIUM | PromptMR+ | 2025 | Enhanced PromptMR, MICCAI 2023 winner update |
| MEDIUM | NODDI-NET | 2019 | DL-based NODDI parameter estimation |
| MEDIUM | DeepDTI | 2020 | CNN-based DTI enhancement from few directions |
| MEDIUM | Probabilistic diffusion framework (ISMRM 2025) | 2025 | IDDPM-based quantitative MRI |
| LOW | TractSeg | 2018 | DL tract segmentation (downstream task) |
| LOW | DESIGNER pipeline | 2021 | Integrated dMRI preprocessing + denoising |

### Algorithm Gap Analysis

The diffusion MRI benchmark has the worst algorithm coverage among all modalities with local data. The single local solver (SENSE WLS) is a parallel imaging method not specific to diffusion MRI. Critical gaps:

- **Zero diffusion-specific algorithms** locally (no tensor fitting, no CSD, no NODDI)
- **Zero deep learning methods** locally (webpage lists 5 DL methods)
- **Zero denoising methods** locally (denoising is the primary DL application in diffusion MRI)
- **No gradient-based mismatch correction** locally (all webpage methods use "+ gradient")
- **8 webpage methods vs 1 local method** -- largest gap of any modality

**Total gap: 18+ algorithms (critical deficit)**

---

## 5. Improvement Suggestions

### 5.1 Dataset (CRITICAL -- no data exists)

1. **Build diffusion MRI dataset pipeline** -- implement HDF5 generator analogous to `datasets/benchmark/mri/build_dataset.py`
2. **Fix synthetic fallback** -- replace `shepp_logan` (CT phantom) with a diffusion-appropriate phantom (e.g., Fiberfox simulation, POSSUM)
3. **Implement HCP data loader** -- script to convert HCP diffusion data to PWM HDF5 format
4. **Define tier data sources** -- public: HCP subset; dev: UK Biobank or in-house; hidden: clinical DTI with pathology
5. **Populate mismatch parameters** -- implement the 4 mismatch types from the webpage (b-value error, eddy-current, gradient direction, susceptibility)

### 5.2 Forward Model (HIGH)

6. **Fix category_module** -- change from `medical_ct_radon` to `medical_mri_kspace` or create `medical_diffusion_mri`
7. **Implement diffusion-specific forward model** -- Stejskal-Tanner signal equation combined with k-space encoding
8. **Add eddy-current distortion model** -- critical for realistic diffusion MRI simulation
9. **Add susceptibility distortion model** -- fieldmap-based geometric distortion

### 5.3 Algorithms (HIGH)

10. **Add DTI tensor fitting baseline** -- weighted least-squares DTI fit (already declared but not connected)
11. **Add MPPCA denoising** -- standard clinical denoising, Veraart et al. 2016
12. **Add Patch2Self** -- self-supervised denoising, DIPY implementation available
13. **Add E2E-VarNet** -- unrolled network baseline (already on webpage leaderboard)
14. **Add SDnDTI** -- self-supervised DTI denoising, no clean data needed
15. **Sync n_directions** -- resolve 30 vs 64 discrepancy between YAML and webpage

### 5.4 Infrastructure (MEDIUM)

16. **Sync TR value** -- resolve 8000 ms (YAML) vs 4000 ms (webpage) discrepancy
17. **Add consistency metric** -- implement the data-fidelity scoring term from webpage
18. **Define PSNR_norm** -- document normalization method
19. **Add gradient-based correction module** -- all webpage methods use "+ gradient" suffix
20. **Update learn materials** -- fix incorrect physics engine references in 01/02/03 docs

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Build diffusion MRI dataset (HDF5, 3 tiers) | Dataset team | TODO |
| CRITICAL | Fix `category_module` from `medical_ct_radon` to MRI-based | Config team | TODO |
| CRITICAL | Populate `mismatch_params` in YAML (4 params from webpage) | Config team | TODO |
| CRITICAL | Resolve n_directions discrepancy (30 in YAML vs 64 on webpage) | Config team | TODO |
| CRITICAL | Resolve TR discrepancy (8000 ms in YAML vs 4000 ms on webpage) | Config team | TODO |
| HIGH | Replace `shepp_logan` fallback with diffusion-appropriate phantom | Dataset team | TODO |
| HIGH | Implement diffusion-specific forward model (Stejskal-Tanner + k-space) | Physics team | TODO |
| HIGH | Add DTI tensor fitting solver (WLS, beyond SENSE) | Algorithm team | TODO |
| HIGH | Add MPPCA denoising baseline | Algorithm team | TODO |
| HIGH | Add Patch2Self denoising baseline | Algorithm team | TODO |
| HIGH | Add E2E-VarNet or PromptMR (DL baseline) | Algorithm team | TODO |
| MEDIUM | Add consistency metric to scoring pipeline | Framework team | TODO |
| MEDIUM | Define PSNR_norm formula in scoring | Framework team | TODO |
| MEDIUM | Add gradient-based mismatch correction module | Algorithm team | TODO |
| MEDIUM | Add SDnDTI self-supervised denoising | Algorithm team | TODO |
| MEDIUM | Update learn materials (fix physics engine references) | Docs team | TODO |
| LOW | Add voxel_size field to YAML | Config team | TODO |
| LOW | Add alt-text to gallery images | Web team | TODO |
| LOW | Add TractSeg downstream evaluation | Algorithm team | TODO |

---

## Appendix: Key References

- Basser, Mattiello & Le Bihan. "MR diffusion tensor spectroscopy and imaging." Biophysical Journal 66:259-267 (1994).
- Stejskal & Tanner. "Spin diffusion measurements." J Chem Phys 42:288-292 (1965).
- Van Essen et al. "The WU-Minn Human Connectome Project: An overview." NeuroImage 80:62-79 (2013).
- Sotiropoulos et al. "Advances in diffusion MRI acquisition and processing in the Human Connectome Project." NeuroImage 80:125-143 (2013).
- Veraart et al. "Denoising of diffusion MRI using random matrix theory." NeuroImage 142:394-406 (2016).
- Fadnavis et al. "Patch2Self: Denoising diffusion MRI with self-supervised learning." NeurIPS (2020).
- Bai et al. "PromptMR: Prompt-based MRI Reconstruction." ECCV (2024).
- Sriram et al. "End-to-End Variational Networks for Accelerated MRI Reconstruction." MICCAI (2020).
- Kang et al. "Self-supervised learning for denoising of multidimensional MRI data." MRM 92(3):1299-1313 (2024).
- Luo et al. "SDnDTI: Self-supervised deep learning-based denoising for diffusion tensor MRI." NeuroImage 253:119033 (2022).
- Xiang et al. "DDM^2: Self-supervised diffusion MRI denoising." ICLR (2023).
- Mani et al. "Model-based deep learning for joint k-q undersampled diffusion MRI reconstruction." MRM 86(4):2120-2137 (2021).

---

*Comprehensive 6-point review on 2026-03-03. Diffusion MRI has the worst local implementation status among benchmarked modalities: no dataset, 1 solver (vs 8 on webpage), wrong physics engine, and empty mismatch parameters. The webpage presents a well-designed benchmark with HCP data, 4 mismatch types, 8 baseline methods, and a 3-tier composite score -- but none of this is reflected in the local codebase. This modality requires a ground-up build of dataset, forward model, mismatch engine, and solver pipeline.*