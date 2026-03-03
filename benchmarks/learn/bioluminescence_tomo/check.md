# Comprehensive Benchmark QA Check — Bioluminescence Tomography (BLT)

**URL:** https://pwm.platformai.org/benchmark/bioluminescence_tomo
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

**H1. WRONG DATASET SOURCE — SEG/EAGE Salt Model cited for BLT**
- Webpage says "SEG/EAGE Salt Model (Aminzadeh et al., SEG 1997)"
- This is a seismic/geophysics velocity model — completely unrelated to bioluminescence
- Obvious copy-paste error from another modality
**Fix:** Remove SEG/EAGE citation. Use real BLT data or tissue-optical simulation.

**H2. Forward model wrong — PSF convolution instead of diffusion equation**
- BLT requires the diffusion equation or radiative transfer equation (3D volume-to-surface)
- Current model uses `microscopy_psf` category module (2D convolution)
- BLT is NOT a PSF-convolution modality
**Fix:** Implement BLT-specific forward model using diffusion approximation (DE) or Monte Carlo.

**H3. BLT is inherently 3D but benchmark uses 2D**
- Config shows data dimensions [64, 64] — a 2D grid
- Real BLT reconstructs 3D volumetric source distributions from 2D surface measurements
- 2D treatment fundamentally misrepresents the inverse problem
**Fix:** Upgrade to 3D data format (e.g., 64×64×64 or 128×128×64).

**H4. PSNR_norm undefined in scoring formula**
- Scoring: `0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × Consistency`
- No normalization bounds or method specified
**Fix:** Define PSNR_norm explicitly with bounds and worked example.

**H5. Mismatch parameter "autofluorescence_background" range [-6, 13.8] is unphysical**
- Negative autofluorescence has no physical meaning (it's always positive)
- The range is unreasonably wide for a realistic parameter
**Fix:** Clamp to [0, max_autofluorescence] based on tissue optics literature.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Only 5 scenes per tier — too few for statistical significance |
| M2 | Leaderboard uses generic inverse-problem solvers, no BLT-specific methods |
| M3 | ResUNet collapses from 31.3 dB (public) to 20.3 dB (hidden) — 11 dB drop |
| M4 | DAG includes "Rotation" element — not standard in BLT forward models |
| M5 | Mismatch parameter ranges have equal span across tiers — no difficulty progression |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | No references to BLT literature (Wang, Cong, Alexandrakis) |
| L2 | SSIM window size not specified |
| L3 | HDF5 schema undocumented |

---

## 2. Local Dataset Inspection

### File Inventory

**NO LOCAL DATASET FILES** — Directory `datasets/benchmark/bioluminescence_tomo/` does not exist.

| Tier | File | Status |
|------|------|--------|
| Public | — | No local data |
| Dev | — | No local data |
| Hidden | — | No local data |

### Config Files

- `benchmarks/configs/bioluminescence_tomo.yaml` — base config, maturity M0
- Fallback: `shepp_logan` synthetic generator (CT phantom, wrong domain)

### Dataset Integrity Assessment: **FAIL** (no data, wrong domain fallback)

---

## 3. Public Dataset Source Assessment

### Current: SEG/EAGE Salt Model — **FAIL (wrong domain)**

- SEG/EAGE Salt Model is a seismic velocity model
- Zero relevance to bioluminescence or tissue optics
- No BLT researcher would accept this

### Recommended BLT Datasets

| Dataset | Source | Type | Suitability |
|---------|--------|------|-------------|
| Digimouse (Segars et al.) | Simulation | 3D mouse atlas | HIGH — standard BLT phantom |
| MOBY/MOBY2 mouse phantom | Simulation | 4D mouse atlas | HIGH — dynamic BLT |
| Monte Carlo (MCX/MCML) | Simulation | Photon transport | HIGH — provides ground truth |
| In-vivo BLT (Wang lab) | Experimental | Real mouse data | MEDIUM — limited availability |

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Public: Well-known? | FAIL | SEG/EAGE is geophysics, not optics |
| Public: Accepted by professors? | FAIL | No BLT professor would accept |
| Dev: Protected? | UNKNOWN | No data exists |
| Hidden: Protected? | UNKNOWN | No data exists |

---

## 4. Algorithm Coverage Assessment

### Currently Tested: 4 generic algorithms

| # | Algorithm | Type | Public PSNR | Dev PSNR | Hidden PSNR |
|---|-----------|------|-------------|----------|-------------|
| 1 | SwinIR + gradient | Transformer | 32.99 dB | 29.88 dB | 27.53 dB |
| 2 | ResUNet + gradient | CNN | 31.30 dB | 23.17 dB | 20.32 dB |
| 3 | PnP-RED + gradient | Plug-and-play | 26.87 dB | 21.38 dB | 19.05 dB |
| 4 | Tikhonov + gradient | Classical | 22.95 dB | 22.42 dB | 21.39 dB |

### Missing Famous/Recent BLT Algorithms

| Priority | Algorithm | Year | Why |
|----------|-----------|------|-----|
| CRITICAL | FEM-based BLT (Cong et al.) | 2010s | Standard BLT solver using finite element method |
| CRITICAL | FISTA-Net for BLT | 2022 | Model-driven unrolled network for BLT (SPIE 2022) |
| HIGH | Neural Field BLT | 2025 | SOTA neural implicit representation for BLT (JIOHS 2025) |
| HIGH | 1D-CNN-BLT | 2021 | First DL method for BLT (Frontiers Oncology 2021) |
| HIGH | Monte Carlo reference (MCX) | Standard | Ground truth simulator for validation |
| MEDIUM | VoxDMRN | 2022 | Voxel-based deep model for BLT |
| MEDIUM | Log-TV regularization | 2024 | Total variation with log-penalty for BLT |
| MEDIUM | Self-training FEM-Net | 2022 | Self-supervised FEM approach (IEEE) |
| LOW | Adaptive FEM | 2013 | Mesh-adaptive finite element BLT solver |
| LOW | Sparse reconstruction | 2010s | L1-regularized BLT with sparsity prior |

### Algorithm Gap Analysis

The current algorithm set uses **zero BLT-specific methods**. All 4 methods are generic inverse-problem solvers. The community would expect at minimum:
- FEM-based forward model solver
- Monte Carlo photon transport validation
- At least one dedicated deep learning BLT method

**Total gap: 10 algorithms (3 CRITICAL)**

---

## 5. Improvement Suggestions

### 5.1 Dataset (CRITICAL)

1. **Remove SEG/EAGE citation** — replace with BLT-relevant source
2. **Implement BLT-specific forward model** using diffusion equation
3. **Switch to 3D data format** — BLT is a 3D→2D inverse problem
4. **Use Digimouse phantom** or Monte Carlo (MCX) for simulation
5. **Increase sample count** from 5 to 20+ per tier

### 5.2 Algorithms

6. **Add FEM-based BLT solver** — standard in the community
7. **Add FISTA-Net for BLT** — model-driven unrolled network
8. **Add Neural Field BLT** — 2025 SOTA approach
9. **Investigate ResUNet collapse** — 11 dB drop from public to hidden

### 5.3 Infrastructure

10. **Define PSNR_norm** — specify normalization bounds
11. **Fix autofluorescence range** — clamp to non-negative
12. **Add BLT references** (Wang, Cong, Alexandrakis)
13. **Document HDF5 schema** — key names, shapes, dtypes

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Remove SEG/EAGE Salt Model citation (wrong domain) | Content | TODO |
| CRITICAL | Implement diffusion equation forward model | Physics | TODO |
| CRITICAL | Switch to 3D data format (64³ or 128³) | Data | TODO |
| CRITICAL | Define PSNR_norm formula | Metrics | TODO |
| HIGH | Use Digimouse phantom or MCX simulation | Data | TODO |
| HIGH | Add FEM-based BLT solver | Algorithm | TODO |
| HIGH | Add FISTA-Net for BLT | Algorithm | TODO |
| MEDIUM | Add Neural Field BLT (2025) | Algorithm | TODO |
| MEDIUM | Fix autofluorescence range (non-negative) | Physics | TODO |
| MEDIUM | Increase sample count (5→20+) | Data | TODO |
| LOW | Add BLT references (Wang, Cong) | Content | TODO |
| LOW | Document HDF5 schema | Docs | TODO |
| LOW | Add multi-spectral BLT support | Data | TODO |

---

## Appendix: Key References

- Wang, G., Cong, W., et al. "Recent development in bioluminescence tomography." Current Medical Imaging Reviews (2006).
- Cong, W. et al. "Practical reconstruction method for bioluminescence tomography." Optics Express (2005).
- FISTA-Net for BLT: SPIE 2022, doi:10.1117/12.2654054
- Neural Field BLT: JIOHS 2025, doi:10.1142/S1793545825500026
- 1D-CNN-BLT: Frontiers in Oncology 2021, doi:10.3389/fonc.2021.760689
- VoxDMRN: PubMed 35363720 (2022)
- Log-TV for BLT: ScienceDirect 2024

---

*Comprehensive 6-point review on 2026-03-03. No local dataset — M0 maturity. CRITICAL: wrong domain source (SEG/EAGE), wrong forward model (PSF instead of diffusion equation), 2D instead of 3D. Zero BLT-specific algorithms tested.*
