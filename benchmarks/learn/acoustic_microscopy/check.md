# Comprehensive Benchmark QA Check — Acoustic Microscopy (SAM)

**URL:** https://pwm.platformai.org/benchmark/acoustic_microscopy
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
| HIGH     | 3     |
| MEDIUM   | 6     |
| LOW      | 3     |

### HIGH Severity

**H1. Leaderboard ranking inconsistency**
- Composite score formula `0.4*PSNR_norm + 0.4*SSIM + 0.2*Consistency` but tier weighting not transparent
- SAM-Net #1 overall but AcousticFormer beats it on public PSNR — verify arithmetic
**Fix:** Show composite score formula worked example for every method.

**H2. Spec range nesting broken — no proper difficulty progression**
- focus_depth_error: Public [-4, 8], Dev [-4.8, 7.2], Hidden [-2.8, 9.2]
- Ranges shift rather than widen — Dev narrows high end, Hidden narrows low end
- This violates the expected Public ⊂ Dev ⊂ Hidden nesting
**Fix:** Ensure proper spec range nesting (Public narrowest, Hidden widest).

**H3. Data dimensions not specified**
- No signal/image shape, sampling rate, or transducer frequency documented
- Users cannot design algorithms without knowing dimensions
**Fix:** Add data dimensions section.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | PSNR_norm undefined — normalization method not specified |
| M2 | Only 3 scenes per tier — too few for statistical significance |
| M3 | Forward model DAG incomplete: shows P→D but missing transducer bandwidth, beam forming, frequency-dependent attenuation |
| M4 | Gate position error units shown as "—" (dash) — should specify ns or sample indices |
| M5 | Dev tier scoring unclear — no visible ground truth, server-side GT not stated |
| M6 | Incomplete references — DAGM 2007 has no DOI, NDT Transformer 2024 no arXiv, U-Net for NDT 2021 no venue |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Gallery JavaScript may not render — selectGalleryScene() references DOM IDs |
| L2 | SAFT (2003) consistently last — should be labeled "Baseline" |
| L3 | Spec tables repeated 3× — should consolidate with tier columns |

---

## 2. Local Dataset Inspection

### File Inventory

**NO LOCAL DATASET FILES** — No HDF5 files at `datasets/benchmark/acoustic_microscopy/`

| Tier | File | Status |
|------|------|--------|
| Public | — | GCS only |
| Dev | — | GCS only |
| Hidden | — | Server-only |

### Dataset Integrity Assessment: **CANNOT VERIFY** (no local files)

---

## 3. Public Dataset Source Assessment

### DAGM 2007 (Wieler & Hahn): **FAIR**

- Wieler & Hahn, DAGM German Conference on Pattern Recognition (2007)
- Defect detection dataset for industrial inspection
- Moderately cited in NDT community
- Not the most widely-used benchmark in acoustic microscopy

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Public: Well-known? | FAIR | DAGM is known in industrial vision, not specific to SAM |
| Public: Accepted by professors? | FAIR | Used in NDT research but not premier SAM benchmark |
| Dev: Protected? | UNKNOWN | Cannot verify without local data |
| Hidden: Protected? | UNKNOWN | Cannot verify without local data |

### Recommendations

- Consider adding data from real SAM acquisitions (e.g., IC/semiconductor inspection)
- DAGM is primarily a **visual defect detection** dataset — unclear how it maps to acoustic microscopy physics
- SAM community would expect datasets from actual acoustic microscope instruments

---

## 4. Algorithm Coverage Assessment

### Currently Tested: 4 algorithms

| # | Algorithm | Type | Public PSNR | Dev PSNR | Hidden PSNR |
|---|-----------|------|-------------|----------|-------------|
| 1 | SAM-Net + gradient | Domain-specific DL | 33.09 dB | 29.40 dB | 26.79 dB |
| 2 | AcousticFormer + gradient | Transformer | 30.66 dB | 26.68 dB | 22.66 dB |
| 3 | PnP-ADMM + gradient | Plug-and-play | 27.35 dB | 26.28 dB | 24.21 dB |
| 4 | SAFT + gradient | Classical | 23.18 dB | 23.68 dB | 22.24 dB |

### Missing Famous/Recent Algorithms

| Priority | Algorithm | Year | Why |
|----------|-----------|------|-----|
| HIGH | Synthetic Aperture Focusing (pure SAFT) | Classical | Gold standard baseline — current uses "+gradient" variant |
| HIGH | V(z) Curve Analysis | Classical | Standard SAM characterization method |
| HIGH | DiffPam (Diffusion) | 2024 | Diffusion-based reconstruction, 5× scanning acceleration |
| MEDIUM | EDSR-M (Enhanced Deep SR) | 2024 | Residual learning + attention for microscopy SR |
| MEDIUM | Wiener Deconvolution | Classical | Standard deconvolution baseline for microscopy |
| MEDIUM | U-Net (basic) | 2017 | Universal DL baseline |
| LOW | RCAB (Residual Channel Attention) | 2024 | Channel attention for scale-variant features |
| LOW | Sparse reconstruction (CS) | 2010s | Compressive sensing for SAM |

### Algorithm Gap Analysis

Current coverage is reasonable with SAM-Net and AcousticFormer as domain-specific methods. Main gaps:
- No pure classical SAFT baseline (current one has "+gradient" augmentation)
- No deconvolution-based methods
- No diffusion-based methods (emerging 2024)

**Total gap: 8 algorithms (3 HIGH priority)**

---

## 5. Improvement Suggestions

### 5.1 Dataset

1. **Increase sample count** from 3 to at least 10 per tier
2. **Fix spec range nesting** — ensure Public ⊂ Dev ⊂ Hidden
3. **Add real SAM data** from semiconductor/IC inspection
4. **Document data dimensions** and HDF5 schema

### 5.2 Algorithms

5. **Add pure SAFT baseline** without gradient augmentation
6. **Add V(z) curve analysis** — standard SAM method
7. **Add Wiener deconvolution** — classical baseline
8. **Add DiffPam** — latest diffusion approach (2024)

### 5.3 Infrastructure

9. **Define PSNR_norm formula**
10. **Fix gate_position_error units** (currently "—")
11. **Add complete references** with DOIs
12. **Build local dataset copies**

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Fix spec range nesting (Public ⊂ Dev ⊂ Hidden) | Data team | TODO |
| CRITICAL | Increase sample count (3→10+) | Data team | TODO |
| CRITICAL | Define PSNR_norm formula | Main server | TODO |
| HIGH | Add pure SAFT baseline | Algorithm team | TODO |
| HIGH | Document data dimensions | Main server | TODO |
| HIGH | Add complete references with DOIs | Main server | TODO |
| MEDIUM | Add DiffPam diffusion method | Algorithm team | TODO |
| MEDIUM | Add Wiener deconvolution | Algorithm team | TODO |
| MEDIUM | Add real SAM instrument data | Data team | TODO |
| LOW | Fix gate_position_error units | Main server | TODO |

---

## Appendix: Key References

- Wieler, M. & Hahn, S. "Weakly Supervised Factories." DAGM (2007).
- Briggs, G.A.D. & Kolosov, O.V. "Acoustic Microscopy." Oxford Univ. Press (2010).
- Liang, J. et al. "SwinIR: Image Restoration Using Swin Transformer." ICCVW (2021).
- DiffPam: Diffusion model for photoacoustic microscopy acceleration. Sci. Rep. (2024).

---

*Comprehensive 6-point review on 2026-03-03. No local dataset — GCS-only. 4 algorithms tested, 8 missing. CRITICAL: spec ranges don't nest properly, only 3 scenes per tier.*
