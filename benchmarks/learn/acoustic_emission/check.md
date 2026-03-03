# Comprehensive Benchmark QA Check — Acoustic Emission (AE)

**URL:** https://pwm.platformai.org/benchmark/acoustic_emission
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
| HIGH     | 2     |
| MEDIUM   | 4     |
| LOW      | 3     |

### HIGH Severity

**H1. Data dimensions not specified on webpage**
- No image/signal shape, sampling rate, or array geometry documented
- Users cannot design algorithms without knowing input/output dimensions
**Fix:** Add data dimensions section (signal length, number of sensors, spatial grid size).

**H2. PSNR_norm undefined in scoring formula**
- Scoring: `0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × Consistency`
- PSNR_norm normalization method not defined (min-max? percentile? per-sample?)
**Fix:** Define normalization method explicitly.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Mismatch ranges on webpage may not match actual data (cannot verify without local H5) |
| M2 | Only 5 scenes per tier — very small sample size for statistical significance |
| M3 | ResUNet drops from 30.75 dB (public) to 20.45 dB (hidden) — 10 dB collapse, may indicate overfitting or extreme mismatch |
| M4 | HDF5 schema undocumented — key names, dtypes, array dimensions not specified |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | No alt-text on gallery images |
| L2 | SSIM window size not specified |
| L3 | Mismatch parameter units inconsistently formatted (mm, m/s, μs vs dimensionless) |

---

## 2. Local Dataset Inspection

### File Inventory

**NO LOCAL DATASET FILES** — No HDF5 files found at `datasets/benchmark/acoustic_emission/`

| Tier | File | Status |
|------|------|--------|
| Public | — | GCS only (referenced on webpage) |
| Dev | — | GCS only (referenced on webpage) |
| Hidden | — | Server-only |

### GCS Status (from webpage)

- Challenge public HDF5 on GCS: **OK** (referenced)
- Challenge dev HDF5 on GCS: **OK** (referenced)

### Dataset Integrity Assessment: **CANNOT VERIFY** (no local files)

---

## 3. Public Dataset Source Assessment

### SEG/EAGE Salt Model: **GOOD**

- Aminzadeh et al. (1997), Society of Exploration Geophysicists
- Standard velocity model used in seismic/acoustic research
- Widely cited in geophysics community (~1,000+ citations)
- Open-access, well-documented geometry

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Public: Well-known? | GOOD | SEG/EAGE Salt Model is a standard in geophysics |
| Public: Accepted by professors? | GOOD | Widely used in seismic research and teaching |
| Dev: Protected? | UNKNOWN | Cannot verify without local data |
| Hidden: Protected? | UNKNOWN | Cannot verify without local data |

### Limitations

- SEG/EAGE Salt Model is primarily a **seismic velocity model**, not an AE-specific dataset
- AE community typically uses experimental datasets from structural health monitoring (SHM)
- Consider adding data from real AE experiments (e.g., pencil lead break tests, fatigue crack monitoring)

---

## 4. Algorithm Coverage Assessment

### Currently Tested: 4 algorithms

| # | Algorithm | Type | Public PSNR | Dev PSNR | Hidden PSNR |
|---|-----------|------|-------------|----------|-------------|
| 1 | SwinIR + gradient | Transformer | 32.14 dB | 27.91 dB | 22.92 dB |
| 2 | PnP-RED + gradient | Plug-and-play | 27.00 dB | 23.05 dB | 22.43 dB |
| 3 | ResUNet + gradient | CNN | 30.75 dB | 21.72 dB | 20.45 dB |
| 4 | Tikhonov + gradient | Classical | 23.58 dB | 22.29 dB | 20.90 dB |

### Missing Famous/Recent Algorithms

| Priority | Algorithm | Year | Why |
|----------|-----------|------|-----|
| HIGH | Time Reversal (TR) | Classical | Gold standard for AE source localization, widely used in SHM |
| HIGH | Beamforming (delay-and-sum) | Classical | Standard array processing baseline, expected by AE community |
| HIGH | CNN-based AE localization | 2024 | Deep residual networks for AE, active research area |
| MEDIUM | MUSIC (MUltiple SIgnal Classification) | Classical | High-resolution DOA estimation, standard in array processing |
| MEDIUM | GAN-Inception hybrid | 2024 | Hybrid GAN + Inception for coordinate-based AE localization |
| MEDIUM | CWT + CNN | 2024 | Time-frequency image + CNN, popular in recent AE literature |
| LOW | Time-of-arrival (TOA) triangulation | Classical | Simplest geometric baseline |
| LOW | Delta-T mapping | Classical | Data-driven localization without velocity model |

### Algorithm Gap Analysis

The current algorithm set uses **generic inverse-problem solvers** (SwinIR, PnP-RED, ResUNet, Tikhonov) rather than **AE-specific methods**. While acceptable for a cross-modality benchmark, the leaderboard would be more credible with AE-specific algorithms.

**Total gap: 8 algorithms (5 HIGH/MEDIUM priority)**

---

## 5. Improvement Suggestions

### 5.1 Dataset

1. **Add real AE experimental data** to public tier (e.g., pencil lead break tests, CFRP fatigue)
2. **Increase sample count** from 5 to at least 10-15 per tier for statistical significance
3. **Document data dimensions** — signal length, sensor count, spatial grid, sampling rate
4. **Document HDF5 schema** — key names, shapes, dtypes

### 5.2 Algorithms

5. **Add Time Reversal baseline** — gold standard in AE community, expected by reviewers
6. **Add Beamforming baseline** — standard array processing method
7. **Add CNN-based AE localization** — recent (2024) deep learning approach
8. **Investigate ResUNet collapse** — 10 dB drop from public to hidden

### 5.3 Infrastructure

9. **Define PSNR_norm formula** explicitly
10. **Specify SSIM window size**
11. **Add data dimensions to webpage**
12. **Build local dataset copies** for offline development

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Define PSNR_norm formula | Main server | TODO |
| CRITICAL | Document data dimensions on webpage | Main server | TODO |
| HIGH | Add Time Reversal baseline | Algorithm team | TODO |
| HIGH | Add Beamforming baseline | Algorithm team | TODO |
| HIGH | Build local dataset copies | Data team | TODO |
| MEDIUM | Add CNN-based AE localization | Algorithm team | TODO |
| MEDIUM | Increase sample count (5→15) | Data team | TODO |
| MEDIUM | Add real experimental AE data | Data team | TODO |
| LOW | Add MUSIC algorithm | Algorithm team | TODO |
| LOW | Document HDF5 schema on webpage | Main server | TODO |

---

## Appendix: Key References

- Aminzadeh, F., Brac, J., Kunz, T. "SEG/EAGE 3-D Salt and Overthrust Models." SEG (1997).
- Kundu, T. "Acoustic source localization." Ultrasonics 54(1) (2014).
- Liang, J. et al. "SwinIR: Image Restoration Using Swin Transformer." ICCVW (2021).
- Romano, Y. et al. "The Little Engine that Could: Regularization by Denoising (RED)." SIAM J. Imaging Sci. (2017).
- Ebrahimkhanlou, A. & Salamone, S. "A generalizable DL framework for AE source localization." Mech. Syst. Signal Process. (2019).
- LOCATA Challenge: Acoustic source localization and tracking benchmark. arXiv:1909.01008 (2019).

---

*Comprehensive 6-point review on 2026-03-03. No local dataset — GCS-only assessment. 4 generic algorithms tested, 8 AE-specific algorithms missing.*
