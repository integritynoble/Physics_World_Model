# Comprehensive Benchmark QA Check — SD-CASSI

**URL:** https://pwm.platformai.org/benchmark/sd_cassi
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
| HIGH     | 4     |
| MEDIUM   | 6     |
| LOW      | 3     |

### HIGH Severity

**H1. Webpage shows 9 mismatch parameters but local data has only 5**
- Webpage: mask_dx, mask_dy, mask_rotation, dispersion_slope, dispersion_axis, sigma_read, dark_current, gain, mask_theta
- Local H5: mask_dx, mask_dy, mask_rotation, dispersion_slope, dispersion_axis (5 only)
- 4 extra parameters on webpage don't exist in actual data
**Fix:** Sync webpage to match HDF5 spec_ranges (5 parameters).

**H2. Massive performance drop from public to dev unexplained**
- SSR-L: 38.03 dB (public) -> 19.53 dB (dev) -- 18.5 dB drop!
- This is extreme and suggests the dev tier is unreasonably harder or data format issues
- Dev spatial size is 500x500 vs public 256x256, which may cause algorithm failure
**Fix:** Investigate if algorithms properly handle resolution change; document the cause.

**H3. Spec ranges identical across tiers (same issue as CACTI)**
- All 3 tiers show the same spec_ranges in HDF5
- Only true_spec values differ
**Fix:** Define tier-specific spec_ranges.

**H4. PSNR_norm undefined in scoring formula**
**Fix:** Define normalization method explicitly.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Spatial resolution differs: public 256x256, dev/hidden 500x500 -- not documented on webpage |
| M2 | Measurement shape differs: public y(256,313), dev y(500,559), hidden y(500,555) -- dispersion offset varies |
| M3 | Rank inversion: SSR-L 1st on public but 2nd on dev/hidden; GAP-TV jumps to 1st on dev/hidden (more robust) |
| M4 | Missing references: SSR-L, MST-L, HDNet, PnP-HSICNN not cited |
| M5 | HDNet collapses to 10.96 dB on hidden (nearly random) suggesting catastrophic failure |
| M6 | Dataset sizes very large: dev 1.0 GB, hidden 1.0 GB -- may limit accessibility |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Placeholder links: /benchmark/sd_cassi/compete, /benchmark/sd_cassi/contribute |
| L2 | No gallery showing spectral reconstruction comparisons |
| L3 | SSIM computed per-band or on RGB rendering? Not specified |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | File | Size | Samples | Spatial | Spectral | y Shape |
|------|------|------|---------|---------|----------|---------|
| Public | sd_cassi_challenge_public.h5 | 139 MB | 10 | 256x256 | 28 bands | 256x313 |
| Dev | sd_cassi_challenge_dev.h5 | 1029 MB | 20 | 500x500 | 28 bands | 500x559 |
| Hidden | sd_cassi_challenge_hidden.h5 | 1038 MB | 20 | 500x500 | 28 bands | 500x555 |

**Total: 2.2 GB across 50 samples**

### HDF5 Schema

| Key | Shape (Public) | Shape (Dev/Hidden) | Dtype | Description |
|-----|----------------|-------------------|-------|-------------|
| x_true | (256,256,28) | (500,500,28) | float64 | GT hyperspectral cube |
| y | (256,313) | (500,~557) | float64 | Compressed snapshot |
| H_ideal | (256,256) | (500,500) | float64 | Ideal coded aperture mask |

### Spec Range Analysis (ISSUE: identical across tiers)

| Parameter | Public | Dev | Hidden |
|-----------|--------|-----|--------|
| mask_dx | [0.3, 0.7] | [0.3, 0.7] | [0.3, 0.7] |
| mask_dy | [0.1, 0.5] | [0.1, 0.5] | [0.1, 0.5] |
| mask_rotation | [0.0, 0.2] | [0.0, 0.2] | [0.0, 0.2] |
| dispersion_slope | [1.9, 2.15] | [1.9, 2.15] | [1.9, 2.15] |
| dispersion_axis | [0.0, 0.3] | [0.0, 0.3] | [0.0, 0.3] |

### Dataset Integrity Assessment: **PASS with WARNING** (spec ranges need tier differentiation)

---

## 3. Public Dataset Source Assessment

### Current Source: KAIST HSI (Choi et al., ICCV 2017) -- **EXCELLENT**

- 10 scenes from KAIST TSA_simu_data
- THE standard dataset in CASSI/SCI research (~500 citations)
- Used in virtually every CASSI paper since 2017
- 28 spectral bands (450-650nm), 256x256 spatial

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Public: Well-known? | EXCELLENT | KAIST HSI is the field standard |
| Public: Accepted by professors? | EXCELLENT | Universal adoption in CASSI community |
| Dev: Protected? | EXCELLENT | Procedural (seed=5000, K=6) |
| Hidden: Protected? | EXCELLENT | Procedural (seed=6000, K=8) |

### Recommendations

- KAIST source is ideal -- keep it
- Consider supplementing with CAVE (Columbia, 32 scenes) or ICVL (201 scenes) for diversity

---

## 4. Algorithm Coverage Assessment

### Currently Tested: 5 algorithms

SSR-L, MST-L, HDNet, GAP-TV, PnP-HSICNN

### Missing Famous/Recent Algorithms (MUST ADD)

| Priority | Algorithm | Year | Why |
|----------|-----------|------|-----|
| CRITICAL | TwIST | Classical | Universal CS baseline |
| CRITICAL | lambda-net | 2020 | First learned HSI reconstruction |
| HIGH | CST | 2022 | ECCV, state-of-the-art HSI Transformer |
| HIGH | DAUHST | 2022 | NeurIPS, degradation-aware unfolding |
| MEDIUM | DGSMP | 2021 | CVPR, Gaussian scale mixture prior |
| MEDIUM | Mask-guided SS-MLP | 2024 | Latest spatial-spectral MLP |
| MEDIUM | SCFNet | 2025 | End-to-end spatial-enhanced Transformer |

**Total gap: 9 algorithms missing across 5 categories**

---

## 5. Improvement Suggestions

1. Fix tier-specific spec_ranges (narrow -> medium -> wide)
2. Standardize spatial resolution or document 256 vs 500 difference
3. Investigate 18.5 dB public-to-dev performance drop
4. Add CST, DAUHST state-of-the-art Transformers
5. Add lambda-net foundational learned reconstruction
6. Add TwIST classical baseline
7. Investigate HDNet collapse on hidden (10.96 dB)
8. Sync webpage (5 params not 9, correct sample counts)
9. Add spectral metrics (SAM, ERGAS)
10. Add gallery with spectral band comparisons

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Fix spec_ranges to be tier-specific | Dataset team | TODO |
| CRITICAL | Sync webpage mismatch params (5 not 9) | Main server | TODO |
| CRITICAL | Investigate 18.5 dB public->dev drop | Algorithm team | TODO |
| CRITICAL | Add TwIST classical baseline | Algorithm team | TODO |
| HIGH | Add CST and DAUHST Transformers | Algorithm team | TODO |
| HIGH | Add lambda-net deep unfolding | Algorithm team | TODO |
| HIGH | Document resolution difference across tiers | Dataset team | TODO |
| HIGH | Define PSNR_norm formula | Main server | TODO |
| MEDIUM | Add spectral metrics (SAM, ERGAS) | Metrics team | TODO |
| MEDIUM | Investigate HDNet collapse | Algorithm team | TODO |
| LOW | Add gallery and missing references | Main server | TODO |

---

## Appendix: Key References

- Wagadarikar et al. "Single disperser design for CASSI." Appl. Opt. 47.10 (2008).
- Choi et al. "High-quality HSI reconstruction using spectral prior (KAIST)." SIGGRAPH Asia (2017).
- Miao et al. "lambda-net: Reconstruct HSI from snapshot measurement." ICCV (2019).
- Cai et al. "CST for HSI reconstruction." ECCV (2022).
- Cai et al. "DAUHST." NeurIPS (2022).
- Yuan et al. "Snapshot compressive imaging." IEEE SPM (2021).

---

*Comprehensive 6-point review on 2026-03-03.*
