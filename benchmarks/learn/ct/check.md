# Comprehensive Benchmark QA Check — CT

**URL:** https://pwm.platformai.org/benchmark/ct
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
| LOW      | 4     |

### HIGH Severity

**H1. Webpage image size 512x512 but local data is 362x362**
- Webpage claims 512x512 pixel images
- Local HDF5 files have x_true shape (362, 362)
- This is a significant discrepancy affecting algorithm design
**Fix:** Sync webpage to match actual data (362x362).

**H2. PSNR_norm undefined in scoring formula**
**Fix:** Define normalization method explicitly.

**H3. Mismatch ranges on webpage differ from local spec.json**
- Webpage: center_offset +/-4-9 px, angle_error +/-6.5-14 deg, beam_hardening -0.1-0.37, detector_tilt +/-2.5-5.5 deg
- Local spec.json: center_offset [-2,2] px, angle_error [-3,3] deg, beam_hardening [0,0.1], detector_tilt [-1,1] deg
- Webpage ranges are much wider than actual data
**Fix:** Sync webpage to match local spec.json values.

**H4. Webpage says 60 views but hidden tier has variable N_views (40-90)**
- README states hidden tier uses per-sample random view counts
- Webpage shows fixed 60 views
**Fix:** Document variable views on webpage.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Rank inversion: DuDoTrans 1st (public) -> 2nd (dev) -> 4th (hidden); DOLCE more robust |
| M2 | RED-CNN collapses from 5th (public, 31.14 dB) to 8th (hidden, 20.44 dB) |
| M3 | Missing references: DuDoTrans, DOLCE, Learned Primal-Dual, RED-CNN |
| M4 | Webpage says "120 kVp" but this is a parameter of polychromatic beam, not relevant for monochromatic simulation |
| M5 | Webpage says "25% dose" but noise model is Beer-Lambert + Poisson(I0=10000), not directly dose-related |
| M6 | Fan-beam geometry parameters on webpage don't match local README (different D_so, D_sd values) |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Placeholder links: /benchmark/ct/compete, /benchmark/ct/contribute |
| L2 | No alt-text on gallery images |
| L3 | SSIM window size not specified |
| L4 | Sparse-view artifact patterns not documented |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | File | Size | Samples |
|------|------|------|---------|
| Public | ct_challenge_public.h5 | ~50 MB | 11 |
| Dev | ct_challenge_dev.h5 | ~100 MB | 20 |
| Hidden | ct_challenge_hidden.h5 | ~100 MB | 20 |

### HDF5 Schema (verified from earlier inspection)

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| x_true | (362, 362) | float32 | GT attenuation map [0, 1] |
| sinogram_ideal | (60, 736) | float32 | Nepers, no mismatch |
| sinogram_measured | (60, 736) | float32 | Nepers, with mismatch |
| angles_nominal | (60,) | float32 | Radians |

### Source: LoDoPaB-CT (Leuschner et al., Scientific Data 2021) -- **EXCELLENT**

| Tier | LoDoPaB Split | Patients |
|------|--------------|----------|
| Public | Test | Test patients (11 slices) |
| Dev | Validation (first half) | Patients 0-63 (20 slices) |
| Hidden | Validation (second half) + adversarial | Patients 64-127 (20 slices) |

### Spec Range Nesting (from spec.json)

| Parameter | Public | Dev | Hidden | Status |
|-----------|--------|-----|--------|--------|
| center_offset_px | [-2, 2] | widens | widens | PASS |
| angle_error_deg | [-3, 3] | widens | widens | PASS |
| beam_hardening_beta | [0, 0.1] | widens | widens | PASS |
| detector_tilt_deg | [-1, 1] | widens | widens | PASS |

### Adversarial Modifications (Hidden Tier)

| Modification | Frequency | Challenge |
|---|---|---|
| Metal implants | 35% | High-density streaks, dynamic range |
| Low-contrast lesions | 30% | Subtle nodules, hepatic cysts |
| Calcifications | 20% | Punctate high-density spots |
| High-contrast bone | 15% | Extreme dynamic range |

### Dataset Integrity Assessment: **PASS**

---

## 3. Public Dataset Source Assessment

### LoDoPaB-CT: **EXCELLENT**

- Leuschner et al. (2021), Scientific Data, doi:10.1038/s41597-021-00893-z
- Based on LIDC/IDRI lung CT database (~200 citations for LoDoPaB, ~3,000 for LIDC)
- Zenodo record 3384092, CC BY 4.0 license
- Widely used in CT reconstruction research
- Accepted by professors and PhDs worldwide

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Public: Well-known? | EXCELLENT | LoDoPaB-CT is a standard benchmark |
| Public: Accepted by professors? | EXCELLENT | Published in Scientific Data, widely cited |
| Dev: Protected? | EXCELLENT | Different patients (0-63), augmented |
| Hidden: Protected? | EXCELLENT | Different patients (64-127) + adversarial mods (metal, lesions) |

---

## 4. Algorithm Coverage Assessment

### Currently Tested: 8 algorithms -- **BEST COVERAGE of any modality**

| # | Algorithm | Type | Notes |
|---|-----------|------|-------|
| 1 | DuDoTrans | Dual-domain Transformer | Top on public |
| 2 | DOLCE | Diffusion-based | Most robust across tiers |
| 3 | Learned Primal-Dual | Unrolled optimization | Adler & Oktem, strong on hidden |
| 4 | FBPConvNet | CNN post-processing | Classic DL baseline |
| 5 | RED-CNN | Residual encoder-decoder | Collapses on hidden |
| 6 | PnP-ADMM | Plug-and-play | Consistent performer |
| 7 | TV-ADMM | Total variation | Classical iterative |
| 8 | FBP | Analytical | Baseline |

### Missing Famous/Recent Algorithms

| Priority | Algorithm | Year | Why |
|----------|-----------|------|-----|
| HIGH | SART / SIRT | Classical | Standard iterative algebraic, missing from leaderboard |
| HIGH | ASD-POCS | 2008 | TV-constrained, Sidky & Pan (~2,500 citations) |
| HIGH | U-Net (basic) | 2017 | Universal DL baseline |
| MEDIUM | iRadonMAP | 2020 | Learned inverse Radon |
| MEDIUM | 3DGR-CT | 2025 | 3D Gaussian representation, newest approach |
| MEDIUM | DiffusionMBIR | 2023 | Diffusion + model-based, Chung et al. |
| LOW | CGLS | Classical | Conjugate gradient least squares |

### Algorithm Gap Analysis

CT has the best algorithm coverage among all modalities (8 algorithms spanning 5 categories). Main gaps:
- Classical iterative (SART/SIRT, ASD-POCS) -- expected for completeness
- 3D Gaussian / NeRF approaches (emerging 2024-2025)

**Total gap: 7 algorithms (less critical than other modalities)**

---

## 5. Improvement Suggestions

### 5.1 Dataset

1. **Fix webpage image size** (362x362, not 512x512)
2. **Sync webpage mismatch ranges** with local spec.json
3. **Document hidden tier variable views** (40-90 per sample)
4. **Consider adding AAPM Mayo dataset** for diverse anatomy (beyond lung)

### 5.2 Algorithms

5. **Add SART/SIRT iterative baseline** -- expected by CT community
6. **Add ASD-POCS** -- gold standard TV-regularized iterative
7. **Add 3DGR-CT (2025)** -- newest Gaussian approach
8. **Investigate RED-CNN collapse** on hidden tier (31 -> 20 dB)

### 5.3 Infrastructure

9. **Sync all webpage numbers with local data**
10. **Add per-sample view count to metadata** for hidden tier
11. **Add missing references** with DOIs

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Fix webpage image size (362x362 not 512x512) | Main server | TODO |
| CRITICAL | Sync webpage mismatch ranges with spec.json | Main server | TODO |
| CRITICAL | Define PSNR_norm formula | Main server | TODO |
| HIGH | Add SART/SIRT iterative baseline | Algorithm team | TODO |
| HIGH | Add ASD-POCS | Algorithm team | TODO |
| HIGH | Document variable views in hidden tier | Main server | TODO |
| MEDIUM | Add 3DGR-CT (2025) | Algorithm team | TODO |
| MEDIUM | Investigate RED-CNN collapse | Algorithm team | TODO |
| MEDIUM | Add missing references with DOIs | Main server | TODO |
| LOW | Add CGLS and iRadonMAP | Algorithm team | TODO |

---

## Appendix: Key References

- Leuschner et al. "LoDoPaB-CT." Scientific Data 8:109 (2021). doi:10.1038/s41597-021-00893-z
- Feldkamp, Davis & Kress. JOSA A 1:612-619 (1984).
- Sidky & Pan. "Image reconstruction in circular cone-beam CT." PMB 53:4777 (2008).
- Adler & Oktem. "Learned Primal-Dual Reconstruction." IEEE TMI 37.6 (2018).
- Chen et al. "RED-CNN." IEEE TMI 36.12 (2017).
- Jin et al. "FBPConvNet." IEEE TIP 26.9 (2017).
- Lin et al. "DuDoTrans: Dual-Domain Transformer." MICCAI (2022).
- DOLCE: Diffusion-based Low-dose CT Enhancement (2023).

---

*Comprehensive 6-point review on 2026-03-03. CT has the best algorithm coverage (8 methods) among all modalities with local data.*
