# Comprehensive Benchmark QA Check — SPC-Kronecker

**URL:** https://pwm.platformai.org/benchmark/spc_kronecker
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
| MEDIUM   | 5     |
| LOW      | 3     |

### HIGH Severity

**H1. Sample count mismatch**
- Webpage: "20 scenes" per tier (60 total)
- Local H5: 11 public / 20 dev / 20 hidden (51 total)
**Fix:** Update webpage to match actual sample counts (11/20/20).

**H2. Spec ranges identical across tiers**
- All 3 tiers: gain_decay_alpha [0.001, 0.01], noise_sigma [0.01, 0.05]
- Difficulty is conveyed only through true_spec values, not ranges
**Fix:** Define tier-specific spec_ranges that widen.

**H3. PSNR_norm undefined in scoring formula**
**Fix:** Define normalization method explicitly.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Webpage claims source "KAIST HSI" but local public data is Set11 grayscale images (README confirms) |
| M2 | Webpage shows mismatch range gain_decay_alpha [0.0005, 0.0125] but local H5 shows [0.001, 0.01] |
| M3 | ISTA-Net collapses from dev (26.05 dB, rank 5) to hidden (19.99 dB, rank 6) -- no explanation |
| M4 | PnP-BM3D consistently worst -- unusual for a well-established denoiser prior |
| M5 | Block size 33x33 is non-standard (powers of 2 more common: 32x32) |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Placeholder links: /benchmark/spc_kronecker/compete, /benchmark/spc_kronecker/contribute |
| L2 | No gallery images |
| L3 | Image size 231x231 (7x33 blocks) is unusual |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | File | Size | Samples | Blocks |
|------|------|------|---------|--------|
| Public | spc_kronecker_challenge_public.h5 | 27 MB | 11 | 49 (7x7) |
| Dev | spc_kronecker_challenge_dev.h5 | 49 MB | 20 | 49 |
| Hidden | spc_kronecker_challenge_hidden.h5 | 49 MB | 20 | 49 |

### HDF5 Schema

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| x_true | (231, 231) | float32/float64 | GT grayscale image |
| y | (49, 272) | float64 | Compressed measurements |
| H_ideal | (272, 1089) | float64 | Sensing matrix (Phi) |

### Compression Analysis

- Block size: 33x33 = 1089 pixels per block
- Measurements per block: 272
- Compression ratio: 272/1089 = 24.97% (4x compression)
- Image: 231x231 = 7x7 = 49 blocks

### Spec Range Analysis (identical across tiers)

| Parameter | All Tiers | True (Public) | True (Hidden) |
|-----------|-----------|---------------|---------------|
| gain_decay_alpha | [0.001, 0.01] | 0.0065 | 0.0093 |
| noise_sigma | [0.01, 0.05] | 0.0106 | 0.0432 |

**Hidden uses much higher noise and gain decay -- difficulty comes from true_spec, not ranges.**

### Dataset Integrity Assessment: **PASS** (structurally sound, spec ranges need differentiation)

---

## 3. Public Dataset Source Assessment

### Current Source: Set11 (11 classic test images) -- **GOOD**

- Standard CS/image processing test images (Lena, Barbara, Peppers, etc.)
- Widely used but considered outdated by modern standards
- Grayscale, 256x256 -> 231x231 (7x7 blocks of 33)

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Public: Well-known? | GOOD | Set11 is a classic CS benchmark set |
| Public: Accepted by professors? | MODERATE | Dated; modern benchmarks prefer BSD68, Urban100, DIV2K |
| Dev: Protected? | EXCELLENT | Procedural (fBm + textures + objects) |
| Hidden: Protected? | EXCELLENT | Adversarial (stripes + edges + HDR + checkerboard) |

### Recommendations

1. **Consider supplementing with BSD68 or Urban100** for modern credibility
2. **Add real SPC measurements** from DMD-based prototype if available
3. **Note: "Lena" image is controversial** -- many journals now ban it

---

## 4. Algorithm Coverage Assessment

### Currently Tested: 6 algorithms

PnP-DRUNet, FISTA-TV (tuned), FISTA-TV (paper), HATNet+FISTA-TV, ISTA-Net, PnP-BM3D

### Missing Famous/Recent Algorithms (MUST ADD)

| Priority | Algorithm | Year | Why |
|----------|-----------|------|-----|
| CRITICAL | OMP / CoSaMP | Classical | Greedy CS solvers, fundamental baselines |
| CRITICAL | ISTA / FISTA (standard) | Classical | Already have FISTA-TV, but pure ISTA needed |
| HIGH | ADMM-Net / ISTA-Net+ | 2018/2019 | Deep unfolding, Zhang et al. TPAMI |
| HIGH | DGUNet (Deep Generalized Unfolding) | 2022 | Mou et al., CVPR |
| HIGH | Dual-Scale Transformer (DST) | 2024 | CVPR 2024, large-scale SPI |
| MEDIUM | AMP-Net | 2021 | Approximate message passing with DL |
| MEDIUM | TransCS | 2022 | Transformer for CS reconstruction |
| MEDIUM | CST-UNet | 2025 | CS Transformer Unfolding Network |
| LOW | DCAN (Deep CS Attention Network) | 2020 | Attention-based CS |

### Algorithm Gap Analysis

| Category | Have | Missing | Gap |
|----------|------|---------|-----|
| Greedy CS | -- | OMP, CoSaMP | 2 |
| Iterative CS | FISTA-TV, ISTA-Net | Standard ISTA/FISTA | 1 |
| Deep unfolding | HATNet | ADMM-Net, ISTA-Net+, DGUNet | 3 |
| Transformer | -- | DST (CVPR 2024), TransCS, CST-UNet | 3 |
| Plug-and-play | PnP-DRUNet, PnP-BM3D | -- | 0 |

**Total gap: 9 algorithms missing across 4 categories**

---

## 5. Improvement Suggestions

1. **Fix tier-specific spec_ranges**
2. **Fix webpage source claim** (Set11, not KAIST HSI)
3. **Add OMP/CoSaMP classical baselines** -- fundamental for CS benchmarks
4. **Add DST (CVPR 2024)** -- latest Transformer for SPC
5. **Add ADMM-Net / ISTA-Net+** -- deep unfolding standard
6. **Consider BSD68 / Urban100** for public tier modernization
7. **Add block-overlap option** for hidden tier (harder reconstruction)
8. **Add variable compression ratios** (10%, 25%, 50%) across tiers
9. **Sync webpage mismatch ranges with actual H5 data**
10. **Replace controversial "Lena" image** in test set

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Fix spec_ranges to be tier-specific | Dataset team | TODO |
| CRITICAL | Fix webpage source (Set11 not KAIST) | Main server | TODO |
| CRITICAL | Add OMP/CoSaMP classical baselines | Algorithm team | TODO |
| HIGH | Add DST (CVPR 2024) Transformer | Algorithm team | TODO |
| HIGH | Add ADMM-Net / ISTA-Net+ deep unfolding | Algorithm team | TODO |
| HIGH | Fix webpage sample counts (11/20/20) | Main server | TODO |
| HIGH | Define PSNR_norm formula | Main server | TODO |
| MEDIUM | Modernize public images (BSD68/Urban100) | Dataset team | TODO |
| MEDIUM | Add variable compression ratios | Dataset team | TODO |
| LOW | Remove/replace Lena image | Dataset team | TODO |
| LOW | Add gallery images | Main server | TODO |

---

## Appendix: Key References

- Duarte et al. "Single-pixel imaging via compressive sampling." IEEE SPM 25.2 (2008): 83-91.
- Zhang & Ghanem. "ISTA-Net: Interpretable optimization-inspired deep network." CVPR (2018).
- Mou et al. "Deep generalized unfolding networks (DGUNet)." CVPR (2022).
- Qu et al. "Dual-Scale Transformer for Large-Scale Single-Pixel Imaging." CVPR (2024).

---

*Comprehensive 6-point review on 2026-03-03.*
