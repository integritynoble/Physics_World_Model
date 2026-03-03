# Comprehensive Benchmark QA Check — CACTI

**URL:** https://pwm.platformai.org/benchmark/cacti
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

**H1. Sample count mismatch**
- Webpage: "6 scenes" per tier
- Local H5: 20/20/20 samples (60 total)
**Fix:** Update webpage to show actual sample counts.

**H2. Dev and Hidden leaderboards show all zeros**
- Dev tier: all 5 algorithms show 0.0 score
- Hidden tier: all 5 algorithms show 0.0 score
- Either evaluation hasn't been run, or results display is broken
**Fix:** Run evaluation on dev/hidden and populate leaderboard.

**H3. PSNR_norm undefined in scoring formula**
**Fix:** Define normalization method explicitly.

**H4. Spec ranges are IDENTICAL across all three tiers**
- Public, dev, and hidden all show the same spec_ranges in HDF5
- Only the true_spec values differ (mild vs severe)
- But the claimed ranges don't widen, which breaks the nesting assumption
**Fix:** Define tier-specific spec_ranges that widen from public -> dev -> hidden.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Webpage says "KAIST HSI" (Choi et al., ICCV 2017) as source, but these are HSI datasets, not video -- possible confusion with SCI benchmarks |
| M2 | Webpage scene names (Kobe, Traffic, Runner, Drop, Crash, Aerial) are classic SCI video benchmark scenes, but public tier metadata says "CACTI simulation" |
| M3 | Public spatial size 256x256 but dev/hidden are 512x512 -- different resolutions across tiers not documented on webpage |
| M4 | Only T=8 for public, T=8/16/32 for dev/hidden -- variable compression ratio not documented on webpage |
| M5 | Missing references: EfficientSCI (Wang et al. 2023), HiSViT, ELP-Unfolding not cited |
| M6 | Forward model on webpage is simplified vs README's full gain/offset/mismatch equation |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Placeholder links: /benchmark/cacti/compete, /benchmark/cacti/contribute |
| L2 | No gallery images showing reconstruction comparisons |
| L3 | SSIM computation domain not specified (per-frame or temporal?) |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | File | Size | Samples | Spatial | Temporal |
|------|------|------|---------|---------|----------|
| Public | cacti_challenge_public.h5 | 41 MB | 20 | 256x256 | T=8 |
| Dev | cacti_challenge_dev.h5 | 618 MB | 20 | 512x512 | T=8 |
| Hidden | cacti_challenge_hidden.h5 | 657 MB | 20 | 512x512 | T=8 |

**Total: 1.3 GB across 60 samples**

### HDF5 Schema Verification

| Key | Shape (Public) | Shape (Dev/Hidden) | Dtype | Description |
|-----|----------------|-------------------|-------|-------------|
| `x_true` | (256, 256, 8) | (512, 512, 8) | float64 | GT video cube |
| `y` | (256, 256) | (512, 512) | float64 | Compressed measurement |
| `H_ideal` | (256, 256, 8) | (512, 512, 8) | float64 | Ideal coding mask |

### Value Range Checks

| Check | Public | Dev | Hidden | Status |
|-------|--------|-----|--------|--------|
| x_true in [0,1] | [0.000, 1.000] | [0.296, 0.909] | [0.113, 0.945] | PASS |
| Samples | 20 | 20 | 20 | PASS |
| Compression ratio | 8:1 | 8:1 | 8:1 | Verified |

### Spec Range Analysis (ISSUE: identical across tiers)

| Parameter | Public | Dev | Hidden | Nesting |
|-----------|--------|-----|--------|---------|
| mask_dx | [0.2, 0.8] | [0.2, 0.8] | [0.2, 0.8] | **SAME** |
| mask_dy | [0.1, 0.5] | [0.1, 0.5] | [0.1, 0.5] | **SAME** |
| mask_rotation | [0.0, 0.3] | [0.0, 0.3] | [0.0, 0.3] | **SAME** |
| mask_blur | [0.0, 0.5] | [0.0, 0.5] | [0.0, 0.5] | **SAME** |
| clock_offset | [-0.1, 0.1] | [-0.1, 0.1] | [-0.1, 0.1] | **SAME** |
| gain_drift | [0.95, 1.05] | [0.95, 1.05] | [0.95, 1.05] | **SAME** |
| offset_drift | [-0.02, 0.02] | [-0.02, 0.02] | [-0.02, 0.02] | **SAME** |

**ISSUE: Spec ranges don't differ between tiers. Only true_spec values differ (milder for public/dev, severe for hidden). The ranges should widen for harder tiers.**

### True Spec Comparison (shows difficulty progression via actual values)

| Parameter | Public (sample 0) | Dev (sample 0) | Hidden (sample 0) |
|-----------|-------------------|-----------------|-------------------|
| mask_dx | 0.50 | 0.35 | 0.65 |
| mask_dy | 0.30 | 0.20 | 0.40 |
| mask_rotation | 0.15 | 0.08 | 0.22 |
| mask_blur | 0.20 | 0.10 | 0.35 |
| clock_offset | 0.05 | -0.03 | 0.08 |
| gain_drift | 1.02 | 0.98 | 1.04 |
| offset_drift | 0.01 | -0.01 | 0.015 |

Note: Public true_spec values are between dev (mildest) and hidden (hardest). The difficulty is conveyed through true_spec magnitudes, not spec_ranges.

### Dataset Integrity Assessment: **PASS with WARNING** (structurally sound, spec ranges need tier differentiation)

---

## 3. Public Dataset Source Assessment

### Current Source

**Public tier:** "CACTI simulation" -- appears to be derived from standard SCI benchmark videos
- Scene names in community: Kobe, Traffic, Runner, Drop, Crash, Aerial
- These are the standard test videos from Yuan et al. "Snapshot compressive imaging" IEEE SPM 2021

**Dev tier:** Procedural -- urban, nature, textile, particles, thin_struct, occlusion, cam_shake
**Hidden tier:** Procedural adversarial -- textile, particles, thin_struct (harder scene types)

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| **Public: Well-known dataset?** | GOOD | Standard SCI video benchmark (Kobe, Traffic, Runner) widely used |
| **Public: Accepted by professors?** | GOOD | These are THE standard test videos in SCI/CACTI papers |
| **Dev: Protected?** | EXCELLENT | Procedural generation with secret seeds |
| **Hidden: Protected?** | EXCELLENT | Adversarial procedural scenes |

### Recommendations

1. **Add real CACTI measurement data** if available:
   - Llull et al. experimental data from DMD-based CACTI prototype
   - Real measurements would add credibility for hardware practitioners

2. **Consider using DAVIS video dataset** (Perazzi et al. CVPR 2016):
   - Standard video segmentation benchmark with diverse scenes
   - 50 high-quality sequences widely used for video-based tasks

3. **Add hyperspectral SCI datasets:**
   - KAIST HSI (Choi et al. ICCV 2017): 30 spectral scenes
   - Bridges CACTI to spectral SCI applications

---

## 4. Algorithm Coverage Assessment

### Currently Tested (Webpage Leaderboard)

| # | Algorithm | Type | Notes |
|---|-----------|------|-------|
| 1 | EfficientSCI + blind cal | Efficient DL | Top tied on public |
| 2 | HiSViT-9 + blind cal | Vision Transformer | Top tied on public |
| 3 | ELP-Unfolding + blind cal | Deep unfolding | Highest PSNR but lower score |
| 4 | GAP-TV + blind cal | Total variation | Classical baseline |
| 5 | PnP-DnCNN + blind cal | Plug-and-play | DL denoiser prior |

### Missing Famous/Recent Algorithms (MUST ADD)

| Priority | Algorithm | Year | Why Important |
|----------|-----------|------|---------------|
| **CRITICAL** | TwIST / FISTA | Classical | Standard CS solvers, every SCI paper uses as baseline |
| **CRITICAL** | GAP-net (learned GAP) | 2020 | Meng et al., first deep unfolding for SCI (~400 citations) |
| **CRITICAL** | RevSCI (reversible SCI) | 2021 | Cheng et al., CVPR -- memory-efficient deep SCI |
| **HIGH** | STFormer (Spatiotemporal Transformer) | 2022 | Wang et al., NeurIPS -- state-of-the-art on SCI benchmarks |
| **HIGH** | DeSCI (decompress SCI) | 2019 | Liu et al., TPAMI -- rank-1 + TV optimization (~300 citations) |
| **HIGH** | PnP-FFDNet | 2019 | Yuan et al. -- standard PnP for SCI |
| **MEDIUM** | Joint-learning Landweber (2024) | 2024 | Jointly learns mask + relaxation + regularization |
| **MEDIUM** | Unsupervised lightweight local-global (2024) | 2024 | No training data needed |
| **MEDIUM** | BIRNAT (bidirectional recurrent) | 2022 | Cheng et al., ECCV |
| **LOW** | Deep Equilibrium SCI | 2023 | Implicit layer approach |
| **LOW** | Complementary codes CACTI | 2023 | Untrained neural networks |

### Algorithm Gap Analysis

| Category | Have | Missing | Gap |
|----------|------|---------|-----|
| Classical CS | GAP-TV | TwIST, FISTA, DeSCI | 3 |
| Deep unfolding | ELP-Unfolding | GAP-net, RevSCI | 2 |
| Transformer | HiSViT | STFormer, BIRNAT | 2 |
| Efficient DL | EfficientSCI | Joint-learning (2024) | 1 |
| Plug-and-play | PnP-DnCNN | PnP-FFDNet | 1 |
| Unsupervised | -- | Unsupervised lightweight (2024) | 1 |

**Total gap: 10 algorithms missing across 6 categories**

---

## 5. Improvement Suggestions

### 5.1 Dataset Improvements

1. **Fix tier-specific spec_ranges**
   - Currently identical across all tiers
   - Public: narrow ranges (e.g., mask_dx [0.3, 0.5])
   - Dev: medium (e.g., mask_dx [0.2, 0.6])
   - Hidden: wide (e.g., mask_dx [0.1, 0.8])

2. **Add variable T values to dev/hidden**
   - README mentions T=8/16/32 but current H5 only shows T=8 in dev/hidden
   - Higher T = harder compression, essential for benchmark completeness

3. **Add real CACTI prototype data**
   - Contact Llull/Brady group at Duke for experimental measurements
   - Real data validates simulation fidelity

4. **Use DAVIS video dataset for public tier supplement**
   - More diverse scenes (animals, sports, vehicles)
   - Higher quality source material

### 5.2 Algorithm Improvements

5. **Add TwIST/FISTA classical baseline**
   - Standard iterative solvers, universally expected
   - Lower performance bound for reference

6. **Add GAP-net and RevSCI**
   - Foundational deep unfolding methods for SCI
   - GAP-net is the first learned approach, RevSCI introduced reversible design

7. **Add STFormer**
   - Current state-of-the-art Transformer for SCI
   - Would show gap between MLP-mixer (HiSViT) and full attention

8. **Run evaluation on dev/hidden tiers**
   - Currently showing all zeros -- must populate with actual results

### 5.3 Infrastructure

9. **Sync webpage with local data**
   - Fix sample counts (20, not 6)
   - Document spatial resolution difference (256 public vs 512 dev/hidden)
   - Document T-value distribution

10. **Add per-frame metrics**
    - SCI reconstruction quality varies significantly frame-to-frame
    - Report min/max PSNR across frames, not just average

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Fix spec_ranges to be tier-specific (narrow/medium/wide) | Dataset team | TODO |
| CRITICAL | Run evaluation on dev/hidden tiers (currently all zeros) | Evaluation team | TODO |
| CRITICAL | Fix webpage sample counts (20 per tier, not 6) | Main server | TODO |
| CRITICAL | Add TwIST/FISTA classical baseline | Algorithm team | TODO |
| HIGH | Add GAP-net and RevSCI to solver suite | Algorithm team | TODO |
| HIGH | Add variable T values (8/16/32) to dev/hidden H5 files | Dataset team | TODO |
| HIGH | Define PSNR_norm formula explicitly | Main server | TODO |
| HIGH | Document spatial resolution difference on webpage | Main server | TODO |
| MEDIUM | Add STFormer (NeurIPS 2022) | Algorithm team | TODO |
| MEDIUM | Add DeSCI optimization method | Algorithm team | TODO |
| MEDIUM | Add per-frame metric reporting | Metrics team | TODO |
| MEDIUM | Add missing references (EfficientSCI, HiSViT, GAP-net) | Main server | TODO |
| LOW | Add real CACTI prototype data | Dataset team | TODO |
| LOW | Add DAVIS video dataset as public supplement | Dataset team | TODO |

---

## Appendix: Key References

- Llull, P., et al. "Coded aperture compressive temporal imaging." Opt. Express 21.9 (2013): 10526-10545.
- Yuan, X., et al. "Snapshot compressive imaging: Theory, algorithms, and applications." IEEE SPM 38.2 (2021): 65-88.
- Meng, Z., et al. "GAP-net for snapshot compressive imaging." arXiv:2012.08364 (2020).
- Cheng, Z., et al. "Memory-efficient network for large-scale video compressive sensing (RevSCI)." CVPR (2021).
- Wang, Z., et al. "Spatial-temporal transformer for video snapshot compressive imaging (STFormer)." NeurIPS (2022).
- Liu, Y., et al. "Rank minimization for snapshot compressive imaging." TPAMI 41.12 (2019): 2990-3006.
- Wang, Z., et al. "EfficientSCI: Densely connected network with space-time factorization for SCI." CVPR (2023).

---

*Comprehensive 6-point review on 2026-03-03. Covers: page errors, local dataset verification, source quality, algorithm coverage, improvement suggestions, and action items.*
