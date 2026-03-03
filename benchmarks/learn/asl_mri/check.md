# Comprehensive Benchmark QA Check -- ASL MRI

**URL:** https://pwm.platformai.org/benchmark/asl_mri
**HTTP Status:** 200
**Check Date:** 2026-03-03 (comprehensive 6-point review)
**Reviewer:** Manual deep analysis + web research

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

**H1. Dataset source citation is wrong -- references CT dataset instead of ASL MRI**
- Webpage cites: "AAPM Low-Dose CT Grand Challenge (McCollough et al., Med. Phys. 2017)"
- The AAPM Low-Dose CT Grand Challenge is a CT-specific dataset of contrast-enhanced abdominal CT scans
- An ASL MRI benchmark must use an ASL perfusion dataset (e.g., OASIS-3, HCP ASL, OpenNeuro ASL-BIDS, UK Biobank ASL)
- This is either a copy-paste error from the CT modality or a placeholder that was never updated
**Fix:** Replace with the actual dataset source used for ground-truth perfusion maps. If synthetic, state the simulation pipeline and reference atlas explicitly.

**H2. PSNR_norm is undefined in the scoring formula**
- Scoring formula: 0.4 x PSNR_norm + 0.4 x SSIM + 0.2 x (1 - ||y - Hx||/||y||)
- "PSNR_norm" normalization method is not defined anywhere on the page
- Users cannot reproduce the score without knowing the normalization range
**Fix:** Define PSNR_norm explicitly (e.g., PSNR_norm = clip(PSNR / 50, 0, 1) or min-max over leaderboard).

**H3. Data dimensions not specified**
- Image size, number of channels, voxel resolution, and FOV are not stated
- ASL data has unique dimensionality considerations: label/control pairs, multi-delay volumes, 3D vs 2D acquisition
- Users cannot design architectures without knowing input/output tensor shapes
**Fix:** Add explicit data dimensions: image shape (H x W or H x W x D), number of label-control pairs, number of post-labeling delays, voxel size.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Mismatch parameter ranges are inconsistently formatted: labeling_efficiency shows overlapping min/max (0.826-0.836 / 0.896-0.89) with unclear tier separation |
| M2 | t1_blood_error parameter has both negative and positive ranges (-2.4 to 4.6) but units listed as "dimensionless" -- should specify if this is a percentage or absolute T1 deviation in seconds |
| M3 | Only 3 scenes per tier (public/dev/hidden = 9 total) -- extremely small sample count for a perfusion imaging benchmark where inter-subject variability is high |
| M4 | Large performance gap between public and hidden tiers (e.g., PromptMR: 38.37 -> 29.29 dB, ~9 dB drop) suggests mismatch parameters may be too aggressive or data distribution shift is too extreme |
| M5 | L1-Wavelet (ESPIRiT) collapses catastrophically on hidden tier (27.95 -> 17.64 dB, ~10 dB drop) but this is not flagged or discussed |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | SSIM computation parameters (window size, data range, channel handling) not specified |
| L2 | No gallery/preview images showing ASL-specific features (CBF maps, label-control difference, transit delay maps) |
| L3 | All algorithm names appended with "+ gradient" but the meaning of this suffix is not explained |

---

## 2. Local Dataset Inspection

### File Inventory

**No local dataset found.** The directory `datasets/benchmark/asl_mri/` does not exist.

| Tier | Expected File | Status |
|------|---------------|--------|
| Public | asl_mri_challenge_public.h5 | MISSING |
| Dev | asl_mri_challenge_dev.h5 | MISSING |
| Hidden | asl_mri_challenge_hidden.h5 | MISSING |

### Schema Verification: NOT POSSIBLE

Without local data, the following cannot be verified:
- HDF5 key names and shapes
- Ground-truth value ranges (CBF maps should be ~0-100 mL/100g/min)
- Mismatch parameter ranges vs. spec.json
- Image size consistency with webpage claims
- Whether data represents true ASL (label-control pairs) or just CBF maps

### Dataset Integrity Assessment: **FAIL -- NO LOCAL DATA**

---

## 3. Public Dataset Source Assessment

### Cited Source: AAPM Low-Dose CT Grand Challenge -- **INCORRECT**

The cited source (McCollough et al., Med. Phys. 2017) is a CT dataset, not ASL MRI. This is a critical citation error.

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Public: Well-known? | FAIL | Wrong modality cited; actual source unknown |
| Public: Accepted by professors? | FAIL | Cannot assess without knowing the real dataset |
| Dev: Protected? | UNKNOWN | No local data to verify patient/subject separation |
| Hidden: Protected? | UNKNOWN | No local data to verify tier isolation |

### Recommended ASL MRI Datasets (if data needs to be sourced)

| Dataset | Subjects | Resolution | Availability | Notes |
|---------|----------|------------|--------------|-------|
| OASIS-3 | 1,378 | Multi-sequence incl. ASL | Open (CC BY 4.0) | Longitudinal, includes T1/T2/ASL/BOLD |
| HCP (Human Connectome Project) | ~3,000 ASL | High-res multi-delay | Restricted access | Gold standard perfusion dataset |
| UK Biobank ASL | ~100,000+ | 3D pCASL | Restricted access | Largest population-level ASL dataset |
| OpenNeuro ASL-BIDS datasets | Varies | Multi-site, multi-vendor | Open | BIDS-formatted, standardized |
| ADNI (Alzheimer's) | ~1,300 ASL | Standard clinical | Restricted | Disease-relevant perfusion data |

---

## 4. Algorithm Coverage Assessment

### Currently on Leaderboard: 8 algorithms

| # | Algorithm | Type | Public PSNR/SSIM | Dev PSNR/SSIM | Hidden PSNR/SSIM | Overall |
|---|-----------|------|------------------|---------------|-------------------|---------|
| 1 | PromptMR + gradient | Deep unrolled (Transformer) | 38.37 / 0.983 | 32.83 / 0.951 | 29.29 / 0.905 | 0.788 |
| 2 | E2E-VarNet + gradient | Deep unrolled (VarNet) | 38.01 / 0.982 | 32.78 / 0.951 | 29.07 / 0.901 | 0.785 |
| 3 | U-Net + gradient | CNN post-processing | 33.72 / 0.959 | 28.71 / 0.895 | 28.51 / 0.891 | 0.740 |
| 4 | PnP-DnCNN + gradient | Plug-and-play | 29.67 / 0.912 | 27.21 / 0.863 | 25.82 / 0.827 | 0.695 |
| 5 | ReconFormer + gradient | Recurrent Transformer | 30.56 / 0.925 | 26.52 / 0.846 | 23.67 / 0.757 | 0.672 |
| 6 | Score-MRI + gradient | Diffusion/score-based | 30.87 / 0.929 | 24.68 / 0.792 | 23.82 / 0.762 | 0.668 |
| 7 | Zero-Filled IFFT + gradient | Analytical baseline | 23.10 / 0.735 | 22.59 / 0.715 | 21.34 / 0.661 | 0.576 |
| 8 | L1-Wavelet (ESPIRiT) + gradient | Compressed sensing | 27.95 / 0.880 | 20.52 / 0.623 | 17.64 / 0.482 | 0.562 |

### Observations

1. **PromptMR and E2E-VarNet dominate** -- both are deep unrolled networks originally designed for k-space MRI reconstruction (fastMRI leaderboard top performers). PromptMR won ECCV 2024 Oral and CMRxRecon2024 1st place.
2. **U-Net is surprisingly robust** -- only 0.20 dB gap between dev and hidden, suggesting it generalizes well under mismatch.
3. **L1-Wavelet (ESPIRiT) collapses** on hidden tier (27.95 -> 17.64 dB) -- classical CS fails catastrophically under model mismatch.
4. **Score-MRI underperforms** relative to simpler methods, possibly due to distribution shift in the score function under mismatch.
5. **ReconFormer degrades significantly** (30.56 -> 23.67 dB, ~7 dB) despite being a state-of-the-art Transformer architecture.

### Discrepancy with Existing Auto-Generated check.md

The auto-generated check.md (from `scripts/check_modality.py`) reports only 4 methods (SwinMR, MoDL, GRAPPA, CS-Wavelet) while the live webpage shows 8 different methods. This is a significant discrepancy -- either the auto-check is stale or the webpage was updated after the check was run.

### Missing Famous/Recent Algorithms

| Priority | Algorithm | Year | Why Important for ASL MRI |
|----------|-----------|------|---------------------------|
| HIGH | SwinIR / SwinMR | 2021-2024 | Swin Transformer outperforms CNN-based methods for ASL denoising (Shou et al., MRM 2024); already shown superior to ResNet and DWAN specifically on ASL data |
| HIGH | DWAN (Dilated Wide Activation Network) | 2020 | Established ASL-specific denoising baseline; widely compared in ASL literature |
| HIGH | MoDL (Model-based Deep Learning) | 2019 | Aggarwal et al.; physics-informed unrolled network; standard MRI reconstruction baseline (listed in auto-check but missing from webpage) |
| HIGH | GRAPPA | 2002 | Standard parallel imaging reference for multi-coil MRI (listed in auto-check but missing from webpage) |
| MEDIUM | ASLRDB | 2025 | Purpose-built ASL architecture with Residual Dense Blocks; reduces acquisition time by 75% |
| MEDIUM | Transformer-KWIA | 2024 | Transformer with k-space weighted image average; state-of-the-art multi-delay ASL denoising |
| MEDIUM | Joint LM-DL (Jointly Learned Model-DL) | 2023 | Self-supervised ASL-specific joint learning approach |
| MEDIUM | CycleGAN / Unsupervised DL | 2020 | Kim et al.; unsupervised ASL MRI denoising and reconstruction, no paired training data needed |
| LOW | SENSE | 1999 | Classical parallel imaging, expected as baseline |
| LOW | Total Variation (TV) | Classical | Standard regularized iterative, expected as baseline |

### Algorithm Gap Analysis

The leaderboard has 8 general-purpose MRI reconstruction algorithms but is **missing all ASL-specific methods**. ASL MRI has unique challenges (very low SNR, label-control subtraction noise, transit delay effects, T1-dependent signal decay) that demand domain-specific algorithms. The absence of SwinIR (proven superior on ASL by Shou et al. 2024), DWAN, and ASLRDB is a significant gap.

Additionally, 4 methods appearing in the auto-generated check (SwinMR, MoDL, GRAPPA, CS-Wavelet) do not appear on the live webpage, suggesting an inconsistency between versions.

**Total gap: 10+ algorithms, including critically important ASL-specific methods**

---

## 5. Improvement Suggestions

### 5.1 Critical Data Fixes

1. **Fix dataset source citation** -- replace AAPM Low-Dose CT reference with actual ASL MRI data source
2. **Add data dimensions** -- specify image shape, number of label-control pairs, delay times, voxel resolution
3. **Define PSNR_norm** in the scoring formula with an explicit equation
4. **Create local dataset** -- download or generate `datasets/benchmark/asl_mri/` with public/dev/hidden tiers

### 5.2 Dataset Improvements

5. **Increase sample count** -- 3 scenes per tier is far too few for perfusion imaging; aim for 20+ per tier
6. **Use a recognized ASL dataset** as source (OASIS-3 or OpenNeuro ASL-BIDS collections)
7. **Clarify mismatch parameter units** -- t1_blood_error should have physical units (ms or %), labeling_efficiency ranges should be cleanly separated per tier
8. **Investigate public-to-hidden performance gap** -- 9+ dB drops suggest overly aggressive mismatch escalation

### 5.3 Algorithm Additions

9. **Add ASL-specific algorithms** -- SwinIR (ASL-tuned), DWAN, ASLRDB are essential for domain credibility
10. **Restore missing methods** -- MoDL, GRAPPA, SwinMR, CS-Wavelet appear in auto-check but not on webpage
11. **Add classical baselines** -- SENSE and TV regularization expected by MRI community
12. **Investigate L1-Wavelet collapse** -- 17.64 dB on hidden tier indicates fundamental failure mode

### 5.4 Page Quality

13. **Explain "+ gradient" suffix** on all algorithm names
14. **Add ASL-specific gallery images** showing CBF maps, label-control difference images, transit delay maps
15. **Add references with DOIs** for all leaderboard algorithms

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Fix dataset source citation (CT -> ASL MRI) | Main server | TODO |
| CRITICAL | Define PSNR_norm formula explicitly | Main server | TODO |
| CRITICAL | Add data dimensions (image shape, delays, voxel size) | Main server | TODO |
| CRITICAL | Create local dataset in datasets/benchmark/asl_mri/ | Data team | TODO |
| HIGH | Increase sample count from 3 to 20+ per tier | Data team | TODO |
| HIGH | Add SwinIR (ASL-tuned) to leaderboard -- proven SOTA for ASL denoising | Algorithm team | TODO |
| HIGH | Add DWAN baseline -- standard ASL DL comparison | Algorithm team | TODO |
| HIGH | Reconcile auto-check vs webpage algorithm lists (SwinMR/MoDL/GRAPPA missing from page) | Main server | TODO |
| MEDIUM | Add ASLRDB (2025 ASL-specific architecture) | Algorithm team | TODO |
| MEDIUM | Add Transformer-KWIA (2024 multi-delay ASL SOTA) | Algorithm team | TODO |
| MEDIUM | Clarify mismatch parameter units and tier ranges | Main server | TODO |
| MEDIUM | Investigate L1-Wavelet ESPIRiT collapse (27.95 -> 17.64 dB on hidden) | Algorithm team | TODO |
| MEDIUM | Explain "+ gradient" suffix on algorithm names | Main server | TODO |
| LOW | Add SENSE and TV classical baselines | Algorithm team | TODO |
| LOW | Add gallery images with ASL-specific visualizations (CBF maps) | Main server | TODO |
| LOW | Add DOI references for all algorithms | Main server | TODO |

---

## Appendix: Key References

- Shou et al. "Transformer-based deep learning denoising of single and multi-delay 3D arterial spin labeling." MRM 91(4):1542-1555 (2024). doi:10.1002/mrm.29887
  -- Demonstrated SwinIR outperforms ResNet and DWAN on ASL denoising
- Guo et al. "Optimization of deep learning-based denoising for arterial spin labeling: Effects of averaging and training strategies." MRM (2025). doi:10.1002/mrm.70013
  -- Windowed averaging + DL denoising for clinical ASL
- Kim et al. "Arterial spin labeling MR image denoising and reconstruction using unsupervised deep learning." MRM 83(4):1369-1382 (2020). doi:10.1002/mrm.28012
  -- CycleGAN unsupervised approach for ASL
- Xie et al. "Denoising arterial spin labeling perfusion MRI with deep machine learning." MRI 68:68-76 (2020). doi:10.1016/j.mri.2019.12.005
  -- Early DnCNN-based ASL denoising
- Bapst et al. "A deep learning architecture for ASL MRI to improve SNR with short acquisition time." SIVP (2025). doi:10.1007/s11760-025-04860-8
  -- ASLRDB dual-pathway CNN, reduces acquisition by 75%
- Sriram et al. "End-to-End Variational Networks for Accelerated MRI Reconstruction." MICCAI (2020). doi:10.1007/978-3-030-59713-9_7
  -- E2E-VarNet, fastMRI SOTA
- Bai et al. "PromptMR: Prompting for Dynamic and Multi-Contrast MRI Reconstruction." ECCV Oral (2024).
  -- PromptMR+, 1st place CMRxRecon2024
- Guo et al. "ReconFormer: Accelerated MRI Reconstruction Using Recurrent Transformer." IEEE TMI 43(1):134-147 (2024). doi:10.1109/TMI.2023.3293842
  -- Lightweight recurrent Transformer (1.1M params)
- Chung & Ye. "Score-based diffusion models for accelerated MRI." MedIA 80:102479 (2022). doi:10.1016/j.media.2022.102479
  -- Score-MRI diffusion-based reconstruction
- McCollough et al. "Low-dose CT for the detection and classification of metastatic liver lesions." Med. Phys. 44(10):e339-e352 (2017). doi:10.1002/mp.12345
  -- AAPM CT dataset (INCORRECTLY cited as ASL MRI source on webpage)
- Amukotuwa et al. "ASL-BIDS, the brain imaging data structure extension for arterial spin labeling." Sci. Data 9:543 (2022). doi:10.1038/s41597-022-01615-9
  -- ASL data standardization format

---

*Comprehensive 6-point review on 2026-03-03. ASL MRI benchmark has 3 CRITICAL issues: wrong dataset source citation (CT instead of ASL), undefined PSNR_norm, and missing data dimensions. No local dataset exists. Leaderboard has 8 general MRI algorithms but zero ASL-specific methods (SwinIR, DWAN, ASLRDB). Auto-generated check.md is stale -- reports different algorithms than live webpage.*