# Comprehensive Benchmark QA Check — Coded Aperture Compressive Temporal Imaging (CACTI)

**URL:** https://pwm.platformai.org/benchmark/cacti
**HTTP Status:** TBD (check on deployment)
**Check Date:** 2026-03-03 (automated 6-point review)
**Reviewer:** Automated generator + modality database

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
| HIGH     | 0     |
| MEDIUM   | 2     |
| LOW      | 2     |


### MEDIUM Severity


### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Documentation may need updates as benchmark matures |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | File | Size | Samples | Status |
|------|------|------|---------|--------|
| Public | {variant}_challenge_public.h5 | ~50 MB | TBD | Check GCS |
| Dev | {variant}_challenge_dev.h5 | ~100 MB | TBD | Check GCS |
| Hidden | {variant}_challenge_hidden.h5 | ~100 MB | TBD | Blocked |

### Modality Information

**Display Name:** Coded Aperture Compressive Temporal Imaging (CACTI)

**Physics Class:** temporal_coding
**Forward Model:** coded_aperture_temporal
**Noise Model:** gaussian

### Dataset Integrity Assessment: TODO

---

## 3. Public Dataset Source Assessment

### Canonical Datasets

- Kobe, Runner, Drop, Traffic (grayscale SCI benchmarks)
- DAVIS 2017 (adapted for SCI simulation)

### Assessment: TODO

To be completed upon dataset publication.

---

## 4. Algorithm Coverage Assessment

### Currently Tested: 5 algorithms

| # | Algorithm | Type | Source |
|---|-----------|------|--------|
| 1 | GAP-TV | Classical | InverseNet |
| 2 | PnP-FFDNet | PnP | InverseNet |
| 3 | ELP-Unfolding | Deep Unfolding | ECCV 2022 |
| 4 | EfficientSCI | Deep Learning | CVPR 2023 |
| 5 | HiSViT-9 | Transformer | ECCV 2024 |

### Known Gaps

To be completed during algorithm development phase.

---

## 5. Improvement Suggestions

### Priority Actions

3. **Validate metrics** — Ensure PSNR/SSIM/consistency measures are appropriate
4. **Document physics** — Add to modality database with calibration parameters
5. **Define mismatch modes** — mask_shift_error, motion_blur_within_frame, mask_diffraction etc.

---

## 6. Action Items

| Priority | Action | Status |
|----------|--------|--------|
| HIGH | Validate assessment metrics | TODO |
| HIGH | Complete modality database entry | TODO |
| MEDIUM | Add missing references | TODO |
| MEDIUM | Identify algorithm gaps | TODO |
| LOW | Optimize gallery previews | TODO |

---

## Appendix: Key References

- Llull et al., 'Coded aperture compressive temporal imaging', Optics Express 19, 10526 (2011)
- Yuan et al., 'Generalized alternating projection based total variation minimization (GAP-TV)', IEEE ICIP 2016
- Wang et al., 'Spatial-Temporal Transformer for Video Snapshot Compressive Imaging (STFormer)', ECCV 2022

## Algorithm References

- CVPR 2023
- ECCV 2022
- ECCV 2024
- InverseNet

*Automated 6-point review on 2026-03-03 — Coded Aperture Compressive Temporal Imaging (CACTI)*
