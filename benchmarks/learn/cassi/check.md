# Comprehensive Benchmark QA Check — Coded Aperture Snapshot Spectral Imaging (CASSI)

**URL:** https://pwm.platformai.org/benchmark/cassi
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

**M1. Algorithm catalog not yet populated**
- No validated algorithms assigned to this modality
**Status:** Awaiting algorithm selection and validation

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

**Display Name:** Coded Aperture Snapshot Spectral Imaging (CASSI)

**Physics Class:** spectral_coding
**Forward Model:** coded_aperture_dispersion
**Noise Model:** gaussian

### Dataset Integrity Assessment: TODO

---

## 3. Public Dataset Source Assessment

### Canonical Datasets

- CAVE (Columbia, 32 scenes, 512x512x31)
- KAIST (30 scenes, 2704x3376x28)
- ARAD_1K (1000 hyperspectral images)

### Assessment: TODO

To be completed upon dataset publication.

---

## 4. Algorithm Coverage Assessment

### Algorithm Coverage: TODO

Algorithm catalog not yet populated for this modality.

### Known Gaps

To be completed during algorithm development phase.

---

## 5. Improvement Suggestions

### Priority Actions

2. **Select and validate algorithms** — Curate domain-appropriate methods
3. **Validate metrics** — Ensure PSNR/SSIM/consistency measures are appropriate
4. **Document physics** — Add to modality database with calibration parameters
5. **Define mismatch modes** — mask_misalignment, dispersion_curve_error, spectral_response_drift etc.

---

## 6. Action Items

| Priority | Action | Status |
|----------|--------|--------|
| CRITICAL | Select 4+ algorithms (Classical, PnP, DL, Transformer) | TODO |
| HIGH | Validate assessment metrics | TODO |
| HIGH | Complete modality database entry | TODO |
| MEDIUM | Add missing references | TODO |
| LOW | Optimize gallery previews | TODO |

---

## Appendix: Key References

- Wagadarikar et al., 'Single disperser design for coded aperture snapshot spectral imaging', Applied Optics 47, B44-B51 (2008)
- Cai et al., 'Mask-guided Spectral-wise Transformer (MST++)', CVPRW 2022


*Automated 6-point review on 2026-03-03 — Coded Aperture Snapshot Spectral Imaging (CASSI)*
