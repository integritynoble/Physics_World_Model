# Comprehensive Benchmark QA Check — Lensless (Diffuser Camera) Imaging

**URL:** https://pwm.platformai.org/benchmark/lensless
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
| HIGH     | 1     |
| MEDIUM   | 1     |
| LOW      | 1     |

### HIGH Severity

**H1. Benchmark page not yet live**
- This modality is in the database but the challenge dataset is not yet available
**Status:** Awaiting challenge data generation and deployment

### MEDIUM Severity


### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Documentation may need updates as benchmark matures |

---

## 2. Local Dataset Inspection

### File Inventory

No local challenge dataset currently available.

Status: Awaiting benchmark dataset generation.

### Modality Information

**Display Name:** Lensless (Diffuser Camera) Imaging

**Physics Class:** lensless_computational
**Forward Model:** psf_convolution_or_linear_operator
**Noise Model:** poisson_gaussian

### Dataset Integrity Assessment: TODO

---

## 3. Public Dataset Source Assessment

### Canonical Datasets

- DiffuserCam lensless mirflickr dataset (Monakhova et al.)
- PhlatCam benchmark (Boominathan et al., IEEE TPAMI 2022)

### Assessment: TODO

To be completed upon dataset publication.

---

## 4. Algorithm Coverage Assessment

### Currently Tested: 4 algorithms

| # | Algorithm | Type | Source |
|---|-----------|------|--------|
| 1 | Wiener-ADMM | Classical | Antipa et al., Optica 2018 |
| 2 | PnP-ADMM | PnP | Monakhova et al., Opt. Express 2019 |
| 3 | FlatNet | Deep Learning | Khan et al., IEEE TPAMI 2020 |
| 4 | Uformer | Transformer | Wang et al., CVPR 2022 |

### Known Gaps

To be completed during algorithm development phase.

---

## 5. Improvement Suggestions

### Priority Actions

1. **Generate challenge dataset** — Implement forward model and phantom generator
3. **Validate metrics** — Ensure PSNR/SSIM/consistency measures are appropriate
4. **Document physics** — Add to modality database with calibration parameters
5. **Define mismatch modes** — psf_calibration_error, mask_sensor_misalignment, depth_dependent_psf_variation etc.

---

## 6. Action Items

| Priority | Action | Status |
|----------|--------|--------|
| CRITICAL | Generate challenge dataset | TODO |
| HIGH | Validate assessment metrics | TODO |
| HIGH | Complete modality database entry | TODO |
| MEDIUM | Add missing references | TODO |
| MEDIUM | Identify algorithm gaps | TODO |
| LOW | Optimize gallery previews | TODO |

---

## Appendix: Key References

- Antipa et al., 'DiffuserCam: lensless single-exposure 3D imaging', Optica 5, 1-9 (2018)
- Asif et al., 'FlatCam: Thin, Lensless Cameras Using Coded Aperture', IEEE TCI 3, 384-397 (2017)

## Algorithm References

- Antipa et al., Optica 2018
- Khan et al., IEEE TPAMI 2020
- Monakhova et al., Opt. Express 2019
- Wang et al., CVPR 2022

*Automated 6-point review on 2026-03-03 — Lensless (Diffuser Camera) Imaging*
