# Comprehensive Benchmark QA Check — Diffuse Optical Tomography

**URL:** https://pwm.platformai.org/benchmark/dot
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

**Display Name:** Diffuse Optical Tomography

**Physics Class:** diffuse_optical
**Forward Model:** diffusion_equation
**Noise Model:** poisson_gaussian

### Dataset Integrity Assessment: TODO

---

## 3. Public Dataset Source Assessment

### Canonical Datasets

- UCL DOT phantom datasets
- BU fNIRS-DOT brain imaging benchmarks

### Assessment: TODO

To be completed upon dataset publication.

---

## 4. Algorithm Coverage Assessment

### Currently Tested: 4 algorithms

| # | Algorithm | Type | Source |
|---|-----------|------|--------|
| 1 | Tikhonov-Born | Classical | Arridge, Inverse Probl. 1999 |
| 2 | L-BFGS-TV | Classical | Schweiger & Arridge, PMB 2005 |
| 3 | PnP-Diffusion | PnP | Yoo et al., IEEE TMI 2020 |
| 4 | DeepDOT | Deep Learning | Yoo et al., IEEE TMI 2020 |

### Known Gaps

To be completed during algorithm development phase.

---

## 5. Improvement Suggestions

### Priority Actions

1. **Generate challenge dataset** — Implement forward model and phantom generator
3. **Validate metrics** — Ensure PSNR/SSIM/consistency measures are appropriate
4. **Document physics** — Add to modality database with calibration parameters
5. **Define mismatch modes** — coupling_variation, position_uncertainty, boundary_model_error etc.

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

- Arridge, 'Optical tomography in medical imaging', Inverse Problems 15, R41-R93 (1999)
- Boas et al., 'Imaging the body with diffuse optical tomography', IEEE Signal Processing Magazine 18, 57-75 (2001)

## Algorithm References

- Arridge, Inverse Probl. 1999
- Schweiger & Arridge, PMB 2005
- Yoo et al., IEEE TMI 2020

*Automated 6-point review on 2026-03-03 — Diffuse Optical Tomography*
