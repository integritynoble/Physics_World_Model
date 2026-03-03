# Comprehensive Benchmark QA Check — Integral Photography

**URL:** https://pwm.platformai.org/benchmark/integral
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

**Display Name:** Integral Photography

**Physics Class:** light_field
**Forward Model:** elemental_image_formation
**Noise Model:** gaussian

### Dataset Integrity Assessment: TODO

---

## 3. Public Dataset Source Assessment

### Canonical Datasets

- ETRI integral imaging test set
- Middlebury multi-view stereo (adapted)

### Assessment: TODO

To be completed upon dataset publication.

---

## 4. Algorithm Coverage Assessment

### Currently Tested: 4 algorithms

| # | Algorithm | Type | Source |
|---|-----------|------|--------|
| 1 | Shift-and-Add | Classical | Ng et al., Stanford Tech Report 2005 |
| 2 | PnP-LF | PnP | PnP-ADMM with LF prior |
| 3 | LFAttNet | Deep Learning | Tsai et al., IEEE TIP 2020 |
| 4 | DistgSSR | Transformer | Wang et al., CVPR 2022 |

### Known Gaps

To be completed during algorithm development phase.

---

## 5. Improvement Suggestions

### Priority Actions

1. **Generate challenge dataset** — Implement forward model and phantom generator
3. **Validate metrics** — Ensure PSNR/SSIM/consistency measures are appropriate
4. **Document physics** — Add to modality database with calibration parameters
5. **Define mismatch modes** — microlens_alignment, crosstalk, fill_factor_loss etc.

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

- Lippmann, C. R. Acad. Sci. Paris 146, 446 (1908)
- Park et al., 'Recent progress in 3D imaging systems', J. Opt. Soc. Am. A 26, 2538 (2009)

## Algorithm References

- Ng et al., Stanford Tech Report 2005
- PnP-ADMM with LF prior
- Tsai et al., IEEE TIP 2020
- Wang et al., CVPR 2022

*Automated 6-point review on 2026-03-03 — Integral Photography*
