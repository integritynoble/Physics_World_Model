# Comprehensive 6-Point Check -- machine_vision

**Modality:** Machine Vision / Automated Optical Inspection (AOI)
**Category:** industrial_inspection
**Variant override:** Yes (in `_VARIANT_OVERRIDES`)
**Check date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

Machine vision uses visible-light cameras (sometimes with structured
illumination) for automated inspection of manufactured parts. The forward
model for defect detection/localization is:

    y(r) = I(r) * R(r) + n

where `I(r)` is the illumination pattern, `R(r)` is the scene reflectance
(including surface texture, defects, and nominal geometry), and `n` is sensor
noise. The reconstruction/analysis task is anomaly detection: identifying
regions where the observed image deviates from the expected (defect-free)
appearance.

Key physics: illumination geometry (diffuse, directional, ring light),
surface specularity, depth of field, camera resolution vs. defect size,
and the statistical definition of "normal" appearance for anomaly detection.

**Verdict:** Physics correctly modeled. Machine vision AOI is fundamentally
an optical anomaly detection problem.

## 2. Mismatch Parameters

Relevant mismatch/calibration parameters:
- Illumination variation (intensity, angle changes)
- Camera focus and depth-of-field limitations
- Part positioning/alignment variability
- Surface reflectance variation (normal vs. defective)
- Camera lens distortion and vignetting
- Environmental contamination (dust, oil on parts)

The benchmark models illumination variation and part positioning as primary
mismatch parameters, which dominate real-world AOI system performance.

**Verdict:** Appropriate. Key machine vision calibration challenges captured.

## 3. Reconstruction Methods

Current algorithms (from `_VARIANT_OVERRIDES["machine_vision"]`):

| # | Algorithm | Type | Params | Source |
|---|-----------|------|--------|--------|
| 1 | Template Match | Classical | 0 | Brunelli, Template Matching, 2009 |
| 2 | PnP-ADMM | PnP | 0 | Venkatakrishnan et al., 2013 |
| 3 | PatchCore | Deep Learning | 2M | Roth et al., CVPR 2022 |
| 4 | UniAD | Transformer | 15M | You et al., NeurIPS 2022 |

- **Template Match** is the classical approach for defect detection: compare
  observed images against reference templates and flag deviations. The
  standard industrial baseline. Correct.
- **PnP-ADMM** provides general-purpose image enhancement/denoising as a
  pre-processing step for inspection. Acceptable as a general method. Correct.
- **PatchCore** is a memory-bank-based anomaly detection method that achieves
  state-of-the-art on MVTec-AD. One of the most widely used methods for
  industrial defect detection. Correct.
- **UniAD** is a unified anomaly detection framework using transformer
  architecture. Published at NeurIPS 2022, designed specifically for
  industrial inspection. Correct.

**Verdict:** PASS. Three of four algorithms are specifically for industrial
anomaly detection (Template Match, PatchCore, UniAD). PnP-ADMM is general
but applicable. This replaces the thermal/NDT pool (TSR, PnP-ADMM, DefectNet,
LSTM-NDT) where TSR and LSTM-NDT were thermography-specific methods
inappropriate for optical inspection.

## 4. Literature (2024-2025)

Recent relevant publications:
- Liang et al., "AnomalyGPT: Detecting Industrial Anomalies using LVLMs,"
  AAAI 2024
- Liu et al., "EfficientAD: Accurate Visual Anomaly Detection at
  Millisecond-Level Latencies," WACV 2024
- Gu et al., "AnomalyDiffusion: Anomaly Detection with Diffusion Models,"
  CVPR 2024
- MVTec-AD benchmark update with new categories, 2024

The current set covers the template-to-transformer progression. 2024 adds
LLM-based anomaly detection (AnomalyGPT) and diffusion-based approaches.
PatchCore and UniAD remain strong baselines on MVTec-AD.

**Verdict:** Acceptable. PatchCore remains competitive; UniAD covers the
transformer paradigm.

## 5. Dataset & GCS Status

- Challenge HDF5 files on GCS: `machine_vision_challenge_public.h5`,
  `machine_vision_challenge_dev.h5`, `machine_vision_challenge_hidden.h5`
  -- all present
- Gallery images on GCS: `img/benchmark_gallery/machine_vision/scene_0{0-3}/`
  -- present
- Per-tier differentiation: different inspection scenes per tier
- Dev tier: no `x_true` (ground truth stripped)
- Hidden tier: download blocked (403)
- Learning materials: 5 markdown files + README present

**Verdict:** PASS. All dataset and GCS assets verified.

## 6. Assessment

| Criterion | Status |
|-----------|--------|
| Physics accuracy | PASS |
| Algorithm correctness | PASS |
| Algorithm domain-specificity | PASS -- 3/4 anomaly-detection-specific |
| Literature coverage | PASS (through 2022; PatchCore/UniAD remain SOTA baselines) |
| Dataset completeness | PASS |
| Overall | **PASS** |

No code changes required. The variant override correctly separates optical
machine vision (anomaly detection) from thermal/acoustic NDT methods in the
generic industrial_inspection pool.
