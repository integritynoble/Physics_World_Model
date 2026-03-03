# Comprehensive 6-Point Check -- integral

**Modality:** Integral Imaging (Microlens Array Light Field)
**Category:** computational
**Variant override:** Yes (in `_VARIANT_OVERRIDES`)
**Check date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

Integral imaging captures the 4D light field L(x, y, u, v) through a
microlens array (MLA) placed in front of the image sensor. The forward model
is:

    I(s, t) = integral integral L(x, y, u(s), v(t)) * M(s, t) du dv

where `(s, t)` are sensor pixel coordinates, `(x, y)` are spatial positions,
`(u, v)` are angular coordinates determined by the microlens geometry, and
`M(s, t)` is the microlens aperture function. Each microlens captures a
sub-image encoding angular information, trading spatial resolution for
angular sampling.

The reconstruction tasks include: depth estimation from the 4D light field,
spatial super-resolution (recovering full resolution from sub-aperture images),
refocusing, and all-in-focus image synthesis.

Key physics: microlens pitch and f-number, diffraction at microlens apertures,
vignetting, chromatic aberration, and the spatial-angular resolution tradeoff.

**Verdict:** Physics correctly modeled. Integral imaging is a plenoptic/light
field modality using a microlens array.

## 2. Mismatch Parameters

Relevant mismatch/calibration parameters:
- Microlens array alignment (rotation, translation)
- Microlens focal length variation
- Vignetting across sub-aperture images
- Chromatic aberration from microlens
- Sensor pixel crosstalk
- Main lens aberrations

The benchmark models microlens alignment and vignetting as primary mismatch
parameters, which are the dominant calibration challenges for integral cameras.

**Verdict:** Appropriate. Key plenoptic camera calibration errors captured.

## 3. Reconstruction Methods

Current algorithms (from `_VARIANT_OVERRIDES["integral"]`):

| # | Algorithm | Type | Params | Source |
|---|-----------|------|--------|--------|
| 1 | Shift-and-Add | Classical | 0 | Ng et al., Stanford Tech Report 2005 |
| 2 | PnP-LF | PnP | 0 | PnP-ADMM with LF prior |
| 3 | LFAttNet | Deep Learning | 4.5M | Tsai et al., IEEE TIP 2020 |
| 4 | DistgSSR | Transformer | 12M | Wang et al., CVPR 2022 |

- **Shift-and-Add** is the fundamental light field refocusing algorithm that
  shifts sub-aperture images according to target depth and sums them.
  Universal baseline for plenoptic imaging. Correct.
- **PnP-LF** applies plug-and-play ADMM with light-field-aware priors
  (angular consistency, disparity regularization). Appropriate. Correct.
- **LFAttNet** is an attention-based network for light field depth estimation
  and angular super-resolution. Published in IEEE TIP 2020. Correct.
- **DistgSSR** (Disentangling Spatial-Angular Super-Resolution) is a
  transformer-based method for light field super-resolution that separately
  processes spatial and angular dimensions. CVPR 2022. Correct.

**Verdict:** PASS. All four algorithms are light-field-specific, replacing
the generic computational pool (Tikhonov, PnP-RED, DIP, SwinIR) that had
no awareness of the 4D light field structure.

## 4. Literature (2024-2025)

Recent relevant publications:
- Jin et al., "Light Field Super-Resolution via Implicit Neural
  Representations," CVPR 2024
- Wang et al., "Epipolar Transformer for Light Field Processing," IEEE TPAMI
  2024
- Liang et al., "DistgSSR-V2: Improved Light Field SR," IEEE TIP 2024
- Neural light field compression and rendering, SIGGRAPH 2024

The current set covers methods through CVPR 2022 (DistgSSR). 2024 adds
neural implicit representations and improved transformers but the core
paradigm (shift-and-add -> attention -> transformer SR) is well-represented.

**Verdict:** Acceptable. DistgSSR remains competitive.

## 5. Dataset & GCS Status

- Challenge HDF5 files on GCS: `integral_challenge_public.h5`,
  `integral_challenge_dev.h5`, `integral_challenge_hidden.h5` -- all present
- Gallery images on GCS: `img/benchmark_gallery/integral/scene_0{0-3}/`
  -- present
- Per-tier differentiation: different light field scenes per tier
- Dev tier: no `x_true` (ground truth stripped)
- Hidden tier: download blocked (403)
- Learning materials: 5 markdown files + README present

**Verdict:** PASS. All dataset and GCS assets verified.

## 6. Assessment

| Criterion | Status |
|-----------|--------|
| Physics accuracy | PASS |
| Algorithm correctness | PASS |
| Algorithm domain-specificity | PASS -- all 4 are light-field-specific |
| Literature coverage | PASS (through 2022; still competitive in 2024) |
| Dataset completeness | PASS |
| Overall | **PASS** |

No code changes required. The variant override provides light-field-specific
algorithms that exploit the 4D plenoptic structure, a critical improvement
over the generic computational pool.
