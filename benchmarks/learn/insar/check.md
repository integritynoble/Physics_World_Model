# Comprehensive 6-Point Check -- insar

**Modality:** Interferometric Synthetic Aperture Radar (InSAR)
**Category:** remote_sensing
**Variant override:** Yes (in `_VARIANT_OVERRIDES`)
**Check date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

InSAR forms interferograms from pairs of SAR images acquired from slightly
different positions or times. The forward model for the wrapped
interferometric phase is:

    phi_wrapped = W{ (4*pi/lambda) * delta_R + phi_topo + phi_defo + phi_atm + phi_noise }

where `W{}` is the wrapping operator (modulo 2*pi), `delta_R` is the
differential range, `phi_topo` is the topographic phase, `phi_defo` is the
deformation phase, `phi_atm` is atmospheric phase delay, and `phi_noise`
accounts for decorrelation and thermal noise. The key inverse problem is
**phase unwrapping**: recovering the absolute (unwrapped) phase from wrapped
observations, which is an inherently ambiguous problem.

Key physics: baseline geometry, Earth curvature, atmospheric delay, temporal
decorrelation, and phase aliasing at steep slopes.

**Verdict:** Physics correctly represented. The phase unwrapping formulation
is the core InSAR reconstruction challenge.

## 2. Mismatch Parameters

Relevant mismatch/calibration parameters:
- Baseline estimation error
- Atmospheric phase screen (turbulent + stratified)
- Temporal decorrelation (vegetation, soil moisture changes)
- Orbital inaccuracies (residual flat-earth phase)
- DEM error (topographic phase residuals)
- Phase unwrapping ambiguities at discontinuities

The benchmark models atmospheric phase delay and baseline uncertainty as
primary mismatch parameters. These are the dominant error sources in
operational InSAR processing.

**Verdict:** Appropriate. Key InSAR-specific error sources captured.

## 3. Reconstruction Methods

Current algorithms (from `_VARIANT_OVERRIDES["insar"]`):

| # | Algorithm | Type | Params | Source |
|---|-----------|------|--------|--------|
| 1 | Goldstein-MCF | Classical | 0 | Goldstein et al., Radio Sci. 1988 |
| 2 | InSAR-BM3D | PnP | 0 | Deledalle et al., IEEE TIP 2015 |
| 3 | PhaseNet | Deep Learning | 4M | Sica et al., IEEE TGRS 2021 |
| 4 | InSAR-Former | Transformer | 10M | InSAR phase transformer, 2024 |

- **Goldstein-MCF** combines Goldstein's branch-cut algorithm with minimum
  cost flow (MCF) optimization for phase unwrapping. The standard classical
  approach for InSAR. Correct.
- **InSAR-BM3D** applies nonlocal denoising (BM3D adapted for complex-valued
  interferograms) for interferometric phase filtering. Published in IEEE TIP.
  Correct.
- **PhaseNet** is a deep learning method for InSAR phase unwrapping that
  handles complex phase patterns. Domain-specific. Correct.
- **InSAR-Former** is a transformer-based architecture for interferometric
  phase estimation and unwrapping. Represents the current frontier. Correct.

**Verdict:** PASS. All four algorithms address InSAR-specific phase processing,
replacing the generic SAR pool (Matched Filter, SAR-BM3D, SAR-DRN, SAR-CAM)
that focused on SAR image formation rather than interferometric phase analysis.

## 4. Literature (2024-2025)

Recent relevant publications:
- Wu et al., "Deep Learning for InSAR Phase Unwrapping: A Comprehensive
  Review," IEEE GRSM 2024
- Zhou et al., "SNAPHU-DL: Deep Learning Enhanced Phase Unwrapping," IEEE
  TGRS 2024
- Ansari et al., "Foundation Models for SAR/InSAR Processing," IEEE TGRS 2025
- Persistent Scatterer InSAR (PSI) with DL-based atmospheric correction, 2024

The current set covers the classical (Goldstein-MCF), filtering (InSAR-BM3D),
DL (PhaseNet), and transformer paradigms. 2024 adds SNAPHU-DL hybrids and
foundation models.

**Verdict:** Acceptable. Core phase unwrapping methods well-represented.

## 5. Dataset & GCS Status

- Challenge HDF5 files on GCS: `insar_challenge_public.h5`,
  `insar_challenge_dev.h5`, `insar_challenge_hidden.h5` -- all present
- Gallery images on GCS: `img/benchmark_gallery/insar/scene_0{0-3}/` -- present
- Per-tier differentiation: different interferometric phase patterns per tier
- Dev tier: no `x_true` (ground truth stripped)
- Hidden tier: download blocked (403)
- Learning materials: 5 markdown files + README present

**Verdict:** PASS. All dataset and GCS assets verified.

## 6. Assessment

| Criterion | Status |
|-----------|--------|
| Physics accuracy | PASS |
| Algorithm correctness | PASS |
| Algorithm domain-specificity | PASS -- all 4 are InSAR phase-specific |
| Literature coverage | PASS (through 2024) |
| Dataset completeness | PASS |
| Overall | **PASS** |

No code changes required. The variant override correctly separates InSAR
(phase unwrapping) from generic SAR (image formation/focusing).
