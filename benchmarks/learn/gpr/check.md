# Comprehensive 6-Point Check -- gpr

**Modality:** Ground-Penetrating Radar (GPR)
**Category:** remote_sensing
**Variant override:** Yes (in `_VARIANT_OVERRIDES`)
**Check date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

GPR transmits short electromagnetic pulses (typically 100 MHz -- 2 GHz) into
the subsurface and records reflected signals from dielectric discontinuities.
The forward model is:

    y(t, x_r) = integral G(x_r, x', t) * sigma(x') dx' + n

where `G` is the Green's function for EM wave propagation in a lossy medium,
`sigma(x')` is the subsurface reflectivity/permittivity distribution, and
`y(t, x_r)` is the recorded B-scan (time vs. receiver position). The inverse
problem recovers subsurface structure from time-domain radar traces using
migration or tomographic methods.

Key physics: near-field propagation (unlike far-field SAR), frequency-dependent
attenuation in lossy soils, hyperbolic diffraction patterns from point
scatterers, and velocity variations due to soil moisture content.

**Verdict:** Physics correctly modeled. GPR is appropriately distinguished
from SAR despite both using RF carriers.

## 2. Mismatch Parameters

Relevant mismatch/calibration parameters:
- Soil dielectric constant uncertainty (affects wave velocity)
- Antenna coupling and ground surface reflection
- Frequency-dependent attenuation (soil conductivity)
- Antenna height variations during scanning
- Clutter from surface irregularities and antenna ringing

The benchmark models velocity uncertainty and attenuation as primary
mismatch parameters, which are dominant for GPR imaging quality.

**Verdict:** Appropriate. Key subsurface propagation uncertainties captured.

## 3. Reconstruction Methods

Current algorithms (from `_VARIANT_OVERRIDES["gpr"]`):

| # | Algorithm | Type | Params | Source |
|---|-----------|------|--------|--------|
| 1 | Kirchhoff Migration | Classical | 0 | Stolt, Geophysics 1978 |
| 2 | RTM | Classical | 0 | Baysal et al., Geophysics 1983 |
| 3 | GPR-RCNN | Deep Learning | 6M | Pham & Lefevre, JECE 2020 |
| 4 | HyperDet | Deep Learning | 10M | GPR detection transformer, 2023 |

- **Kirchhoff Migration** is the standard GPR imaging method that collapses
  diffraction hyperbolae to their apices. Universally used. Correct.
- **RTM (Reverse-Time Migration)** is a wave-equation-based migration that
  handles complex velocity structures. Standard in seismic/GPR. Correct.
- **GPR-RCNN** is a region-based CNN for detecting and localizing subsurface
  objects (rebar, pipes, voids) from B-scans. Domain-specific. Correct.
- **HyperDet** is a transformer/attention-based detector for hyperbola
  detection and characterization in GPR radargrams. Domain-specific. Correct.

**Verdict:** PASS. All four algorithms are GPR-specific, replacing the
previously-assigned SAR methods (Matched Filter, SAR-BM3D, SAR-DRN, SAR-CAM)
that were inappropriate for subsurface radar.

## 4. Literature (2024-2025)

Recent relevant publications:
- Xu et al., "GPR-YOLO: Real-Time Underground Object Detection," IEEE TGRS
  2024 -- YOLO-based detection from GPR B-scans
- Liu et al., "Deep Learning for GPR Data Interpretation: A Review," Remote
  Sensing 2024 -- comprehensive survey
- Feng et al., "Transformer-Based GPR Signal Classification," IEEE TGRS 2024
- Yang et al., "Physics-Informed GPR Inversion with Neural Networks,"
  Geophysics 2024

The current set (Kirchhoff, RTM, GPR-RCNN, HyperDet) covers the classical
and deep learning landscape well. 2024 trends add YOLO-style detectors and
physics-informed neural networks.

**Verdict:** Acceptable. Core methods well-represented.

## 5. Dataset & GCS Status

- Challenge HDF5 files on GCS: `gpr_challenge_public.h5`, `gpr_challenge_dev.h5`,
  `gpr_challenge_hidden.h5` -- all present in `challenge-data/v1.0/`
- Gallery images on GCS: `img/benchmark_gallery/gpr/scene_0{0-3}/` -- present
- Per-tier differentiation: different subsurface phantom models per tier
- Dev tier: no `x_true` (ground truth stripped)
- Hidden tier: download blocked (403)
- Learning materials: 5 markdown files + README present

**Verdict:** PASS. All dataset and GCS assets verified.

## 6. Assessment

| Criterion | Status |
|-----------|--------|
| Physics accuracy | PASS |
| Algorithm correctness | PASS |
| Algorithm domain-specificity | PASS -- all 4 are GPR-specific |
| Literature coverage | PASS (through 2023; 2024 adds YOLO variants) |
| Dataset completeness | PASS |
| Overall | **PASS** |

No code changes required. The variant override correctly replaces SAR-family
algorithms with GPR-specific migration and detection methods.
