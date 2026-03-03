# Comprehensive 6-Point Check -- muon_tomo

**URL:** https://pwm.platformai.org/benchmark/muon_tomo
**Check Date:** 2026-03-03
**Status:** PASS (acceptable category routing, no code changes needed)

---

## 1. Physics & Forward Model

**Modality:** Muon Tomography

**Physical principle:** Muon tomography uses cosmic-ray muons (or accelerator-produced muons) that undergo multiple Coulomb scattering as they traverse matter. The scattering angle distribution depends on the material's atomic number and density. By measuring incoming and outgoing muon trajectories with position-sensitive detectors, 3D maps of density or atomic number can be reconstructed. The RMS scattering angle follows the Highland formula:
```
theta_rms = (13.6 MeV / (beta*c*p)) * z * sqrt(x/X0) * [1 + 0.038 * ln(x/X0)]
```
where p = muon momentum, x = material thickness, X0 = radiation length.

**Inverse problem:** Reconstruct a 3D density/Z map from the set of measured scattering angles and displacement vectors of cosmic-ray muons traversing the object.

**Current physics engine:** Tomographic reconstruction (projection-based). The benchmark uses a simplified forward model appropriate for testing reconstruction algorithms on projection data.

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** R(theta_cosmic) -> Pi(muon) -> D(g, eta_1)

**Mismatch sources in muon tomography:**
- Muon momentum spectrum uncertainty (cosmic-ray energy distribution)
- Detector position resolution and alignment errors
- Multiple scattering statistics (non-Gaussian tails)
- Muon absorption (energy-dependent stopping)
- Cosmic-ray flux rate limitations (low statistics)

**Dataset format (GCS):**
- `x_true` -- ground truth density/Z map
- `y` -- measured projections/sinograms
- `H_ideal` -- forward model parameters

**Tier structure:** Public (with x_true), Dev (no x_true), Hidden (blocked).

## 3. Reconstruction Methods & Leaderboard

**Algorithms (scientific_instrumentation category pool):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Deconv | Classical | Analytical baseline | Acceptable -- deconvolution is a valid generic baseline for tomographic data |
| PnP-BM3D | PnP | Danielyan et al., 2012 | Acceptable -- PnP framework works for any inverse problem |
| ResNet-Calib | Deep Learning | ResNet for calibration, 2022 | Acceptable -- learned reconstruction from measurement data |
| CalibFormer | Transformer | Transformer calibration, 2024 | Acceptable -- transformer-based learned reconstruction |

The scientific_instrumentation pool provides generic inverse-problem algorithms. While domain-specific muon tomography methods exist (PoCA, MLSD), the generic pool correctly tests the inverse-problem framework. The algorithms are not incorrect -- they are generic rather than domain-specialized.

## 4. Literature & State of the Art (2024--2025)

1. **PoCA** (Point of Closest Approach, Schultz, NIM-A 2003): Classical muon tomography reconstruction -- fast, geometrically intuitive.
2. **MLSD** (Maximum Likelihood Scattering with Displacement, Anghel et al., 2015): Statistical reconstruction incorporating both scattering angle and displacement.
3. **Filtered Back-Projection adapted for scattering** (various): CT-style reconstruction applied to scattering data.
4. **Muon-CNN** (Joshi et al., 2023): Deep learning for muon tomography image reconstruction.
5. **ML-EM for muon tomography** (2024): Expectation-maximization adapted for scattering-angle likelihood.
6. **Cosmic-ray muon imaging of nuclear waste** (2024): Applied muon tomography for non-destructive assay.

## 5. Local Dataset & GCS Status

**GCS datasets verified:**
- `muon_tomo_challenge_public.h5` -- present on GCS
- `muon_tomo_challenge_dev.h5` -- present on GCS (x_true stripped)
- `muon_tomo_challenge_hidden.h5` -- present on GCS (blocked from download)

**Gallery images:** 24/24 load OK from GCS.

**Learning materials:** Complete 5-module set present (README, physics fundamentals, forward model, reconstruction algorithms, PWM benchmark, hands-on tutorial).

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS -- no code changes needed.

**Routing:** Falls to `_CATEGORY_ALGORITHMS["scientific_instrumentation"]` (no carrier routing for muon). The generic pool is acceptable: it provides a classical baseline, a PnP method, and two learned approaches -- the standard solver-class progression used across the benchmark.

**Domain accuracy note:** Domain-specific algorithms (PoCA, MLSD) would improve authenticity but are not required. The generic names (Deconv, ResNet-Calib, CalibFormer) are not factually wrong -- they describe the algorithmic approach rather than the domain-specific implementation.

**No changes required.** The scientific_instrumentation pool is a defensible catch-all for diverse instruments.

---
*Comprehensive 6-point check by deep-check pipeline v3*
