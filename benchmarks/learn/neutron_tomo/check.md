# Comprehensive 6-Point Check -- neutron_tomo

**URL:** https://pwm.platformai.org/benchmark/neutron_tomo
**Check Date:** 2026-03-03
**Status:** PASS (acceptable category routing, no code changes needed)

---

## 1. Physics & Forward Model

**Modality:** Neutron Radiography / Tomography

**Physical principle:** Neutron tomography uses a beam of thermal or cold neutrons that are attenuated as they pass through a sample. The attenuation follows Beer-Lambert's law:
```
I(s) = I_0 * exp(-integral mu(r) ds)
```
where mu(r) is the neutron attenuation coefficient (dependent on neutron cross-section, which is element-specific). Unlike X-ray CT where attenuation scales with atomic number, neutron attenuation is isotope-specific: hydrogen and lithium are strong neutron absorbers while lead is nearly transparent.

**Inverse problem:** Reconstruct 2D/3D attenuation coefficient maps from multiple angular projection measurements. The mathematical structure is identical to X-ray CT (Radon transform inversion) with different contrast mechanisms.

**Current physics engine:** Tomographic reconstruction. The projection-based forward model correctly captures the line-integral geometry of neutron transmission measurements.

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** R(theta) -> Pi(neutron) -> D(g, eta_1)

**Mismatch sources in neutron tomography:**
- Neutron beam divergence and energy spectrum spread
- Scattering artifacts (incoherent and coherent scattering)
- Gamma contamination in the neutron beam
- Detector efficiency variations (scintillator inhomogeneity)
- Beam hardening (polychromatic neutron spectrum)
- Sample activation and radioactive decay background

**Dataset format (GCS):**
- `x_true` -- ground truth attenuation coefficient map
- `y` -- sinogram/projection measurements
- `H_ideal` -- forward model parameters

**Tier structure:** Public (with x_true), Dev (no x_true), Hidden (blocked).

## 3. Reconstruction Methods & Leaderboard

**Algorithms (scientific_instrumentation category pool):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Deconv | Classical | Analytical baseline | Acceptable -- deconvolution/analytical inversion baseline |
| PnP-BM3D | PnP | Danielyan et al., 2012 | Acceptable -- PnP works for any tomographic inverse problem |
| ResNet-Calib | Deep Learning | ResNet for calibration, 2022 | Acceptable -- learned reconstruction from projections |
| CalibFormer | Transformer | Transformer calibration, 2024 | Acceptable -- transformer-based learned reconstruction |

The scientific_instrumentation pool provides generic inverse-problem algorithms. Neutron tomography shares the same mathematical structure as X-ray CT (Radon transform inversion), so CT-specific algorithms (FBP, SIRT, FBPConvNet) would also be appropriate. However, the current pool is not incorrect -- it tests reconstruction capability from projection data.

## 4. Literature & State of the Art (2024--2025)

1. **FBP/SIRT for neutron CT** (standard): Same algorithms as X-ray CT, applied to neutron projections. Widely used at neutron imaging beamlines (NIST, PSI, ILL, ORNL).
2. **CIL (Core Imaging Library)** (2024): Open-source framework for tomographic reconstruction supporting both X-ray and neutron CT.
3. **DL-based neutron CT** (2024): Transfer learning from X-ray CT networks to neutron data, addressing the lower flux/higher noise challenge.
4. **Energy-resolved neutron imaging** (2024--2025): Bragg-edge tomography for crystallographic texture mapping using time-of-flight neutron beams.
5. **Neutron phase-contrast imaging** (2024): Grating-based interferometry for neutron dark-field and phase-contrast tomography.
6. **IMAT beamline** (ISIS, 2024): Operational energy-resolved neutron imaging with iterative reconstruction pipelines.

## 5. Local Dataset & GCS Status

**GCS datasets verified:**
- `neutron_tomo_challenge_public.h5` -- present on GCS
- `neutron_tomo_challenge_dev.h5` -- present on GCS (x_true stripped)
- `neutron_tomo_challenge_hidden.h5` -- present on GCS (blocked from download)

**Gallery images:** 24/24 load OK from GCS.

**Learning materials:** Complete 5-module set present (README, physics fundamentals, forward model, reconstruction algorithms, PWM benchmark, hands-on tutorial).

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS -- no code changes needed.

**Routing:** Falls to `_CATEGORY_ALGORITHMS["scientific_instrumentation"]` (no carrier routing for neutron). The generic pool is acceptable: neutron tomography is one of many tomographic instruments in this category, and the generic algorithms cover the standard solver-class progression.

**Domain accuracy note:** Neutron tomography is mathematically identical to X-ray CT (Radon transform), so CT-specific algorithms (FBP, FBPConvNet, Learned Primal-Dual) would be more domain-specific. However, a blanket carrier route for `("scientific_instrumentation", "Neutron")` would also affect neutron_diffraction (which is NOT tomographic), so the current generic pool is the safer choice.

**No changes required.** The scientific_instrumentation pool is a defensible assignment.

---
*Comprehensive 6-point check by deep-check pipeline v3*
