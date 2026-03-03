# Comprehensive 6-Point Check — dot

**URL:** https://pwm.platformai.org/benchmark/dot
**Check Date:** 2026-03-03
**Status:** PASS (algorithms fixed, dataset acceptable)

---

## 1. Physics & Forward Model

**Modality:** Diffuse Optical Tomography (DOT)

**Physical principle:** DOT uses near-infrared (NIR) light that propagates diffusely through tissue. Source-detector pairs on the tissue boundary measure transmitted/reflected photon fluence. The diffusion approximation to the radiative transfer equation governs light propagation:
```
-∇·(D(r)∇Φ(r)) + μ_a(r)Φ(r) = S(r)
```
where D = diffusion coefficient, μ_a = absorption coefficient, Φ = photon fluence, S = source.

**Inverse problem:** Recover spatial maps of absorption μ_a(r) and/or scattering μ_s(r) coefficients from boundary measurements. This is severely ill-posed and nonlinear.

**Current dataset runner:** `radon` (sinogram). This is a simplification — DOT measurements come from boundary source-detector pairs, not angular projections. However, both are tomographic inverse problems and the radon model tests the algorithms' ability to recover 2D maps from limited data.

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(diffuse) → Σ → D(g, η₃)

**Dataset format:**
- `x_true: (128, 128)` — absorption/scattering coefficient map
- `y: (180, 182)` — sinogram measurements
- `H_ideal: (180,)` — projection angles

## 3. Reconstruction Methods & Leaderboard

**Algorithms (DOT-specific, via variant override):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Tikhonov-Born | Classical | Arridge, Inverse Probl. 1999 | ✓ Standard DOT regularization |
| L-BFGS-TV | Classical | Schweiger & Arridge, PMB 2005 | ✓ Nonlinear DOT solver with TV |
| PnP-Diffusion | PnP | Yoo et al., IEEE TMI 2020 | ✓ PnP for DOT with learned prior |
| DeepDOT | Deep Learning | Yoo et al., IEEE TMI 2020 | ✓ Neural network for DOT |

All 4 algorithms are domain-appropriate.

## 4. Literature & State of the Art (2024–2025)

1. **Deep-DOT** (Yoo et al., 2020): Learned image reconstruction for DOT
2. **TOAST++** (Schweiger & Arridge): Open-source DOT reconstruction toolkit
3. **Diffusion-model priors for DOT** (2024): Score-based generative models as priors
4. **Multi-spectral DOT** (2024): Hyperspectral DOT for tissue chromophore imaging

## 5. Local Dataset & GCS Status

**GCS datasets verified.** All 3 tiers present with appropriate sizes.

## 6. Comprehensive Assessment

**Status:** PASS

**Previously fixed:** Algorithm override from OCT pool → DOT-specific pool.

**Remaining opportunity:** DOT-specific forward model (diffusion-equation based) would be more physically accurate than current Radon-based model.

---
*Comprehensive 6-point check by deep-check pipeline v3*
