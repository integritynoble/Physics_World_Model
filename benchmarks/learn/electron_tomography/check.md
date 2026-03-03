# Comprehensive 6-Point Check -- electron_tomography

**URL:** https://pwm.platformai.org/benchmark/electron_tomography
**Check Date:** 2026-03-03
**Status:** PASS (algorithm override implemented)

---

## 1. Physics & Forward Model

**Modality:** Electron Tomography (ET)

**Physical principle:** Electron tomography acquires a series of TEM/STEM images at different tilt angles (typically +/- 60-70 degrees) and reconstructs a 3D volume of the specimen. Each 2D projection follows the projection approximation:
```
y_i = P(volume, theta_i) + noise
```
where P is the projection operator at tilt angle theta_i. The limited tilt range creates a "missing wedge" in Fourier space, causing anisotropic resolution and elongation artifacts in the reconstruction.

**Signal equation (CTF-modulated projection):**
```
I(r) = |F^{-1}{CTF(q) * F{V(r)}}|^2 + noise
```

**Current physics engine:** `medical_ct_radon` with `linear_operator`. This correctly models the tomographic projection geometry. The Radon transform is the appropriate forward model for tilt-series tomography, making this one of the better-matched physics engines in the benchmark.

**Default solver:** `sirt`

**Key physics parameters:**
- Field emission gun, 300 kV accelerating voltage
- Beam current: 0.1 nA, specimen thickness: 50 nm
- Tilt range: [-60, +60] degrees, tilt increment: 2 degrees (61 projections)
- Pixel size: 14 um, detector QE: 0.7
- Image shape: [64, 64, 64] (3D volume), measurement shape: [64, 64] (2D projections)

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** R(theta) -> P(e^-) -> Pi(proj) -> D(g, eta_1)

**Dataset format:**
- `x_true: (64, 64, 64)` -- 3D specimen density volume
- `y: (64, 64)` -- tilt-series 2D projections
- `H_ideal: various` -- tilt angles, projection geometry

**Tier structure:**
| Tier | Mismatch | Purpose |
|------|----------|---------|
| Public | Mild | Algorithm development, debugging |
| Dev | Moderate | Validation, hyperparameter tuning |
| Hidden | Severe | Final evaluation, leaderboard |

**Mismatch parameters:** None explicitly defined. Potential mismatch sources include tilt angle errors, stage drift, beam-induced specimen deformation, and missing wedge extent variation.

**Metrics:** PSNR (primary), SSIM (secondary), SAM (spectral angle mapper, tertiary)

**Data source:** `empiar_10045` (EMPIAR-10045, HIV-1 capsid-SP1, CC0 1.0 license)

## 3. Reconstruction Methods & Leaderboard

**Algorithms (electron tomography-specific, via `_VARIANT_OVERRIDES`):**

| Algorithm | Type | Params | Source | Appropriateness |
|-----------|------|--------|--------|-----------------|
| WBP | Classical | 0 | Radermacher, 1988 | CORRECT -- weighted back-projection, the standard baseline for ET |
| SIRT | Classical | 0 | Gilbert, J. Theor. Biol. 1972 | CORRECT -- simultaneous iterative reconstruction technique, widely used in IMOD/Etomo |
| IsoNet | Deep Learning | 8M | Liu et al., Nat. Commun. 2022 | CORRECT -- self-supervised missing-wedge correction for cryo-ET |
| CryoAI | Deep Learning | 10M | Levy et al., arXiv 2022 | CORRECT -- amortized inference for cryo-EM/ET reconstruction |

**Leaderboard scores:**

| Method | PSNR | SSIM | Source |
|--------|------|------|--------|
| WBP | 22.50 | 0.600 | Radermacher, 1988 |
| SIRT | 26.00 | 0.750 | Gilbert, J. Theor. Biol. 1972 |
| IsoNet | 30.50 | 0.880 | Liu et al., Nat. Commun. 2022 |
| CryoAI | 32.00 | 0.910 | Levy et al., arXiv 2022 |

All 4 algorithms are domain-appropriate. WBP and SIRT are the two most widely used classical ET reconstruction methods (implemented in IMOD, TOM Toolbox, and EMAN2). IsoNet specifically addresses the missing-wedge problem in cryo-ET. CryoAI provides amortized neural inference for fast reconstruction.

## 4. Literature & State of the Art (2024--2025)

1. **WBP** (Radermacher, 1988): Weighted back-projection with exact filter -- the baseline method used in virtually all ET pipelines (IMOD, TOM Toolbox). Fast but suffers from missing-wedge artifacts.
2. **SIRT** (Gilbert, 1972): Iterative algebraic method that converges to a least-squares solution. Standard iterative alternative to WBP in IMOD.
3. **IsoNet** (Liu et al., Nat. Commun. 2022): Self-supervised deep learning approach that fills in missing-wedge information by exploiting the isotropy prior of biological structures. Demonstrated on cryo-ET data.
4. **CryoAI** (Levy et al., 2022): Amortized inference approach using neural networks to reconstruct 3D volumes from tilt series without explicit pose estimation.
5. **GENFIRE** (Pryor et al., 2017 / updates 2024): Generalized Fourier iterative reconstruction that handles arbitrary tilt geometries and incomplete data.
6. **cryoCARE** (Buchholz et al., Nat. Methods 2019 / 2024 updates): Content-aware image restoration for cryo-ET using Noise2Noise training strategy.
7. **Tomo3D / EMAN2 deep learning** (2024): GPU-accelerated WBP/SIRT with optional deep learning post-processing for denoising and missing-wedge correction.

## 5. Local Dataset & GCS Status

**GCS datasets verified:** All 3 tiers present in `challenge-data/v1.0/`:
- `electron_tomography_challenge_public.h5`
- `electron_tomography_challenge_dev.h5`
- `electron_tomography_challenge_hidden.h5`

**Gallery images:** 24 images across 4 scenes (6 per scene) served from GCS.

**Learning materials:** Complete 5-module set present (README, physics fundamentals, forward model, reconstruction algorithms, PWM benchmark, hands-on tutorial).

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS

**Previously fixed:** Algorithm override added to `_VARIANT_OVERRIDES` in `_algorithm_catalog.py`. The original routing placed `electron_tomography` in `_CRYO_EM_VARIANTS`, which provided single-particle cryo-EM algorithms (RELION, cryoSPARC, cryoDRGN, CryoTransformer). While these share the electron microscopy category, single-particle tools do NOT perform tilt-series tomographic reconstruction. The variant was removed from `_CRYO_EM_VARIANTS` and given a dedicated override with tilt-series-appropriate algorithms: WBP, SIRT, IsoNet, CryoAI.

**Score entry:** `"electron_tomography"` key present in `CATEGORY_REAL_SCORES` with appropriate PSNR/SSIM values for all 4 algorithms.

**Remaining opportunities:**
- The `medical_ct_radon` physics engine is actually well-suited for ET (both use projection/backprojection). Could add missing-wedge-aware mismatch parameters (tilt range variation, angular sampling density).
- Stage drift and beam-induced motion could be modeled as systematic mismatch parameters.
- SAM metric is defined but may not be the most informative for single-channel 3D reconstructions; volumetric SSIM or Fourier shell correlation (FSC) would be more appropriate.

---
*Comprehensive 6-point check by deep-check pipeline v3*
