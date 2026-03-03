# Comprehensive 6-Point Check -- palm_storm

**URL:** https://pwm.platformai.org/benchmark/palm_storm
**Check Date:** 2026-03-03
**Status:** PASS (correct SMLM override, no code changes needed)

---

## 1. Physics & Forward Model

**Modality:** PALM/STORM Single-Molecule Localization Microscopy

**Physical principle:** PALM (Photo-Activated Localization Microscopy) and STORM (STochastic Optical Reconstruction Microscopy) achieve super-resolution by exploiting the stochastic blinking of fluorophores. In each frame, only a sparse subset of fluorophores is active. Each active emitter produces a diffraction-limited spot described by the PSF:
```
y(x, y, t) = sum_i  I_i * PSF(x - x_i, y - y_i) + b(x, y) + noise
```
where (x_i, y_i) are emitter positions, I_i are photon counts, and b is the background.

By localizing individual emitters with sub-pixel precision across thousands of frames and rendering the accumulated localizations, a super-resolved image is constructed with ~20 nm resolution (10x beyond the diffraction limit).

**Inverse problem:** Single-molecule localization -- determine the number, positions, and intensities of active emitters in each frame from noisy, diffraction-limited data. This is fundamentally different from deconvolution: it is a sparse point-source estimation problem.

**Current physics engine:** PSF convolution model with Gaussian PSF. This captures the core spatial degradation, though real SMLM data has additional complexity (blinking kinetics, varying photon counts, axial defocus).

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** C(PSF) -> D(g, eta_3)

**Mismatch sources in PALM/STORM:**
- PSF shape variations (aberrations, refractive index mismatch)
- Emitter density fluctuations (overlapping PSFs at high density)
- Background fluorescence and autofluorescence
- Stage drift during long acquisitions (hours)
- Photo-bleaching and incomplete activation
- Axial position uncertainty (2D vs 3D)

**Dataset format (GCS):**
- `x_true` -- high-resolution ground truth image
- `y` -- PSF-convolved/blinking measurement
- `H_ideal` -- PSF kernel

**Tier structure:** Public (with x_true), Dev (no x_true), Hidden (blocked).

## 3. Reconstruction Methods & Leaderboard

**Algorithms (SMLM-specific, via variant override):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| ThunderSTORM | Classical | Ovesny et al., Bioinformatics 2014 | CORRECT -- gold-standard SMLM localization plugin |
| FALCON | PnP | Min et al., Sci. Rep. 2014 | CORRECT -- fast localization with deconvolution prior |
| Deep-STORM | Deep Learning | Nehme et al., Optica 2018 | CORRECT -- CNN for dense emitter localization |
| DECODE | Deep Learning | Speiser et al., Nat. Methods 2021 | CORRECT -- state-of-the-art probabilistic SMLM |

All 4 algorithms are domain-appropriate for single-molecule localization microscopy. This is a significant improvement over the previous assignment (generic microscopy deconvolution: Richardson-Lucy, PnP-FISTA, CARE, Restormer).

## 4. Literature & State of the Art (2024--2025)

1. **DECODE** (Speiser et al., Nat. Methods 2021): Probabilistic deep learning for 3D SMLM -- currently the state of the art. Already included in the pool.
2. **ANNA-PALM** (Ouyang et al., Nat. Biotechnol. 2018): Deep learning for accelerated PALM with sparse frames.
3. **Deep-STORM3D** (Nehme et al., 2020): Extension of Deep-STORM to 3D localization.
4. **SMLM Challenge 2016** (Sage et al., Nat. Methods 2019): Community benchmark establishing evaluation standards.
5. **FP-INR** (2024): Fourier-parameterized implicit neural representations for SMLM reconstruction.
6. **MINFLUX + PALM hybrid** (2024--2025): Combining SMLM with MINFLUX for angstrom-level resolution.

## 5. Local Dataset & GCS Status

**GCS datasets verified:**
- `palm_storm_challenge_public.h5` -- present on GCS
- `palm_storm_challenge_dev.h5` -- present on GCS (x_true stripped)
- `palm_storm_challenge_hidden.h5` -- present on GCS (blocked from download)

**Gallery images:** 24/24 load OK from GCS.

**Learning materials:** Complete 5-module set present (README, physics fundamentals, forward model, reconstruction algorithms, PWM benchmark, hands-on tutorial).

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS -- no code changes needed.

**Routing:** SMLM-specific override applied. The variant `palm_storm` is correctly routed to localization-specific algorithms (ThunderSTORM, FALCON, Deep-STORM, DECODE) instead of the generic microscopy deconvolution pool.

**Previously fixed:** palm_storm was originally getting generic microscopy algorithms (Richardson-Lucy, PnP-FISTA, CARE, Restormer) which are deconvolution/denoising methods inappropriate for single-molecule localization. The override provides the correct SMLM-specific algorithms.

**No further changes required.** The algorithm assignment is correct and domain-appropriate.

---
*Comprehensive 6-point check by deep-check pipeline v3*
