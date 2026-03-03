# Comprehensive 6-Point Check -- oct

**URL:** https://pwm.platformai.org/benchmark/oct
**Check Date:** 2026-03-03
**Status:** PASS (correct carrier routing, no code changes needed)

---

## 1. Physics & Forward Model

**Modality:** Optical Coherence Tomography (OCT)

**Physical principle:** OCT uses low-coherence interferometry to produce depth-resolved cross-sectional images of biological tissue (primarily retina, skin, and vasculature). A broadband light source is split into sample and reference arms; interference between backscattered sample light and the reference beam encodes depth information. In spectral-domain OCT (SD-OCT), the spectral interferogram is Fourier-transformed to recover the depth profile:
```
I(k) = |E_r|^2 + |E_s|^2 + 2*Re{E_r * E_s* * exp(j*2*k*z)}
```
where k = wavenumber, z = depth, E_r = reference field, E_s = sample field.

**Inverse problem:** Recover the depth-resolved reflectivity profile (A-scan) from the spectral interferogram, then assemble B-scans (cross-sections) and C-scans (volumes). Key challenges include speckle noise reduction, dispersion compensation, and motion artifact correction.

**Current physics engine:** Interferometric forward model with PSF convolution. The carrier routing `(medical, Photon) -> clinical_optics` correctly sends OCT to the optics-specific algorithm pool.

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(low-coherence) -> Sigma(interference) -> D(g, eta_1)

**Mismatch sources in OCT:**
- Speckle noise (inherent in coherent imaging)
- Dispersion mismatch between sample and reference arms
- Motion artifacts (patient eye movement)
- Roll-off (signal attenuation with depth in SD-OCT)
- Complex conjugate ambiguity
- Polarization mode dispersion

**Dataset format (GCS):**
- `x_true` -- ground truth reflectivity map
- `y` -- degraded/noisy OCT measurement
- `H_ideal` -- forward model parameters

**Tier structure:** Public (with x_true), Dev (no x_true), Hidden (blocked).

## 3. Reconstruction Methods & Leaderboard

**Algorithms (clinical_optics pool via carrier routing: medical + Photon -> clinical_optics):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| FFT-OCT | Classical | Standard Fourier-domain OCT processing | CORRECT -- the standard OCT reconstruction baseline |
| BM4D | PnP | Maggioni et al., IEEE TIP 2013 | CORRECT -- 3D block-matching for volumetric speckle reduction |
| Speckle-DenoiseNet | Deep Learning | Devalla et al., BOE 2019 | CORRECT -- CNN for OCT speckle denoising |
| OCTA-Net | Deep Learning | Ma et al., BOE 2020 | CORRECT -- deep learning for OCT angiography |

All 4 algorithms are domain-appropriate for OCT reconstruction and denoising.

## 4. Literature & State of the Art (2024--2025)

1. **FFT-OCT** (standard): Fourier-domain processing is the universal baseline for all spectral-domain and swept-source OCT systems.
2. **DnCNN/BM3D for OCT** (ongoing): Classical denoising applied to OCT -- widely benchmarked.
3. **Self-supervised OCT denoising** (2024): Noise2Void and Noise2Self variants adapted for OCT speckle.
4. **Retinal layer segmentation + reconstruction** (2024): Joint segmentation-denoising networks for retinal OCT.
5. **OCT angiography deep learning** (2024--2025): Transformer-based OCTA networks for vascular imaging.
6. **Computational OCT** (2024): Hardware-software co-design for compressed OCT acquisition.

## 5. Local Dataset & GCS Status

**GCS datasets verified:**
- `oct_challenge_public.h5` -- present on GCS
- `oct_challenge_dev.h5` -- present on GCS (x_true stripped)
- `oct_challenge_hidden.h5` -- present on GCS (blocked from download)

**Gallery images:** 24/24 load OK from GCS.

**Learning materials:** Complete 5-module set present (README, physics fundamentals, forward model, reconstruction algorithms, PWM benchmark, hands-on tutorial).

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS -- no code changes needed.

**Routing:** `(medical, Photon)` -> `clinical_optics` pool. This was previously fixed from the generic medical/CT pool (FBP, FBPConvNet) to the correct clinical optics pool. The current algorithms (FFT-OCT, BM4D, Speckle-DenoiseNet, OCTA-Net) are all domain-appropriate.

**Previously fixed:** OCT was incorrectly getting CT algorithms via the generic medical category. The carrier-based routing `(medical, Photon) -> clinical_optics` resolved this.

**No further changes required.** The algorithm assignment is correct.

---
*Comprehensive 6-point check by deep-check pipeline v3*
