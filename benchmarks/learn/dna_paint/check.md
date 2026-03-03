# Comprehensive 6-Point Check — dna_paint

**URL:** https://pwm.platformai.org/benchmark/dna_paint
**Check Date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** DNA-PAINT Super-Resolution Microscopy

**Physical principle:** DNA-PAINT (Points Accumulation for Imaging in Nanoscale Topography) uses transient binding of fluorescently-labeled DNA strands to complementary target strands. Each binding event produces a diffraction-limited blinking spot. By localizing thousands of individual binding events across many frames, a super-resolved image is reconstructed with ~10 nm resolution.

**Forward model:** The observed image is a superposition of point-spread functions (PSFs) centered at each active emitter:
```
y(x, y, t) = Σ_i  I_i · PSF(x - x_i, y - y_i) + noise
```
where (x_i, y_i) are emitter positions and I_i are intensities.

**Inverse problem:** Single-molecule localization — determine emitter positions and intensities from noisy, diffraction-limited frames. Then render super-resolved image from localizations.

**Current forward model (PSF runner):** Gaussian PSF convolution. This is a simplification — real DNA-PAINT data has temporal blinking dynamics, varying photon counts, and background fluorescence. However, the PSF model captures the core spatial degradation.

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** M → D

**Mismatch sources in DNA-PAINT:**
- PSF shape variations (aberrations, defocus)
- Background fluorescence drift
- Emitter density variations
- Stage drift during acquisition
- Non-specific binding ("noise" localizations)

**Dataset format (GCS):**
- `x_true: (256, 256)` — high-resolution ground truth image
- `y: (256, 256)` — PSF-convolved measurement
- `H_ideal: (13, 13)` — PSF kernel

## 3. Reconstruction Methods & Leaderboard

**Algorithms (SMLM-specific, via variant override):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| ThunderSTORM | Classical | Ovesny et al., Bioinformatics 2014 | ✓ Gold-standard SMLM localization |
| FALCON | PnP | Min et al., Sci. Rep. 2014 | ✓ Fast localization with prior |
| Deep-STORM | Deep Learning | Nehme et al., Optica 2018 | ✓ CNN for dense emitter localization |
| DECODE | Deep Learning | Speiser et al., Nat. Methods 2021 | ✓ State-of-the-art probabilistic SMLM |

All 4 algorithms are domain-appropriate for single-molecule localization microscopy.

## 4. Literature & State of the Art (2024–2025)

Key recent developments:
1. **DECODE v2** (ongoing): Probabilistic deep learning for 3D SMLM with improved calibration
2. **ANNA-PALM** (Ouyang et al.): Deep learning for accelerated PALM/PAINT reconstruction
3. **FP-INR** (2024): Fourier-parameterized implicit neural representations for SMLM
4. **DNA-PAINT exchange** (Jungmann lab, 2024): Multiplexed DNA-PAINT with sequential imaging

Current algorithm selection covers the key approaches well.

## 5. Local Dataset & GCS Status

**GCS datasets (verified):**
- `dna_paint_challenge_public.h5` — 2,987 KB ✓
- `dna_paint_challenge_dev.h5` — 2,805 KB ✓
- `dna_paint_challenge_hidden.h5` — 2,877 KB ✓

**Gallery images:** Served from GCS, 24/24 load OK.

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS

**Previously fixed:** Algorithm override added in previous session (SMLM-specific algorithms instead of generic microscopy deconvolution).

**Remaining opportunities:**
- Consider SMLM-specific forward model (frame-by-frame blinking simulation) instead of single PSF convolution
- Could add ANNA-PALM or FP-INR as additional algorithms
- Public dataset could benefit from SMLM Challenge 2016 data (Sage et al., Nat. Methods 2019)

---
*Comprehensive 6-point check by deep-check pipeline v3*
