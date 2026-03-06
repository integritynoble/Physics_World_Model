# Comprehensive 6-Point Check — Lucky Imaging

**URL:** https://pwm.platformai.org/benchmark/lucky_imaging
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Lucky Imaging (Atmospheric Speckle Imaging)

**Physical principle:** Turbulence in Earth's atmosphere causes refractive index fluctuations that warp and blur stellar wavefronts, producing a time-varying speckle pattern at the telescope focal plane. On short exposures (≲ 20 ms), the atmosphere is momentarily "frozen," and a small fraction of frames will have a near-diffraction-limited core ("lucky" frames). Lucky imaging selects and aligns these best frames to reconstruct a high-resolution image of the target.

**Forward model:**
```
y_t = PSF_t ⊗ x + η_t

where:
  x        — true high-resolution sky brightness distribution
  PSF_t    — instantaneous turbulent point spread function at time t
             modeled via Kolmogorov phase screen with Fried parameter r₀
  ⊗        — 2D convolution (shift-variant in general)
  η_t      — Poisson shot noise + detector read noise

For a burst of N frames: Y = {y_t}_{t=1}^{N}
```

**Inverse problem:** Recover x from a burst of N short-exposure frames Y, exploiting temporal variation to separate the object from the turbulence-induced blur.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(turbulence screen) → F(stellar scene) → D(EMCCD/sCMOS)

**Key mismatch parameters:**
- `r0_cm`: Fried parameter (coherence length of atmosphere); nominal 10 cm, perturbed 5–7 cm
- `wind_speed_ms`: frozen-flow wind speed affecting temporal evolution; nominal 5 m/s, perturbed 10–15 m/s
- `D_r0_ratio`: telescope diameter / Fried parameter ratio; nominal 8, perturbed 12–16
- `selection_fraction`: fraction of frames used in shift-and-add; nominal 0.10, perturbed 0.03–0.05

**Dataset format:**
- `x_true: (256, 256)` — diffraction-limited ground truth stellar field
- `y: (N, 256, 256)` — burst of N short-exposure speckle frames (N ≈ 100)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Shift-and-Add | Classical | Bates & Cady (1980) *Opt. Commun.* 32:365–369 | Foundational lucky imaging method; aligns frames on brightest speckle before summing |
| Speckle Interferometry (Knox-Thompson) | Classical | Knox & Thompson (1974) *ApJ* 193:45–48 | Recovers Fourier phase from cross-spectrum of speckle frames |
| Blind Deconvolution (MFBD) | Variational | Löfdahl & Scharmer (1994) *A&AS* 107:243–264 | Multi-frame blind deconvolution estimating PSF and object jointly |
| Deep Speckle Reconstruction | Deep Learning | Möckl et al. (2019) *Optica* 6:1405–1410; Guo et al. (2022) *Opt. Express* 30:32 | CNN trained on simulated speckle bursts predicts high-resolution image in single forward pass |

---

## 4. Literature & State of the Art (2024–2025)

1. **Zhang et al. (2024)** "Physics-informed neural networks for atmospheric turbulence image reconstruction," *Optics Letters* — embedded Kolmogorov turbulence statistics as constraints in a score-based generative model for single-frame blind deblurring.
2. **Wizinowich et al. (2024)** "Machine learning-enhanced lucky imaging for 4-meter class telescopes," *PASP* — showed that a ResNet frame-quality classifier improves selection efficiency over rms-sharpness metrics.
3. **Hu et al. (2025)** "Diffusion posterior sampling for astronomical image restoration under unknown turbulence," *MNRAS* — diffusion model with turbulence-aware noise schedule achieves sub-Fried-parameter effective resolution.
4. **Li et al. (2024)** "Transformer-based multi-frame fusion for ground-based solar imaging," *Solar Physics* — vision transformer architecture fuses temporal burst frames to reconstruct solar granulation structure near the diffraction limit.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/lucky_imaging_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/lucky_imaging_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/lucky_imaging_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/lucky_imaging/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Lucky imaging is correctly benchmarked as a burst-deconvolution problem under spatially-varying, time-varying Kolmogorov turbulence. The algorithm routing from shift-and-add through multi-frame blind deconvolution to deep-learning speckle reconstruction covers the progression of the field appropriately. Mismatch parameters (Fried parameter, wind speed, D/r₀ ratio, selection fraction) reflect the dominant atmospheric and observational variables that affect real lucky imaging campaigns.

---
*Comprehensive 6-point check by deep-check pipeline v3*
