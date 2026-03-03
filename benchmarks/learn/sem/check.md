# Comprehensive 6-Point Check -- sem

**URL:** https://pwm.platformai.org/benchmark/sem
**Check Date:** 2026-03-03
**Status:** PASS (correct EM routing, no code changes needed)

---

## 1. Physics & Forward Model

**Modality:** Scanning Electron Microscopy (SEM)

**Physical principle:** SEM forms images by rastering a focused electron beam (typically 1-30 keV) across a sample surface. The beam interacts with the sample, producing secondary electrons (SE, topographic contrast), backscattered electrons (BSE, compositional contrast), and characteristic X-rays (elemental analysis). The detected signal at each beam position forms one pixel of the image:
```
y(x, y) = G * eta(E_0) * I_b * x(x, y) + noise
```
where G = detector gain, eta = detection efficiency, I_b = beam current, and x(x,y) = the local emission yield.

**Inverse problem:** SEM images suffer from noise (especially at low dose/fast scan rates), beam damage artifacts, charging (on non-conductive samples), and limited resolution from beam-sample interaction volume. The reconstruction task is primarily image denoising/restoration and resolution enhancement.

**Key distinction from cryo-EM:** SEM images are 2D surface images requiring denoising, NOT 3D single-particle reconstructions from projection images. The algorithm requirements are fundamentally different.

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** C(e-beam) -> D(g, eta_1)

**Mismatch sources in SEM:**
- Beam current fluctuations and drift
- Detector gain variations
- Sample charging (non-conductive specimens)
- Contamination buildup during scanning
- Vibration and electromagnetic interference
- Astigmatism and focus drift
- Beam damage (dose-dependent degradation)

**Dataset format (GCS):**
- `x_true: (512, 512)` -- ground truth high-quality SEM image
- `y: (512, 512)` -- degraded/noisy measurement
- `H_ideal` -- forward model parameters (PSF, noise level)

**Tier structure:** Public (with x_true), Dev (no x_true), Hidden (blocked).

## 3. Reconstruction Methods & Leaderboard

**Algorithms (em_generic pool via special EM routing: non-cryo electron microscopy):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Wiener Filter | Classical | Wiener, 1949 | CORRECT -- standard deconvolution/denoising baseline for SEM |
| BM3D | PnP | Dabov et al., IEEE TIP 2007 | CORRECT -- state-of-the-art patch-based denoising |
| Noise2Void | Deep Learning | Krull et al., CVPR 2019 | CORRECT -- self-supervised denoising (no clean targets needed) |
| SwinIR | Transformer | Liang et al., ICCVW 2021 | CORRECT -- transformer-based image restoration |

All 4 algorithms are domain-appropriate for SEM image denoising/restoration. This is a significant improvement over the previous assignment (cryo-EM single-particle algorithms: RELION, cryoSPARC).

## 4. Literature & State of the Art (2024--2025)

1. **Noise2Void / Noise2Self** (Krull et al., 2019; Batson & Royer, 2019): Self-supervised denoising without paired data -- widely adopted in EM. Already in pool.
2. **Topaz-Denoise** (Bepler et al., 2020): Originally for cryo-EM but also applicable to SEM denoising.
3. **DL-based SEM super-resolution** (2024): GAN and diffusion-based approaches for SEM resolution enhancement.
4. **SEM image segmentation + denoising** (2024): Joint segmentation and restoration for materials science SEM.
5. **Low-dose SEM** (2024--2025): Deep learning for beam-sensitive biological SEM at ultra-low electron doses.
6. **STEM denoising** (2024): Scanning TEM denoising with attention-based architectures applicable to SEM.

## 5. Local Dataset & GCS Status

**GCS datasets verified:**
- `sem_challenge_public.h5` -- present on GCS
- `sem_challenge_dev.h5` -- present on GCS (x_true stripped)
- `sem_challenge_hidden.h5` -- present on GCS (blocked from download)

**Gallery images:** 24/24 load OK from GCS (4 scenes x 6 images).

**Learning materials:** Complete 5-module set present (README, physics fundamentals, forward model, reconstruction algorithms, PWM benchmark, hands-on tutorial).

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS -- no code changes needed.

**Routing:** SEM is routed to the `em_generic` pool via special EM routing logic that distinguishes cryo-EM variants (which need RELION/cryoSPARC) from non-cryo EM variants (which need image denoising/restoration).

**Previously fixed:** SEM was incorrectly getting cryo-EM single-particle reconstruction algorithms (RELION, cryoSPARC) via the generic `electron_microscopy` category. The special EM routing sends non-cryo variants to the `em_generic` pool with appropriate denoising algorithms.

**No further changes required.** The algorithm assignment is correct.

---
*Comprehensive 6-point check by deep-check pipeline v3*
