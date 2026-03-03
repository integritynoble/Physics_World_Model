# Comprehensive 6-Point Check -- radio_astronomy

**URL:** https://pwm.platformai.org/benchmark/radio_astronomy
**Check Date:** 2026-03-03
**Status:** PASS (correct astronomy override, no code changes needed)

---

## 1. Physics & Forward Model

**Modality:** Radio Astronomy (Aperture Synthesis Imaging)

**Physical principle:** Radio telescopes (single-dish or interferometric arrays) observe celestial radio sources at frequencies from ~10 MHz to ~300 GHz. In aperture synthesis, pairs of antennas measure the complex visibility function V(u,v) -- the Fourier transform of the sky brightness distribution I(l,m) -- at spatial frequencies determined by the antenna separation (baseline):
```
V(u, v) = integral integral  I(l, m) * exp(-j*2*pi*(u*l + v*m)) dl dm
```
The (u,v) coverage is incomplete (sparse sampling), making image reconstruction an ill-posed inverse problem.

**Inverse problem:** Recover the sky brightness distribution I(l,m) from sparse, noisy visibility measurements V(u,v). This is a Fourier inversion problem with incomplete sampling, requiring deconvolution of the "dirty beam" (PSF determined by the (u,v) coverage).

**Current physics engine:** Fourier-sampling forward model, appropriate for aperture synthesis.

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** R(theta) -> Sigma(visibility) -> D(g, eta_1)

**Mismatch sources in radio astronomy:**
- Incomplete (u,v) coverage (Earth rotation synthesis)
- Antenna-based gain and phase errors (calibration)
- Atmospheric phase fluctuations (tropospheric/ionospheric)
- Radio frequency interference (RFI)
- Wide-field effects (w-term, non-coplanar baselines)
- Bandwidth smearing and time-average smearing
- Primary beam variations across the field

**Dataset format (GCS):**
- `x_true` -- ground truth sky brightness distribution
- `y` -- visibility measurements (sparse Fourier samples)
- `H_ideal` -- forward model parameters / (u,v) coverage

**Tier structure:** Public (with x_true), Dev (no x_true), Hidden (blocked).

## 3. Reconstruction Methods & Leaderboard

**Algorithms (astronomy override):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| CLEAN | Classical | Hogbom, A&AS 1974 | CORRECT -- the foundational radio imaging algorithm, used for 50 years |
| AIRI | PnP | Terris et al., MNRAS 2022 | CORRECT -- PnP approach specifically for radio interferometric imaging |
| R2D2 | Deep Learning | Aghabiglou et al., ApJS 2024 | CORRECT -- deep learning for radio imaging |
| PRIMO | Deep Learning | Medeiros et al., ApJL 2023 | CORRECT -- principal-component interferometric modeling (used for EHT M87 image) |

All 4 algorithms are domain-appropriate for radio astronomy imaging. This is a significant improvement over the previous assignment (generic experimental_science: Tikhonov, PnP-RED, ResUNet, SwinIR).

## 4. Literature & State of the Art (2024--2025)

1. **CLEAN** (Hogbom, 1974): The standard radio imaging deconvolution algorithm. Variants include MS-CLEAN, MF-CLEAN, and MTMFS. Still the default in CASA.
2. **AIRI** (Terris et al., MNRAS 2022): AI for Regularization in Radio-Interferometric Imaging -- PnP with learned denoisers. Already in pool.
3. **R2D2** (Aghabiglou et al., ApJS 2024): Residual-to-Residual DNN series for radio imaging. Already in pool.
4. **PRIMO** (Medeiros et al., ApJL 2023): Used to produce the sharpest EHT image of M87*. Already in pool.
5. **uSARA** (Terris et al., MNRAS 2023): Unconstrained Sparsity Averaging Reweighted Analysis for wide-band radio imaging.
6. **ngEHT imaging** (2024--2025): Next-generation Event Horizon Telescope with ML-driven image reconstruction pipelines.

## 5. Local Dataset & GCS Status

**GCS datasets verified:**
- `radio_astronomy_challenge_public.h5` -- present on GCS
- `radio_astronomy_challenge_dev.h5` -- present on GCS (x_true stripped)
- `radio_astronomy_challenge_hidden.h5` -- present on GCS (blocked from download)

**Gallery images:** No gallery section for this modality (page size 57,262 bytes).

**Learning materials:** Complete 5-module set present (README, physics fundamentals, forward model, reconstruction algorithms, PWM benchmark, hands-on tutorial).

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS -- no code changes needed.

**Routing:** Radio astronomy is correctly routed to the astronomy algorithm pool (CLEAN, AIRI, R2D2, PRIMO). This was previously fixed from the generic experimental_science pool.

**Previously fixed:** radio_astronomy was originally categorized under `experimental_science` and received generic algorithms (Tikhonov, PnP-RED, ResUNet, SwinIR). The astronomy override provides the correct domain-specific algorithms.

**No further changes required.** The algorithm assignment is correct and domain-appropriate.

---
*Comprehensive 6-point check by deep-check pipeline v3*
