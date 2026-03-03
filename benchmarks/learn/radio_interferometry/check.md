# Comprehensive 6-Point Check -- radio_interferometry

**URL:** https://pwm.platformai.org/benchmark/radio_interferometry
**Check Date:** 2026-03-03
**Status:** PASS (correct astronomy override, no code changes needed)

---

## 1. Physics & Forward Model

**Modality:** Radio Interferometry (VLBI -- Very Long Baseline Interferometry)

**Physical principle:** Radio interferometry uses widely separated antennas (baselines up to thousands of km for VLBI) to achieve angular resolution far beyond what a single dish can provide. Each pair of antennas measures one visibility point V(u,v) -- a sample of the Fourier transform of the sky brightness distribution. The van Cittert-Zernike theorem relates visibility to sky brightness:
```
V(u, v) = integral integral  I(l, m) * exp(-j*2*pi*(u*l + v*m)) dl dm
```
For VLBI, the baselines are so long that micro-arcsecond resolution is achievable (e.g., Event Horizon Telescope imaging of black hole shadows).

**Inverse problem:** Reconstruct the sky brightness distribution from extremely sparse (u,v) coverage. VLBI has far fewer baselines than connected-element arrays, making the inverse problem severely ill-posed. Calibration (antenna gains, atmospheric delays, clock offsets) must be solved simultaneously with imaging.

**Current physics engine:** Fourier-sampling forward model with sparse baseline coverage.

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** R(theta) -> Sigma(visibility) -> D(g, eta_1)

**Mismatch sources in radio interferometry / VLBI:**
- Extremely sparse (u,v) coverage (few baselines)
- Antenna-based complex gain errors (amplitude + phase)
- Tropospheric and ionospheric phase corruption
- Bandwidth decorrelation
- Clock offset and rate errors between stations
- Polarization leakage (D-terms)
- Thermal noise (system temperature dependent)
- Source variability during observation

**Dataset format (GCS):**
- `x_true` -- ground truth sky brightness distribution
- `y` -- visibility measurements
- `H_ideal` -- forward model / baseline parameters

**Tier structure:** Public (with x_true), Dev (no x_true), Hidden (blocked).

## 3. Reconstruction Methods & Leaderboard

**Algorithms (astronomy override):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| CLEAN | Classical | Hogbom, A&AS 1974 | CORRECT -- the workhorse of radio interferometric imaging |
| AIRI | PnP | Terris et al., MNRAS 2022 | CORRECT -- PnP for radio interferometric imaging |
| R2D2 | Deep Learning | Aghabiglou et al., ApJS 2024 | CORRECT -- deep learning for radio imaging |
| PRIMO | Deep Learning | Medeiros et al., ApJL 2023 | CORRECT -- principal-component interferometric modeling |

All 4 algorithms are domain-appropriate. These are the same algorithms used for radio_astronomy, which is correct since both modalities solve the same Fourier inversion problem from sparse visibility data. This is a significant improvement over the previous assignment (SAR algorithms: Matched Filter, SAR-BM3D, SAR-DRN, SAR-CAM).

## 4. Literature & State of the Art (2024--2025)

1. **CLEAN + self-calibration** (Cornwell & Fomalont, 1999): Standard iterative imaging + calibration loop for radio interferometry.
2. **EHT imaging pipelines** (Event Horizon Telescope Collaboration, 2019--2024): Multiple imaging methods (CLEAN, RML, SMILI, Themis) applied to M87* and Sgr A*. PRIMO is among these.
3. **AIRI** (Terris et al., MNRAS 2022): Demonstrated on VLA and MeerKAT data. Already in pool.
4. **R2D2** (Aghabiglou et al., ApJS 2024): Series-based deep learning for radio imaging. Already in pool.
5. **DoG-HiT** (Dabbech et al., MNRAS 2022): Sparsity-based algorithm with data-driven wavelet dictionaries.
6. **ngEHT** (2024--2025): Next-generation Event Horizon Telescope with extended baselines to space -- pushes imaging algorithms to even sparser (u,v) coverage.

## 5. Local Dataset & GCS Status

**GCS datasets verified:**
- `radio_interferometry_challenge_public.h5` -- present on GCS
- `radio_interferometry_challenge_dev.h5` -- present on GCS (x_true stripped)
- `radio_interferometry_challenge_hidden.h5` -- present on GCS (blocked from download)

**Gallery images:** No gallery section for this modality (page size ~57 KB).

**Learning materials:** Complete 5-module set present (README, physics fundamentals, forward model, reconstruction algorithms, PWM benchmark, hands-on tutorial).

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS -- no code changes needed.

**Routing:** Radio interferometry is correctly routed to the astronomy algorithm pool (CLEAN, AIRI, R2D2, PRIMO) via a variant-level override. This was previously fixed from the generic remote_sensing/SAR pool.

**Previously fixed:** radio_interferometry was originally categorized under `remote_sensing` with RF carrier and received SAR algorithms (Matched Filter, SAR-BM3D, SAR-DRN, SAR-CAM). This was fundamentally wrong -- VLBI is astronomical Fourier-plane imaging, not synthetic aperture radar. The variant override provides the correct algorithms.

**Design choice:** A variant-level override was used (rather than carrier routing) because `("remote_sensing", "RF")` should still map to SAR algorithms for actual SAR modalities. This was the correct approach.

**No further changes required.** The algorithm assignment is correct and domain-appropriate.

---
*Comprehensive 6-point check by deep-check pipeline v3*
