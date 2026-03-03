# Comprehensive 6-Point Check -- passive_microwave

**URL:** https://pwm.platformai.org/benchmark/passive_microwave
**Check Date:** 2026-03-03
**Status:** PASS (correct override routing, no code changes needed)

---

## 1. Physics & Forward Model

**Modality:** Passive Microwave Radiometry

**Physical principle:** Passive microwave radiometry measures naturally emitted microwave radiation (brightness temperature) from the Earth's surface and atmosphere. Every object above absolute zero emits thermal radiation; at microwave frequencies (1-200 GHz), the emitted power is proportional to the physical temperature and emissivity:
```
T_B(f, theta) = epsilon(f, theta) * T_phys + (1 - epsilon) * T_sky_reflected
```
where T_B = brightness temperature, epsilon = surface emissivity, T_phys = physical temperature.

For aperture synthesis radiometers (e.g., SMOS, AMSR-E), the measurement is a convolution of the true brightness temperature field with the antenna pattern. The inverse problem is to recover the high-resolution T_B field from the antenna-smoothed measurements.

**Inverse problem:** Antenna pattern deconvolution (aperture synthesis) or brightness temperature retrieval for geophysical parameter estimation (soil moisture, sea surface temperature, atmospheric water vapor).

**Current physics engine:** Convolution/deconvolution forward model, appropriate for aperture synthesis inversion.

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** Sigma -> D

**Mismatch sources in passive microwave:**
- Antenna sidelobe contamination
- Cross-polarization leakage
- Radio frequency interference (RFI)
- Atmospheric attenuation correction errors
- Faraday rotation (L-band, low frequencies)
- Surface roughness and vegetation effects on emissivity

**Dataset format (GCS):**
- `x_true` -- high-resolution brightness temperature field
- `y` -- antenna-smoothed/degraded measurement
- `H_ideal` -- forward model parameters

**Tier structure:** Public (with x_true), Dev (no x_true), Hidden (blocked).

## 3. Reconstruction Methods & Leaderboard

**Algorithms (remote_sensing override for passive microwave):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Backus-Gilbert | Classical | Backus & Gilbert, 1970 | CORRECT -- standard aperture synthesis deconvolution for radiometry |
| Tikhonov-SMOS | Classical/PnP | Tikhonov regularized inversion for SMOS | CORRECT -- regularized inverse for microwave radiometry |
| RadioNet | Deep Learning | CNN-based brightness temperature retrieval | CORRECT -- learned radiometric inversion |
| MWR-Former | Transformer | Transformer for microwave radiometry | CORRECT -- modern DL architecture for radiometric data |

All 4 algorithms are domain-appropriate for passive microwave radiometry. This is a significant improvement over the previous assignment (SAR algorithms: Matched Filter, SAR-BM3D, SAR-DRN, SAR-CAM), which were entirely wrong for passive radiometry.

## 4. Literature & State of the Art (2024--2025)

1. **Backus-Gilbert inversion** (Backus & Gilbert, 1970): Classical optimal linear estimation for aperture synthesis -- the reference method for microwave radiometry deconvolution.
2. **SMOS image reconstruction** (Anterrieu, 2004): Regularized inversion for SMOS L-band aperture synthesis radiometer.
3. **AMSR-E/AMSR2 retrieval algorithms** (ongoing): Operational soil moisture and SST retrieval from radiometric data.
4. **CNN for microwave retrieval** (Turk et al., IEEE TGRS 2022): Deep learning approaches for brightness temperature to geophysical parameter inversion.
5. **RFI mitigation** (2024): Machine learning approaches for detecting and removing radio frequency interference in radiometric measurements.
6. **CIMR mission** (2025): Copernicus Imaging Microwave Radiometer -- next-generation passive microwave instrument with ML-ready data products.

## 5. Local Dataset & GCS Status

**GCS datasets verified:**
- `passive_microwave_challenge_public.h5` -- present on GCS
- `passive_microwave_challenge_dev.h5` -- present on GCS (x_true stripped)
- `passive_microwave_challenge_hidden.h5` -- present on GCS (blocked from download)

**Gallery images:** Served from GCS (no gallery images for this modality -- page size 56,812 bytes indicates no gallery section).

**Learning materials:** Complete 5-module set present (README, physics fundamentals, forward model, reconstruction algorithms, PWM benchmark, hands-on tutorial).

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS -- no code changes needed.

**Routing:** Passive microwave has a dedicated override that provides radiometry-appropriate algorithms (Backus-Gilbert, Tikhonov-SMOS, RadioNet, MWR-Former) instead of the SAR algorithms from the generic remote_sensing pool.

**Previously fixed:** passive_microwave was originally getting SAR-specific algorithms (Matched Filter, SAR-BM3D, SAR-DRN, SAR-CAM) via the `remote_sensing` category default. SAR is an active coherent imaging system fundamentally different from passive radiometry. The override resolved this mismatch.

**No further changes required.** The algorithm assignment is correct and domain-appropriate.

---
*Comprehensive 6-point check by deep-check pipeline v3*
