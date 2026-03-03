# Comprehensive 6-Point Check -- spectral_ct

**URL:** https://pwm.platformai.org/benchmark/spectral_ct
**Check Date:** 2026-03-03
**Status:** PASS (acceptable category routing, no code changes needed)

---

## 1. Physics & Forward Model

**Modality:** Spectral CT (Photon-Counting CT / Dual-Energy CT)

**Physical principle:** Spectral CT acquires X-ray projection data at multiple photon energies simultaneously, either via photon-counting detectors (PCD-CT) that bin individual photons by energy, or via dual-source/dual-layer detector configurations. The energy-dependent X-ray attenuation follows:
```
mu(E) = sum_m  rho_m * sigma_m(E)
```
where rho_m = density of material m, sigma_m(E) = energy-dependent mass attenuation coefficient. By measuring at multiple energies, material composition can be decomposed (bone vs soft tissue vs contrast agent).

**Forward model (per energy bin):**
```
y_E(i) = -ln( integral S(E) * exp(-integral mu(r,E) ds) dE )
```
This is a nonlinear extension of the standard CT forward model, with the additional challenge of joint energy-dependent reconstruction and material decomposition.

**Inverse problem:** Simultaneously reconstruct images at each energy level AND decompose material composition. This is a higher-dimensional inverse problem than conventional CT.

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** Lambda(E1,E2,...) -> Pi(projection) -> D(g, eta_1)

**Mismatch sources in spectral CT:**
- Charge sharing between detector pixels (PCD)
- Pile-up at high count rates
- K-edge artifacts near contrast agent absorption edges
- Energy calibration errors (bin threshold drift)
- Beam hardening within each energy bin
- Cross-talk between energy channels
- Ring artifacts from detector element variations

**Dataset format (GCS):**
- `x_true` -- ground truth attenuation map (possibly multi-channel for material decomposition)
- `y` -- sinogram measurements (per energy bin)
- `H_ideal` -- projection parameters

**Tier structure:** Public (with x_true), Dev (no x_true), Hidden (blocked).

## 3. Reconstruction Methods & Leaderboard

**Algorithms (medical pool via category routing: medical + X-ray -> medical default):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| FBP | Classical | Analytical baseline | Acceptable -- FBP works per energy bin, standard CT baseline |
| PnP-ADMM | PnP | Venkatakrishnan et al., 2013 | Acceptable -- general PnP framework applicable to spectral CT |
| FBPConvNet | Deep Learning | Jin et al., IEEE TIP 2017 | Acceptable -- learned post-processing for CT (applicable per bin) |
| Learned Primal-Dual | Deep Unrolling | Adler & Oktem, IEEE TMI 2018 | Acceptable -- unrolled optimization for CT |

The medical/CT pool is acceptable for spectral CT: all 4 algorithms work on projection data and can be applied per energy bin. While spectral-specific algorithms (One-Step Spectral CT, Butterfly-Net, DECT-MULTRA) would leverage cross-energy information, the current pool is not incorrect -- it provides valid CT reconstruction baselines.

## 4. Literature & State of the Art (2024--2025)

1. **One-Step Spectral CT** (Long & Fessler, IEEE TMI 2014): Joint reconstruction + material decomposition in a single optimization. Not in current pool but would be a natural addition.
2. **Butterfly-Net** (Fan et al., SIAM JSC 2019): Multi-scale spectral CT network.
3. **DECT-MULTRA** (Zeng et al., IEEE TMI 2021): Dictionary-based multi-energy learned transform.
4. **Siemens NAEOTOM Alpha** (2024): First clinical PCD-CT scanner -- drives demand for spectral reconstruction algorithms.
5. **Material decomposition with DL** (2024): End-to-end networks for joint reconstruction and material decomposition.
6. **Virtual monoenergetic imaging** (2024--2025): Deep learning for generating virtual monoenergetic images from spectral CT data.

## 5. Local Dataset & GCS Status

**GCS datasets verified:**
- `spectral_ct_challenge_public.h5` -- present on GCS
- `spectral_ct_challenge_dev.h5` -- present on GCS (x_true stripped)
- `spectral_ct_challenge_hidden.h5` -- present on GCS (blocked from download)

**Gallery images:** No gallery section for this modality (page size ~57 KB).

**Learning materials:** Complete 5-module set present (README, physics fundamentals, forward model, reconstruction algorithms, PWM benchmark, hands-on tutorial).

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS -- no code changes needed.

**Routing:** Falls to `_CATEGORY_ALGORITHMS["medical"]` (X-ray carrier uses the default medical/CT pool). This is acceptable because:
- FBP is the standard baseline for any CT reconstruction (applicable per energy bin)
- PnP-ADMM is a general framework that works for spectral CT
- FBPConvNet and Learned Primal-Dual are proven CT reconstruction methods
- The algorithms are not wrong -- they just do not exploit cross-energy correlations

**Domain accuracy note:** Spectral-specific algorithms (One-Step Spectral, Butterfly-Net) would better represent the field but are optional enhancements. The current CT pool correctly tests the tomographic inverse-problem framework.

**No changes required.** The medical/CT pool is a valid assignment for spectral CT.

---
*Comprehensive 6-point check by deep-check pipeline v3*
