# Comprehensive 6-Point Check -- stm

**URL:** https://pwm.platformai.org/benchmark/stm
**Check Date:** 2026-03-03
**Status:** PASS (acceptable category routing, no code changes needed)

---

## 1. Physics & Forward Model

**Modality:** Scanning Tunneling Microscopy (STM)

**Physical principle:** STM measures the quantum tunneling current between a sharp conductive tip and a sample surface as the tip is rastered at sub-nanometer distances. The tunneling current depends exponentially on the tip-sample distance and the local density of states (LDOS):
```
I_t ~ V_bias * rho(r, E_F) * exp(-2*kappa*d)
```
where rho = local density of electronic states at the Fermi level, d = tip-sample distance, kappa = decay constant (~1 inverse Angstrom), and V_bias = applied bias voltage.

In constant-current mode, a feedback loop adjusts tip height z(x,y) to maintain constant tunneling current, producing a topographic map that convolves true topography with electronic structure (LDOS variations). In constant-height mode, the tunneling current I(x,y) directly maps the LDOS.

**Inverse problem:** Recover the true surface structure (topography and/or electronic properties) from images affected by tip artifacts (electronic and geometric convolution), piezo drift/creep, vibration noise, and thermal drift.

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** C(tunnel) -> D(g, eta_1)

**Mismatch sources in STM:**
- Tip electronic structure (LDOS of the tip apex affects the image)
- Piezo drift and creep (scanner nonlinearity)
- Thermal drift during long scans
- Vibration isolation limitations
- Tip changes during scanning (atomic rearrangement)
- Feedback loop bandwidth limitations
- Bias-dependent contrast variations

**Dataset format (GCS):**
- `x_true` -- ground truth surface map
- `y` -- degraded STM measurement
- `H_ideal` -- forward model parameters

**Tier structure:** Public (with x_true), Dev (no x_true), Hidden (blocked).

## 3. Reconstruction Methods & Leaderboard

**Algorithms (scanning_probe category pool):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| BTR | Classical | Villarrubia, JRNIST 1997 | Acceptable -- Blind Tip Reconstruction, originally for AFM but applicable to STM tip deconvolution |
| Reg-Deconv | PnP | Dongmo et al., 2000 | Acceptable -- regularized deconvolution for tip artifact removal |
| DeepSPM | Deep Learning | Alldritt et al., Commun. Phys. 2020 | CORRECT -- specifically designed for STM molecular identification |
| E2E-BTR | Deep Learning | Kossler et al., Sci. Rep. 2022 | Acceptable -- end-to-end blind tip reconstruction |

The scanning_probe pool is designed primarily for AFM tip deconvolution, but the algorithms are applicable to STM as well. While STM has additional challenges (LDOS convolution, electronic tip effects), the geometric tip artifact problem is shared with AFM. DeepSPM is specifically an STM method.

## 4. Literature & State of the Art (2024--2025)

1. **DeepSPM** (Alldritt et al., Commun. Phys. 2020): Deep learning for identifying molecular structures from STM images. Already in pool -- the most STM-specific algorithm.
2. **AiSTM** (2024): AI-assisted autonomous STM operation and image analysis.
3. **Drift correction** (standard): Post-processing for piezo drift and thermal creep -- a key STM preprocessing step.
4. **DL-based STM simulation** (2024): Neural network potentials for predicting STM images from atomic structures.
5. **Machine learning for STM spectroscopy** (2024--2025): Automated STS analysis for electronic structure mapping.
6. **STM image super-resolution** (2024): Deep learning enhancement of STM resolution beyond tip limitations.

## 5. Local Dataset & GCS Status

**GCS datasets verified:**
- `stm_challenge_public.h5` -- present on GCS
- `stm_challenge_dev.h5` -- present on GCS (x_true stripped)
- `stm_challenge_hidden.h5` -- present on GCS (blocked from download)

**Gallery images:** No gallery section for this modality (page size ~56 KB).

**Learning materials:** Complete 5-module set present (README, physics fundamentals, forward model, reconstruction algorithms, PWM benchmark, hands-on tutorial).

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS -- no code changes needed.

**Routing:** Falls to `_CATEGORY_ALGORITHMS["scanning_probe"]`. The scanning_probe pool is shared between AFM and STM, which is acceptable because:
- Both are scanning probe techniques with tip-artifact convolution
- BTR and Reg-Deconv address tip deconvolution (relevant for both modalities)
- DeepSPM is specifically an STM algorithm (already in the pool)
- E2E-BTR provides a learned end-to-end approach

**Domain accuracy note:** STM has additional complexity beyond tip artifacts (LDOS convolution, electronic structure effects) that AFM does not. However, the current pool's focus on tip deconvolution is the primary shared artifact, and DeepSPM provides STM-specific coverage.

**No changes required.** The scanning_probe pool is a defensible assignment for STM.

---
*Comprehensive 6-point check by deep-check pipeline v3*
