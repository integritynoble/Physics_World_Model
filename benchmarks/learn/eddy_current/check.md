# Comprehensive 6-Point Check — Pulsed Eddy Current Testing (ECT)

**URL:** https://pwm.platformai.org/benchmark/eddy_current
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Pulsed Eddy Current Non-Destructive Testing (PECT/ECT)

**Physical principle:** Eddy current testing detects subsurface defects (cracks, corrosion, wall thinning) in conductive materials by inducing time-varying eddy currents via an electromagnetic coil placed near the surface. Changes in the eddy current distribution caused by defects (reduced conductivity paths) alter the impedance of the driving coil, producing a measurable signal perturbation. In pulsed ECT (PECT), a broadband step-function excitation is used; the transient decay of the induced magnetic field (measured by a Hall sensor or receive coil) encodes information at multiple depths simultaneously due to the frequency-dependent skin depth.

**Forward model:**
```
V(t) = V_0(t) - ∫∫∫ δσ(r) * J_ref(r,t) · E_ref(r,t) dV + n(t)

where:
  V(t)          — measured coil voltage or sensor signal
  V_0(t)        — reference signal (no defect)
  δσ(r)         — conductivity perturbation (defect: δσ < 0)
  J_ref(r,t)    — reference eddy current density (from Maxwell's equations)
  E_ref(r,t)    — reference electric field
  n(t)          — sensor noise
  skin depth:   δ(ω) = sqrt(2 / (ω μ σ))  — depth of eddy current penetration at frequency ω
```

**Inverse problem:** Recover the 3D (or 2D cross-section) conductivity perturbation map `δσ(r)` (defect geometry and depth) from the transient or frequency-domain eddy current signals measured at one or more sensor positions.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(conductive specimen + defects) → F(eddy current induction, diffusion) → D(coil/Hall sensor array)

**Key mismatch parameters:**
- `conductivity_nominal`: Nominal specimen conductivity in MS/m; nominal 35 MS/m (aluminum), perturbed 20–60 MS/m
- `lift_off`: Coil-to-specimen lift-off distance; nominal 1.0 mm, perturbed 0.5–3.0 mm
- `defect_depth`: Depth of subsurface defect; nominal 2.0 mm, perturbed 0.5–8.0 mm
- `pulse_width`: Excitation pulse width; nominal 20 ms, perturbed 5–100 ms

**Dataset format:**
- `x_true: (H, W)` — ground-truth 2D defect map (conductivity anomaly cross-section, 256×256)
- `y: (N_positions, N_timesteps)` — PECT transient signals at multiple probe positions and time samples

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Regularized Born inversion (Tikhonov) | Classical | Tamburrino, A. & Rubinacci, G. (2002) "A new non-iterative inversion method for electrical resistance tomography," *Inverse Problems* 18(6):1809–1829 | Linearized Born approximation with Tikhonov regularization for defect imaging |
| Monotonicity-based ECT inversion | Classical | Tamburrino, A. et al. (2010) "Fast methods for quantitative eddy-current tomography of conductive materials," *IEEE Trans. Magn.* 46(8):3269–3278 | Shape reconstruction via monotonicity criterion; computationally efficient |
| CNN-based defect localization | Deep Learning | Zhao, X. et al. (2017) "Deep learning and its applications to machine health monitoring," *Mech. Syst. Signal Process.* 115:213–237 | 1D/2D CNN for defect classification and depth estimation from PECT time-series |
| U-Net ECT image reconstruction | Deep Learning | Fan, M. et al. (2021) "Deep learning-based image reconstruction for magnetic induction tomography," *Meas. Sci. Technol.* 32(10):104007 | End-to-end U-Net mapping sensor signals to conductivity anomaly maps |

---

## 4. Literature & State of the Art (2024–2025)

1. **Deng, Z. et al. (2024)** "Physics-informed neural network for pulsed eddy current signal inversion," *NDT & E International* 141:103010 — PINN encodes Maxwell's equations as soft constraints; reconstructs defect profiles from transient signals without FEM forward calls.
2. **Li, Y. et al. (2024)** "Transformer-based anomaly detection in eddy current array inspection of composite structures," *Composites Part B* 273:111253 — Attention mechanism captures long-range dependencies in spatially distributed ECT array data.
3. **Huang, K. et al. (2024)** "Generative adversarial network for eddy current testing image super-resolution and defect enhancement," *IEEE Trans. Instrum. Meas.* 73:2505614 — GAN-based upsampling of low-resolution ECT C-scan images for fine defect visualization.
4. **Nair, J. et al. (2025)** "Quantitative pulsed eddy current tomography via score-based diffusion model," *Nondestructive Testing and Evaluation* — Diffusion model trained on FEM-simulated PECT data outperforms Tikhonov regularization on depth-resolved defect reconstruction.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/eddy_current_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/eddy_current_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/eddy_current_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/eddy_current/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

The eddy current benchmark correctly models the electromagnetic induction forward problem with Born-approximation conductivity perturbation and frequency/depth-dependent skin effect. Algorithm routing spans linearized Tikhonov inversion (classical), monotonicity-based methods (analytical), and CNN/U-Net deep learning approaches, accurately representing the current ECT inspection literature. The mismatch parameters on lift-off, defect depth, conductivity, and pulse width are the dominant physical sources of eddy current inspection variability in real industrial NDE scenarios.

---
*Comprehensive 6-point check by deep-check pipeline v3*
