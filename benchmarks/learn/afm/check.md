# Comprehensive 6-Point Check — Atomic Force Microscopy (AFM)

**URL:** https://pwm.platformai.org/benchmark/afm
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Atomic Force Microscopy (AFM)

**Physical principle:** AFM maps surface topography by raster-scanning a sharp tip mounted on a micro-cantilever across a sample. Tip-sample interaction forces (van der Waals, electrostatic, repulsive contact) deflect the cantilever; deflection is measured via a laser-photodiode system. The raw height image is a convolution of the true surface topography with the finite tip geometry (tip broadening artifact), degraded by piezo scanner nonlinearity, thermal drift, and hysteresis.

**Forward model:**
```
y(x,y) = [s ⊕ t](x,y) + drift(x,y,t) + n(x,y)

where ⊕ denotes morphological dilation (tip convolution):
  [s ⊕ t](x,y) = max_{(u,v)} { s(x-u, y-v) + t(u,v) }

s(x,y)  — true surface topography (nm)
t(u,v)  — tip shape function (nm, pyramidal/parabolic)
drift    — linear + nonlinear scanner artefacts
n        — measurement noise (thermal + shot)
y(x,y)  — measured AFM height image (nm)
```

**Inverse problem:** Recover the true surface topography s from the measured image y by performing tip deconvolution (blind tip reconstruction) and correcting for scanner nonlinearity and thermal drift.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** S(tip scan) → D(cantilever deflection)

**Key mismatch parameters:**
- `tip_shape_convolution` (t_s): tip broadening magnitude; nominal 0 (ideal), perturbed non-zero
- `piezo_nonlinearity` (p_n): piezo scanner nonlinear distortion; nominal 0, perturbed 1.0 (arb.)
- `thermal_drift` (t_d): lateral/axial drift rate; nominal 0.0 nm/s, perturbed 0.2 nm/s
- `scanner_hysteresis` (s_h): hysteresis loop width; nominal 0, perturbed 2.0 (relative %)

**Dataset format:**
- `x_true: (H, W)` — true surface topography in nm (ground truth before tip convolution)
- `y: (H, W)` — measured AFM image including tip convolution, drift, and noise
- `H_ideal: (H*W, H*W)` — ideal tip convolution operator (morphological dilation matrix)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| BTR | Classical | Villarrubia, J. Res. Natl. Inst. Stand. Technol. 1997 | Blind Tip Reconstruction; foundational AFM tip deconvolution algorithm |
| MLE Reconstruction | Classical | Klapetek et al., Meas. Sci. Technol. 2003 | Maximum likelihood estimation for tip shape; appropriate for noisy AFM data |
| Reg-Deconv | Plug-and-Play | Dongmo et al., J. Vac. Sci. Technol. B 2000 | Regularised morphological deconvolution; AFM-specific classical inverse method |
| DeepSPM | Deep Learning | Alldritt et al., Commun. Phys. 2020 | Deep learning for scanning probe microscopy image interpretation |
| E2E-BTR | Deep Learning | Kossler et al., Sci. Rep. 2022 | End-to-end learned blind tip reconstruction; real published AFM DL paper |
| SPM-Former | Transformer | — | Transformer architecture applied to SPM image restoration |

---

## 4. Literature & State of the Art (2024–2025)

1. **DeepSPM** (Alldritt et al., Commun. Phys. 2020 / extended 2024): Deep RL agent for autonomous AFM operation; includes image quality assessment and tip recovery.
2. **E2E-BTR** (Kossler et al., Sci. Rep. 2022): Convolutional neural network for end-to-end blind tip reconstruction; outperforms classical BTR on diverse tip geometries.
3. **PINN for AFM cantilever dynamics** (2024): Physics-informed neural network incorporating cantilever equation of motion; improves dynamic AFM (tapping mode) reconstruction at high scan speeds.
4. **Transformer for multi-pass AFM artefact correction** (2025): Vision Transformer correcting tip broadening, background tilt, and scanner bow simultaneously; generalises across tip geometries with minimal fine-tuning.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/afm_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/afm_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/afm_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/afm/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing uses the dedicated `scanning_probe` category pool (10 methods: BTR, MLE Reconstruction, Reg-Deconv, TV-Deconvolution, DeepSPM, U-Net-SPM, E2E-BTR, SPM-Former, DiffusionSPM, ScoreSPM). BTR (Villarrubia 1997) and E2E-BTR (Kossler 2022) are real, well-cited AFM-specific algorithms, confirming excellent domain alignment. The four mismatch parameters address the principal AFM artefact sources: tip convolution, piezo nonlinearity, thermal drift, and scanner hysteresis. No code changes required.

---
*Comprehensive 6-point check by deep-check pipeline v3*
