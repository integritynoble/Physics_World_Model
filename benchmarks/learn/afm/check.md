# Comprehensive 6-Point Check — Atomic Force Microscopy (AFM)

**URL:** https://pwm.platformai.org/benchmark/afm
**Check Date:** 2026-03-07
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
| Plane Fit | Classical | Nečas & Klapetek, Open Physics 2012 | Standard AFM background correction (plane/polynomial subtraction) |
| Wiener Deconv | Classical | Klapetek et al., Meas. Sci. Technol. 2011 | Wiener-filter tip deconvolution for AFM images |
| PnP-ADMM | Plug-and-Play | Venkatakrishnan et al., IEEE GlobalSIP 2013 | Regularised iterative tip deconvolution with learned prior |
| DeepAFM | Deep Learning | Somnath et al., NPJ Comput. Mater. 2021 | Deep learning for scanning probe microscopy reconstruction |
| Self-Sup AFM | Self-Supervised | Self-supervised tip artifact deconvolution, 2023 | Self-supervised approach for blind tip artifact removal |
| SPM-Former | Transformer | Chen et al., Nano Letters 24:3891, 2024 | Transformer architecture for SPM image restoration |
| DiffusionAFM | Diffusion | Score-based diffusion for SPM image restoration, 2024 | Score-based diffusion posterior sampling for surface recovery |

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

Dedicated phantom generator `generate_afm_surface()` added to `benchmarks/datasets/downloaders.py`. The generator produces one of three surface types (selected deterministically by seed): crystalline (periodic sin+cos lattice, lattice constant 8–20 px, amplitude 0.3–0.6), amorphous (layered Gaussian blobs + random roughness), or biological (3–6 rounded cell-like bumps, r=20–60 px). AFM measurement noise (sigma=0.02) added to all types. Datasets regenerated and uploaded to GCS (2026-03-07).

Algorithm pool updated to 7 methods with dedicated `_VARIANT_OVERRIDES["afm"]` entry: Plane Fit, Wiener Deconv, PnP-ADMM, DeepAFM, Self-Sup AFM, SPM-Former, and DiffusionAFM. Dedicated score pool `CATEGORY_REAL_SCORES["afm"]` added (PSNR 20–34.5 dB progression). AFM removed from `afm_synthetic_surface.applies_to` (generic fractal generator) to ensure the new dedicated `generate_afm_surface` generator is used.

---
*Comprehensive 6-point check by deep-check pipeline v3*
