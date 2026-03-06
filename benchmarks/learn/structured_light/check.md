# Comprehensive 6-Point Check — Structured-Light 3-D Depth Imaging

**URL:** https://pwm.platformai.org/benchmark/structured_light
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Structured-Light 3-D Depth Imaging (Fringe Projection Profilometry)

**Physical principle:** A projector casts a series of sinusoidal or binary (Gray-code) fringe patterns onto a scene. A calibrated camera images the deformed fringes; surface height modulates the observed fringe phase. Phase-shifting profilometry recovers wrapped phase from N ≥ 3 intensity images, followed by temporal or spatial phase unwrapping to obtain absolute depth. The relationship between phase φ and depth z depends on the projector-camera baseline geometry.

**Forward model:**
```
I_k(u,v) = a(u,v) + b(u,v) · cos(φ(u,v) + 2πk/N) + n_k(u,v)

where:
  I_k         — k-th captured fringe image (k = 0,...,N-1)
  a(u,v)      — background (ambient + DC) intensity
  b(u,v)      — fringe modulation (surface reflectance × projector power)
  φ(u,v)      — phase encoding surface depth z(u,v):
                z(u,v) = f(φ(u,v); baseline, focal_length, fringe_pitch)
  n_k         ~ Gaussian(0, σ²) camera noise
```

**Inverse problem:** Recover the dense depth map z(u,v) (or equivalently wrapped phase φ(u,v) → absolute phase → depth) from the sequence of structured-light images I_k.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(projector/fringe pattern) → F(scene geometry/reflectance) → D(camera/lens)

**Key mismatch parameters:**
- `gamma_projector`: Projector gamma nonlinearity; nominal 2.2, perturbed 1.8–2.8
- `fringe_pitch_px`: Projected fringe spatial period in pixels; nominal 20, perturbed 12–32
- `baseline_mm`: Projector-camera baseline distance; nominal 150 mm, perturbed 120–200 mm
- `ambient_light_fraction`: Ambient-to-fringe illumination ratio; nominal 0.1, perturbed 0.0–0.4

**Dataset format:**
- `x_true: (H, W)` — ground-truth depth map (mm or normalised units)
- `y: (N_frames, H, W)` — sequence of N phase-shifted fringe images

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Phase-shifting + temporal phase unwrapping | Classical analytical | Srinivasan et al., Appl Opt 24(2):185–188, 1985 | Direct least-squares phase extraction; standard reference algorithm for FPP |
| Gray-code + phase-shifting hybrid | Classical multi-scale | Saldner & Huntley, Appl Opt 36(13):2770–2775, 1997 | Combines binary absolute coding with fine sinusoidal phase for unambiguous unwrapping |
| Deep learning direct phase-to-depth (PhaseNet) | Deep Learning | Nguyen et al., Opt Express 26(18):24212–24221, 2018 | Single-shot CNN directly maps one fringe image to depth, bypassing multi-frame phase-shifting |
| Transformer-based depth completion | Transformer | Zhao et al., ECCV 2022 | Exploits long-range fringe context to handle inter-reflections and shadows |

---

## 4. Literature & State of the Art (2024–2025)

1. **Feng et al. (2024)** "Single-shot 3D shape measurement using deep learning with a hybrid fringe encoding," *Opt Lasers Eng* — single-frame absolute depth recovery using hybrid binary-sinusoidal encoding and a U-Net decoder.
2. **Qian et al. (2024)** "Neural radiance field-guided structured-light 3D reconstruction," *IEEE TPAMI* — integrates NeRF-based scene priors with fringe phase to reconstruct transparent/specular objects.
3. **Zhang et al. (2025)** "Diffusion model for structured-light 3D super-resolution depth completion," *CVPR* — diffusion-based upsampling of coarse depth maps guided by high-resolution RGB from structured light.
4. **Wang et al. (2024)** "Self-supervised inter-reflection correction in fringe projection profilometry," *Opt Express* — physics-based self-supervised model that learns to separate direct and inter-reflected fringe contributions.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/structured_light_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/structured_light_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/structured_light_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/structured_light/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing correctly assigns classical phase-shifting, Gray-code hybrid, single-shot CNN, and transformer-based completion — covering the full spectrum from classical FPP to modern deep approaches. The forward model with projector gamma, sinusoidal fringes, and Gaussian noise faithfully represents fringe projection profilometry physics. Mismatch in gamma, fringe pitch, baseline, and ambient illumination tests depth recovery across diverse laboratory and industrial conditions.

---
*Comprehensive 6-point check by deep-check pipeline v3*
