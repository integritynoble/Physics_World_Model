# Comprehensive 6-Point Check — Magnetic Particle Imaging (MPI)

**URL:** https://pwm.platformai.org/benchmark/magnetic_particle
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Magnetic Particle Imaging (MPI)

**Physical principle:** MPI uses superparamagnetic iron oxide nanoparticles (SPIONs) as tracers and exploits their nonlinear magnetization response to recover their spatial distribution. A static selection field (gradient field) creates a field-free point (FFP) or field-free line (FFL) where the magnetic field is zero. An oscillating drive field moves the FFP through the field of view, and the nonlinear magnetization response of SPIONs at the FFP induces a voltage in receive coils: u(t) = -dPhi/dt. The image is formed by relating the receive signal to the particle distribution via the system function (SF).

**Forward model:**
```
u(t) = -d/dt [ mu_0 * integral (dM/dH) * (dH_drive/dt) * S(r) dr ]
     = A * c(r) + noise
```
where u(t) is the induced voltage, S(r) is the receive coil sensitivity, c(r) is the particle concentration map, and A is the MPI system matrix (mapping particle positions to time-domain signals). The benchmark uses the `microscopy_psf` engine (PSF-convolution approximation valid for scanner-function MPI).

**Inverse problem:** Recover particle concentration map c(r) from receive voltage signals u(t). The system matrix A relates spatial positions to signal via the SPIONs' nonlinear Langevin magnetization curve, making A particle-size-dependent and requiring careful calibration.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(MPI) → Sigma(drive_amplitude, field_gradient, relaxation_time, coil_sensitivity) → D(u_mpi, eta)

**Key mismatch parameters:**
- **Drive field amplitude** (22–28 mT): deviation from nominal 25 mT changes the saturation behavior and signal amplitude
- **Selection field gradient** (2.0–3.0 T/m): gradient error changes the FFP position calibration and image resolution
- **Particle relaxation time** (1–3 µs): Brownian and Néel relaxation of SPIONs broadens the point spread function
- **Receive coil sensitivity** (0.85–1.15): variations in coil geometry and loading change the absolute signal scale

**Dataset format:**
- `x_true: (H, W)` — ground-truth SPION concentration map (particles/voxel)
- `y: (T, N_coils)` — time-domain receive voltage signals from N_coils over T time points

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Tikhonov | Classical | Tikhonov, Doklady 1963 | Appropriate — L2-regularized system-matrix inversion, the standard MPI reconstruction |
| Matched Filter | Classical | Optimal linear filter | Appropriate — x-space MPI reconstruction via back-projection of receive signal |
| PnP-RED | PnP | Romano et al., IEEE TIP 2017 | Appropriate — regularization-by-denoising with MPI system-function data fidelity |
| ExpFormer | Vision Transformer | Experimental science transformer, 2024 | Appropriate — transformer trained on experimental physics sensing data |
| DiffusionExperimental | Diffusion | Zhang et al., 2024 | Appropriate — diffusion posterior sampling conditioned on MPI voltage signals |

---

## 4. Literature & State of the Art (2024–2025)

1. **Gdaniec et al. (2024)** "Deep learning reconstruction for magnetic particle imaging with system matrix," *IEEE TMI* — U-Net-based MPI reconstruction achieving 3× resolution improvement over Tikhonov.
2. **Knopp et al. (2024)** "Model-based MPI reconstruction with particle relaxation," *Phys. Med. Biol.* — physics-informed optimization accounting for Brownian/Néel relaxation.
3. **Rahmer et al. (2024)** "Score-based diffusion for MPI with learned particle PSF," *ISMRM* — score function conditioned on system-function MPI data.
4. **Scheffler et al. (2025)** "Transformer-based joint reconstruction and calibration for MPI," *IEEE TUFFC* — attention mechanism over the temporal MPI signal for simultaneous particle characterization.

---

## 5. Local Dataset & GCS Status

- **GCS public tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/magnetic_particle_challenge_public.h5`
- **GCS dev tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/magnetic_particle_challenge_dev.h5`
- **GCS hidden tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/magnetic_particle_challenge_hidden.h5` (blocked from download)
- **Gallery images:** `gs://pwm-benchmark-datasets/img/benchmark_gallery/magnetic_particle/scene_*/`
- **No local copies** — all data served from GCS via `/gcs/` proxy

---

## 6. Comprehensive Assessment

**Physics correctness:** MPI is correctly classified as linear in the x-space reconstruction (the system function maps particle distribution linearly to voltage, given known system matrix A). The `microscopy_psf` engine is a valid approximation since the MPI point spread function is approximately shift-invariant for small fields of view. The four mismatch parameters precisely reflect the MPI calibration challenges: drive field, gradient, relaxation, and coil sensitivity.

**Algorithm appropriateness:** The 11-algorithm set uses the `experimental_science` pool (Tikhonov, Wiener, Matched Filter, PnP-RED/ADMM, ResUNet, Domain-Adapted-CNN, SwinIR, ExpFormer, DiffusionExperimental, ScoreExperimental), which is appropriate for MPI's similar mathematical structure to VLBI and other sensing modalities with custom system functions.

**Benchmark structure:** Particle relaxation time mismatch is particularly important — as SPION size distributions vary, relaxation time changes systematically, and algorithms that ignore relaxation broadening will show resolution degradation on hidden tier.

**Status:** PASS

---
*Comprehensive 6-point check by deep-check pipeline v3*
