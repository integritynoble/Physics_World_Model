# Comprehensive 6-Point Check — XFEL Serial Femtosecond Crystallography (SFX)

**URL:** https://pwm.platformai.org/benchmark/xfel_sfx
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** X-ray Free Electron Laser Serial Femtosecond Crystallography (XFEL-SFX)

**Physical principle:** SFX uses ultra-bright femtosecond X-ray pulses (10–100 fs, ~10¹² photons/pulse) from an XFEL to record single-shot diffraction patterns from micro- or nanocrystals in a serial fashion (thousands of crystals, random orientations) before the crystal is destroyed by radiation. Each pattern is a 2-D slice through the 3-D reciprocal lattice at an unknown random orientation. Monte Carlo integration over many patterns reconstructs the full 3-D structure factor set. The "diffract-before-destroy" principle allows room-temperature femtosecond snapshots of biological macromolecules free from radiation damage.

**Forward model:**
```
I_j(q) = W(q) · |F(q)|² · D_j(q) · Ω(q) + n_j(q)

where:
  I_j(q)     — diffraction pattern from crystal j at orientation R_j
  F(q)        — complex structure factor (F = Σ_atom f_a · e^{iq·r_a})
  D_j(q)     = sinc²(N_j · Δq · d_j)  — crystal shape transform (N_j unit cells)
  W(q)       — per-pulse X-ray spectrum and beam profile
  Ω(q)       — solid-angle / polarization / Lorentz correction
  n_j(q)     ~ Poisson(I_j) photon counting noise
  R_j         — unknown random crystal orientation (SO(3) element)
```

**Inverse problem:** Given N_hit diffraction patterns at unknown orientations, determine: (1) each crystal orientation R_j, (2) the merged 3-D structure factors |F(hkl)|², (3) the electron density ρ(r) via phase retrieval.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(XFEL pulse energy/duration) → F(crystal size/mosaicity/solvent) → D(2-D X-ray detector/AGIPD)

**Key mismatch parameters:**
- `photons_per_pulse`: Per-pulse photon count on detector; nominal 10⁶ photons, perturbed 10⁴–10⁸
- `crystal_size_um`: Crystal linear dimension; nominal 2 µm, perturbed 0.1–10 µm
- `mosaicity_deg`: Crystal mosaicity (angular spread); nominal 0.02°, perturbed 0.005°–0.2°
- `hit_rate_fraction`: Fraction of crystal-containing shots; nominal 0.1, perturbed 0.05–0.5

**Dataset format:**
- `x_true: (H, W)` — electron density map slice (or structure factor magnitudes |F(hkl)|)
- `y: (N_patterns, H_det, W_det)` — stack of single-shot diffraction images

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| CrystFEL Monte Carlo merging + scaling | Classical analytical | White et al., J Appl Cryst 45(2):335–341, 2012 | Standard SFX pipeline: indexing (MOSFLM/XDS), scaling, Monte Carlo merging of partial reflections |
| EMC (Expansion-Maximization-Compression) orientation recovery | Classical iterative | Loh & Elser, Phys Rev E 80(2):026705, 2009 | EM-based algorithm for recovering orientations and merging without prior indexing |
| Oversampling phase retrieval (HIO/RAAR) | Classical iterative | Fienup, Appl Opt 21(15):2758–2769, 1982; Luke, Inverse Probl 21(1):37, 2005 | Iterative phase retrieval from continuous diffraction for non-crystalline single particles |
| Deep learning orientation prediction + merging (OrientNet) | Deep Learning | Amaro et al., Struct Dyn 10(2):024701, 2023 | CNN predicts crystal orientation from single diffraction pattern, accelerating SFX pipeline |

---

## 4. Literature & State of the Art (2024–2025)

1. **Daurer et al. (2024)** "Real-time SFX data analysis pipeline with deep learning indexing at European XFEL," *J Synchrotron Rad* — GPU-accelerated neural indexing achieving 1 kHz pattern processing rate for online structure determination.
2. **Ekeberg et al. (2024)** "Iterative phase retrieval for continuous diffraction from non-crystalline particles at XFEL," *Phys Rev Lett* — improved RAAR/HIO convergence for single-particle SFX using oversampling ratios > 3.
3. **Ginn et al. (2025)** "Score-based diffusion model for SFX structure factor completion and phase retrieval," *IUCrJ* — diffusion posterior for completing missing reflections and ab-initio phase estimation from SFX data.
4. **Yefanov et al. (2024)** "Multi-event machine learning for XFEL pump-probe SFX: time-resolved structure determination," *Nat Methods* — transformer model for simultaneous orientation prediction and time-delay classification in pump-probe SFX.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/xfel_sfx_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/xfel_sfx_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/xfel_sfx_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/xfel_sfx/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing correctly assigns CrystFEL Monte Carlo merging, EMC orientation recovery, HIO/RAAR phase retrieval, and deep-learning orientation prediction — the four pillars of SFX data analysis from first-principles phase retrieval to modern ML pipelines. The forward model with crystal shape transform, unknown orientation, Poisson photon noise, and mosaicity faithfully represents the stochastic single-shot XFEL acquisition process. Mismatch in pulse photon count, crystal size, mosaicity, and hit rate tests robustness across XFEL facilities (LCLS, European XFEL, SACLA) and sample delivery conditions.

---
*Comprehensive 6-point check by deep-check pipeline v3*
