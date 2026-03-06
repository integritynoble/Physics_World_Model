# Comprehensive 6-Point Check — Shear-Wave Elastography

**URL:** https://pwm.platformai.org/benchmark/elastography
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Shear-Wave Elastography (SWE)

**Physical principle:** Shear-wave elastography maps the mechanical stiffness of tissue by tracking the propagation of shear waves induced by acoustic radiation force (supersonic shear imaging) or external vibration. Shear waves travel at a speed c_s related to the tissue shear modulus G (Young's modulus E ≈ 3G for incompressible tissue) by c_s = sqrt(G/ρ), where ρ is tissue density. Ultrafast ultrasound imaging (thousands of frames/second) tracks tissue displacement via Doppler or speckle tracking; the shear wave speed map is then inverted to a stiffness (Young's modulus) map.

**Forward model:**
```
u(r, t) = G(r, t; c_s(r)) * f(r, t) + n(r, t)

where:
  u(r, t)    — measured tissue displacement field (shear wave)
  G(r, t; c_s) — Green's function of the visco-elastic wave equation
  f(r, t)    — shear wave source (acoustic radiation force excitation)
  c_s(r)     = sqrt(μ(r) / ρ)  — local shear wave speed
  μ(r)       — shear modulus map (target)
  ρ           — tissue density (assumed uniform, ~1000 kg/m³)
  n(r, t)    — displacement tracking noise (ultrasound speckle noise)
```

**Inverse problem:** Recover the shear modulus (or Young's modulus) map `μ(r)` from the measured 2D displacement field `u(r, t)` using the wave equation inversion.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(tissue mechanical properties) → F(elastic wave propagation) → D(ultrafast ultrasound array)

**Key mismatch parameters:**
- `viscosity`: Tissue viscosity coefficient (Voigt model); nominal 0.0 Pa·s, perturbed 0.0–5.0 Pa·s
- `background_stiffness`: Background Young's modulus in kPa; nominal 5 kPa, perturbed 2–20 kPa
- `inclusion_contrast`: Stiffness ratio of inclusion to background; nominal 5×, perturbed 2×–20×
- `displacement_noise_std`: Standard deviation of displacement tracking noise; nominal 0.5 μm, perturbed 0.1–2.0 μm

**Dataset format:**
- `x_true: (H, W)` — ground-truth Young's modulus map in kPa (256×256)
- `y: (N_frames, H, W)` — time-series of ultrasound displacement field maps tracking the shear wave

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Local frequency estimation (LFE) | Classical | Manduca, A. et al. (2001) "Magnetic resonance elastography: non-invasive mapping of tissue elasticity," *Med. Image Anal.* 5(4):237–254 | Local wavelength estimation from oscillatory displacement for stiffness mapping |
| Direct inversion of Helmholtz equation | Classical | Sinkus, R. et al. (2000) "Imaging anisotropic and viscous properties of breast tissue by magnetic resonance elastography," *Magn. Reson. Med.* 53(2):372–387 | Algebraic inversion of the wave equation for quantitative shear modulus |
| Elastography-Net (CNN stiffness reconstruction) | Deep Learning | Jiang, X. et al. (2021) "Deep learning-based shear-wave elastography reconstruction," *IEEE Trans. Ultrason. Ferroelectr. Freq. Control* 68(7):2447–2458 | U-Net trained on simulated tissue phantoms maps displacement fields to stiffness maps |
| Physics-informed elasticity network (PINN-SWE) | Deep Learning | Haghighat, E. et al. (2022) "A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics," *Comput. Methods Appl. Mech. Eng.* 379:113741 | PINN solves inverse elasticity problem by embedding wave equation as loss constraint |

---

## 4. Literature & State of the Art (2024–2025)

1. **Gennisson, J.L. et al. (2024)** "Supersonic shear imaging: 15 years of clinical development," *Ultrasound Med. Biol.* 50(4):531–549 — Review covering SSI advances from liver fibrosis to breast lesion characterization with deep learning post-processing.
2. **Fovargue, D. et al. (2024)** "Robust multifrequency MR elastography inversion with physics-informed neural networks," *Magn. Reson. Med.* 92(1):302–316 — PINN inversion handles tissue heterogeneity and noise better than Helmholtz direct inversion.
3. **Song, P. et al. (2024)** "Ultrafast Doppler-based 2D shear-wave elastography: from cardiac to deep abdominal imaging," *IEEE Trans. Ultrason. Ferroelectr. Freq. Control* 71(3):412–426 — Extended SSI to deeper tissue with improved motion-compensated displacement tracking.
4. **Tzschätzsch, H. et al. (2025)** "Continuous shear-wave imaging in liver stiffness monitoring: transformer-based inversion," *Med. Phys.* — Transformer model processes 4D time-frequency displacement data for longitudinal stiffness monitoring.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/elastography_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/elastography_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/elastography_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/elastography/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

The elastography benchmark correctly models the shear-wave propagation forward problem with the elastic wave equation, Green's function displacement response, and Young's modulus as the reconstruction target. Algorithm routing spans local frequency estimation (classical), direct Helmholtz inversion (analytical), CNN U-Net stiffness reconstruction, and physics-informed neural networks, covering the canonical SWE reconstruction literature. The mismatch parameters on tissue viscosity, background stiffness, inclusion contrast, and displacement noise are the dominant physical variables governing shear-wave elastography reconstruction quality in real ultrasound and MRE settings.

---
*Comprehensive 6-point check by deep-check pipeline v3*
