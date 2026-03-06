# Comprehensive 6-Point Check — Atom Probe Tomography (APT)

**URL:** https://pwm.platformai.org/benchmark/atom_probe
**Check Date:** 2026-03-06
**Status:** PASS (with noted limitations)

---

## 1. Physics & Forward Model

**Modality:** Atom Probe Tomography (APT)

**Physical principle:** Atom probe tomography achieves atomic-resolution 3D elemental mapping by applying high-voltage nanosecond pulses to a field-ion specimen tip (~50 nm apex radius). Individual atoms are field-evaporated one at a time and accelerated to a position-sensitive time-of-flight (ToF) detector. The detector (x,y) hit position and flight time t encode the atom's original 3D position and chemical identity (mass-to-charge ratio m/z = 2eV t²/L² where L is flight path). The spatial reconstruction requires inversion of the complex electrostatic trajectory from tip to detector.

**Forward model:**
```
Mass spectrum:
  (m/z)_i = 2 e V_dc t_i^2 / L^2   [simplified linear ToF model]

Spatial reconstruction (Bas protocol):
  x_atom = ξ X_det / (R_tip * Ω_f)
  y_atom = ξ Y_det / (R_tip * Ω_f)
  z_atom = Σ_i d_z / N_evap         [depth from evaporation order]

where:
  V_dc    — DC standing voltage (kV)
  t_i     — flight time (ns)
  L       — flight path length (mm)
  R_tip   — tip radius (nm, increases during analysis)
  Ω_f     — field factor
  ξ       — image compression factor
```

**Inverse problem:** Recover the 3D atomic composition map (element identity + position) from the detector hit sequence (X_det, Y_det, t_i), given imperfect knowledge of flight path, tip radius evolution, and detector efficiency.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** S(field evaporation) → D(position-sensitive ToF detector)

**Key mismatch parameters:**
- `flight_path_error` (f_p): flight path length L uncertainty; nominal 0.0 mm deviation, perturbed 0.1 mm
- `voltage_calibration` (v_c): DC voltage scaling factor; nominal 1.0, perturbed 1.004
- `detection_efficiency` (d_e): fraction of evaporated atoms detected; nominal 0.60, perturbed 0.62
- `tip_radius_error` (t_r): uncertainty in apex radius estimate; nominal 0.0 nm, perturbed 1.0 nm

**Dataset format:**
- `x_true: (H, W)` — 2D elemental map projection (ground truth composition image)
- `y: (N_hits, 3)` — detector hit list: (X_det, Y_det, t_flight) per atom
- `H_ideal: (H*W, N_hits)` — ideal spatial reconstruction operator (trajectory mapping)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Deconv | Classical | Bas et al., Appl. Surf. Sci. 1995 | Bas protocol spatial reconstruction; standard APT trajectory algorithm |
| Calibration-Lookup | Classical | Geiser et al., Microsc. Microanal. 2007 | Empirical calibration lookup for voltage-trajectory correction |
| Peak Fitting | Classical | — | Gaussian/Lorentzian peak fitting for mass spectrum deconvolution |
| PnP-BM3D | Plug-and-Play | Danielyan et al., IEEE TIP 2012 | BM3D denoising prior applied to reconstructed composition images |
| ResNet-Calib | Deep Learning | — | ResNet-based calibration correction for trajectory artefacts |
| CalibFormer | Transformer | — | Transformer architecture for instrument calibration correction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Deep learning for APT reconstruction** (Wei et al., Ultramicroscopy 2019 / extended 2024): CNN trained on simulated APT datasets for artefact correction in reconstructed atom maps; handles local magnification artefacts at precipitate-matrix interfaces.
2. **Physics-informed neural networks for trajectory correction** (2024): PINN incorporating electrostatic trajectory equations; improves spatial resolution near grain boundaries.
3. **Mass spectrum deconvolution via deep learning** (2023–2024): Attention-based network for overlapping isotope peak separation in complex alloy APT spectra.
4. **Transfer learning across APT instruments** (2025): Domain adaptation between LEAP and wide-angle instruments using shared latent representations; reduces calibration effort for new specimen geometries.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/atom_probe_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/atom_probe_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/atom_probe_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/atom_probe/`.

---

## 6. Comprehensive Assessment

**Status:** PASS (with noted limitations)

Algorithm routing uses the `scientific_instrumentation` category pool (11 methods: Deconv, Calibration-Lookup, Peak Fitting, PnP-BM3D, PnP-NLM, ResNet-Calib, Instrument-CNN, CalibFormer, MassSpecFormer, DiffusionInstrumentation, ScoreInstrumentation). Deconv (Bas protocol) and Calibration-Lookup (Geiser protocol) are domain-appropriate classical methods. The pool is a catch-all for ion/electron instrument modalities; APT-specific deep learning citations (ResNet-Calib, CalibFormer) remain generic but represent plausible algorithm archetypes. The four mismatch parameters (flight path error, voltage calibration, detection efficiency, tip radius) are physically grounded in APT instrument practice.

---
*Comprehensive 6-point check by deep-check pipeline v3*
