# 01 — Physics Fundamentals: Electron Tomography

## 1. Overview

Electron tomography acquires a series of TEM images at different tilt angles (typically +/- 60-70 degrees). The tilt series is used to reconstruct a 3D volume of the specimen via backprojection or iterative methods (SIRT, ART, ADMM). The forward model for each projection is y_i = project(volume, angle_i) + noise. The missing wedge due to limited tilt range causes anisotropic resolution in the reconstruction.

**Category**: Electron Microscopy
**Carrier**: Electron

---

## 2. Electron Physics

Electron imaging uses accelerated electron beams (1-300 keV) whose de Broglie wavelength (λ = h/√(2meV)) is orders of magnitude shorter than visible light, enabling atomic-resolution imaging. Electrons interact with matter via elastic and inelastic scattering, and the contrast transfer function (CTF) describes phase and amplitude modulation.

### Key Concepts

- de Broglie wavelength: λ = h / √(2m_e eV)
- Contrast transfer function (CTF)
- Elastic vs inelastic scattering
- Aberrations: spherical (Cs), chromatic (Cc)
- Electron dose and radiation damage

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
I(r) = |F⁻¹{CTF(q) · F{V(r)}}|² + noise
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| Electron Gun | source | identity | 1.0 | beam_current_fluctuation |
| Thin Specimen on Tilt Stage | interaction | projection | 0.9 | radiation_damage, stage_drift |
| CCD/CMOS Detector | detector | integration | 0.7 | shot_poisson, read_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [64, 64, 64] |
| Measurement shape (y) | [64, 64] |
| Forward model type | linear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | field_emission |
| Accelerating Voltage Kv | 300.0 |
| Beam Current Na | 0.1 |
| Sigma | 0.00729 |
| Thickness Nm | 50.0 |
| Tilt Range Deg | [-60, 60] |
| Tilt Increment Deg | 2.0 |
| Pixel Size Um | 14.0 |
| Quantum Efficiency | 0.7 |
| Gain | 1.0 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Electron |
| Primary contrast | Determined by electron-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
