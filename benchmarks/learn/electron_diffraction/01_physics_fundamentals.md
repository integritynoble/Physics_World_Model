# 01 — Physics Fundamentals: 4D-STEM Electron Diffraction

## 1. Overview

4D-STEM collects a 2D convergent beam electron diffraction (CBED) pattern at each probe position in a 2D raster scan, producing a 4D dataset. The forward model involves far-field diffraction: y = |F{t(r) * P(r)}|^2.

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

### Wavelength / Energy Range

0.002 – 0.004 nm

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
| Electron Beam (200 kV) | source | identity | 1.0 | — |
| Pixelated STEM Detector | detector | integration | 0.8 | shot_poisson, read_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [128, 128] |
| Measurement shape (y) | [128, 128] |
| Forward model type | nonlinear_operator |
| Category module | electron_ctf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Accelerating Voltage Kv | 200 |
| Convergence Semiangle Mrad | 20 |
| N Pixels | [256, 256] |
| Frame Rate Fps | 1000 |


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
