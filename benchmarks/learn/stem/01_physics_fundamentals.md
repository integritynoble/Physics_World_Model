# 01 — Physics Fundamentals: Scanning Transmission Electron Microscopy (STEM)

## 1. Overview

Scanning transmission electron microscopy rasters a converged electron probe across a thin specimen. Different angular ranges of scattered electrons are collected by annular detectors: bright-field (BF), annular bright-field (ABF), annular dark-field (ADF), and high-angle annular dark-field (HAADF). HAADF-STEM provides Z-contrast imaging where intensity scales approximately as Z^1.7. The forward model is y = detector_response * probe_convolution(x) + noise.

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
| Electron Gun + Probe Forming | source | identity | 1.0 | beam_current_fluctuation, scan_noise |
| Thin Specimen | interaction | scattering | 0.85 | radiation_damage |
| HAADF Detector | detector | integration | 0.6 | shot_poisson, electronic_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512] |
| Measurement shape (y) | [512, 512] |
| Forward model type | linear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | cold_field_emission |
| Accelerating Voltage Kv | 200.0 |
| Beam Current Pa | 20.0 |
| Probe Size Angstrom | 0.8 |
| Convergence Angle Mrad | 21.0 |
| Thickness Nm | 20.0 |
| Z Contrast Exponent | 1.7 |
| Inner Angle Mrad | 68.0 |
| Outer Angle Mrad | 200.0 |
| Collection Efficiency | 0.6 |
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
