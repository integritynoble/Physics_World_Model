# 01 — Physics Fundamentals: Scanning Electron Microscopy (SEM)

## 1. Overview

Scanning electron microscopy rasters a focused electron beam across the specimen surface. Secondary electrons (SE) and backscattered electrons (BSE) emitted from the interaction volume are collected by dedicated detectors. SE signal is sensitive to surface topography, while BSE signal encodes compositional (Z-contrast) information. The forward model is y = detector(yield(beam, material_map)) + noise.

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
| Specimen Interaction | interaction | yield_model | 0.1 | charging, contamination |
| Everhart-Thornley Detector | detector | integration | 0.5 | shot_poisson, electronic_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512] |
| Measurement shape (y) | [512, 512] |
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | field_emission |
| Accelerating Voltage Kv | 15.0 |
| Beam Current Na | 1.0 |
| Probe Size Nm | 2.0 |
| Yield Type | SE |
| Yield Coeff | 0.1 |
| Collection Efficiency | 0.5 |
| Gain | 100.0 |
| Bias Voltage V | 300.0 |


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
