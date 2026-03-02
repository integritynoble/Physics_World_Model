# 01 — Physics Fundamentals: Cathodoluminescence (CL) Imaging

## 1. Overview

Cathodoluminescence (CL) Imaging imaging modality with DAG: M --> R --> D.

**Category**: Scientific Instrumentation
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

0 – 0 nm

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
| Coded Mask | mask | modulation | 0.5 | — |
| Detector | detector | integration | 0.8 | poisson, gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [64, 64] |
| Measurement shape (y) | [64, 64] |
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Density | 0.5 |


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
