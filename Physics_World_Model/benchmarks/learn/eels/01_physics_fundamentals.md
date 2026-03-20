# 01 — Physics Fundamentals: Electron Energy Loss Spectroscopy (EELS)

## 1. Overview

EELS measures the energy lost by transmitted electrons after inelastic scattering in a thin specimen. The forward model is a convolution of the specimen loss function with the zero-loss peak.

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
| Electron Beam (200 kV STEM) | source | identity | 1.0 | — |
| Magnetic Sector Spectrometer | detector | integration | 0.7 | shot_poisson, read_gaussian, dark_current |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256] |
| Measurement shape (y) | [1024] |
| Forward model type | linear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Accelerating Voltage Kv | 200 |
| Energy Resolution Ev | 0.3 |
| N Channels | 2048 |
| Dispersion Ev Per Channel | 0.5 |


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
