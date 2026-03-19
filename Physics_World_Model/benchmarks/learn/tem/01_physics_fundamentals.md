# 01 — Physics Fundamentals: Transmission Electron Microscopy (TEM)

## 1. Overview

Transmission electron microscopy passes a broad or focused electron beam through a thin specimen (<100 nm). The transmitted electrons form an image via the objective lens, modulated by the contrast transfer function (CTF). Phase contrast dominates for biological/organic specimens. The forward model involves thin-object phase approximation followed by CTF modulation: y = |IFFT(FFT(T(x) * beam) * CTF(q))|^2 + noise, where T(x) is the transmission function.

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
| Thin Specimen | interaction | phase_modulation | 0.95 | radiation_damage |
| Objective Lens + CTF | lens | ctf_transfer | 0.8 | aberration |
| CCD/CMOS Detector | detector | integration | 0.8 | shot_poisson, read_gaussian |

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
| Accelerating Voltage Kv | 200.0 |
| Beam Current Na | 0.1 |
| Wavelength Pm | 2.51 |
| Sigma | 0.00729 |
| Thickness Nm | 10.0 |
| Defocus Nm | -50.0 |
| Cs Mm | 1.0 |
| Pixel Size Um | 14.0 |
| Quantum Efficiency | 0.8 |
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
