# 01 — Physics Fundamentals: Two-Photon / Multiphoton Microscopy

## 1. Overview

Two-photon excitation microscopy uses near-infrared femtosecond pulsed laser illumination. Fluorescence is generated only at the focal volume where the photon flux is high enough for simultaneous two-photon absorption (intensity squared dependence). This intrinsic optical sectioning enables deep tissue imaging with reduced photobleaching and phototoxicity compared to single-photon confocal.

**Category**: Microscopy
**Carrier**: Photon

---

## 2. Photon Physics

Photon-based imaging uses visible, near-infrared, or ultraviolet light. The image formation is typically modelled as convolution with a point spread function (PSF) determined by the optical system's numerical aperture and wavelength. Key degradations include diffraction blur, aberrations, and photon shot noise (Poisson statistics).

### Key Concepts

- Diffraction limit: d = 0.61 λ / NA
- Point spread function (PSF) and optical transfer function (OTF)
- Numerical aperture (NA) and resolution
- Shot noise (Poisson) and read noise (Gaussian)
- Fluorescence: excitation/emission Stokes shift

### Wavelength / Energy Range

700 – 1100 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
y = PSF ⊛ x + noise  (⊛ = convolution)
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| Ti:Sapphire Pulsed Laser | source | identity | 1.0 | — |
| Objective Lens (25x / 1.05 NA Water) | lens | convolution | 0.75 | aberration |
| PMT / GaAsP Detector | detector | integration | 0.45 | shot_poisson, read_gaussian |

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
| Type | GaAsP_PMT |
| Wavelength Nm | 920 |
| Pulse Width Fs | 100 |
| Rep Rate Mhz | 80 |
| Power Mw | 20 |
| Magnification | 25 |
| Numerical Aperture | 1.05 |
| Psf Sigma Px | 1.5 |
| Immersion | water |
| Quantum Efficiency | 0.45 |
| Dark Current Cps | 100 |
| Bit Depth | 12 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Photon |
| Primary contrast | Determined by photon-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
