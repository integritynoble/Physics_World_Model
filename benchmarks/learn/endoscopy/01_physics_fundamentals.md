# 01 — Physics Fundamentals: Fiber Bundle Endoscopy

## 1. Overview

Fiber bundle endoscopy transmits an image through a coherent fiber bundle consisting of thousands of individual cores. Each core samples one spatial location, introducing a honeycomb pattern and coupling losses. Specular highlights from the tissue surface add nonlinear artifacts. The forward model is y = Poisson(alpha * S(F * x)) + N(0, sigma^2), where F is the fiber bundle sampling operator and S adds specular reflections.

**Category**: Medical Imaging
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

400 – 700 nm

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
| White Light Source | source | identity | 1.0 | — |
| Coherent Fiber Bundle | fiber | sampling | 0.3 | cross_talk, fixed_pattern |
| CCD Sensor | detector | integration | 0.75 | shot_poisson, read_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512] |
| Measurement shape (y) | [512, 512] |
| Forward model type | nonlinear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | LED |
| Power Mw | 200 |
| Spectral Range Nm | [400, 700] |
| N Cores | 30000 |
| Core Pitch Um | 10 |
| Core Diameter Um | 8 |
| Bundle Diameter Mm | 3 |
| Na | 0.3 |
| Pixel Size Um | 5.0 |
| Read Noise E | 3.0 |
| Quantum Efficiency | 0.75 |
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
