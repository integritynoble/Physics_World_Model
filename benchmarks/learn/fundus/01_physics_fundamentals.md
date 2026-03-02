# 01 — Physics Fundamentals: Fundus Camera

## 1. Overview

Fundus photography captures an image of the retina through the pupil using a low-magnification optical system with flash or continuous illumination. The retinal image is blurred by the eye's optics (PSF depends on pupil diameter and refractive error) and degraded by Poisson-Gaussian noise from the detector. The forward model is y = PSF ** x + n.

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

500 – 700 nm

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
| Flash / LED Illumination | source | identity | 1.0 | — |
| Fundus Optics (45 deg FOV) | lens | convolution | 0.7 | aberration |
| CMOS Detector | detector | integration | 0.8 | shot_poisson, read_gaussian |

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
| Type | LED |
| Power Mw | 100 |
| Spectral Width Nm | 80 |
| Fov Deg | 45 |
| Magnification | 2.5 |
| Pupil Diameter Mm | 4.0 |
| Psf Sigma Px | 1.5 |
| Pixel Size Um | 5.0 |
| Read Noise E | 2.0 |
| Quantum Efficiency | 0.8 |
| Bit Depth | 14 |


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
