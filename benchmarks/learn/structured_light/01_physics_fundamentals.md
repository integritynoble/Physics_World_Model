# 01 — Physics Fundamentals: Structured-Light Depth Camera

## 1. Overview

Structured-light depth cameras project known patterns (sinusoidal fringes or random dots) onto the scene and capture them with a camera offset from the projector. Depth is recovered from the phase deformation of the observed fringes via triangulation. The forward model is y = O(P(x)) + n, where P is the fringe projection and O models the camera optics with depth-dependent defocus.

**Category**: Depth Imaging
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

450 – 850 nm

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
| Pattern Projector | source | modulation | 0.9 | speckle |
| Camera (Receiver) | detector | integration | 0.8 | shot_poisson, read_gaussian |

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
| Type | DLP_projector |
| N Patterns | 4 |
| Fringe Freq Cycles | 8 |
| Resolution | [1024, 768] |
| Focal Length Mm | 50 |
| Aperture | 2.8 |
| Pixel Size Um | 5.0 |
| Read Noise E | 2.0 |
| Quantum Efficiency | 0.8 |
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
