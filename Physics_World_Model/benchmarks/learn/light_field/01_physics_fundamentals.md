# 01 — Physics Fundamentals: Light Field Imaging

## 1. Overview

Light field imaging captures the full 4D light field L(x,y,u,v) encoding both spatial (x,y) and angular (u,v) information about the scene. A microlens array placed at the sensor focal plane redirects rays from different directions to different pixels, sampling the angular domain. The 4D light field is compressed into a single 2D sensor image. Digital refocusing, depth estimation, and view synthesis are enabled by computational reconstruction via shift-and-sum or learning-based methods.

**Category**: Computational Optics
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
| Scene Illumination | source | identity | 1.0 | — |
| Main Lens | lens | projection | 0.9 | aberration |
| Microlens Array | modulator | sampling | 0.75 | alignment, fixed_pattern |
| CMOS Image Sensor | detector | integration | 0.8 | shot_poisson, read_gaussian, fixed_pattern, quantization |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512, 9, 9] |
| Measurement shape (y) | [512, 512] |
| Forward model type | linear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | ambient |
| Spectral Range Nm | [400, 700] |
| Focal Length Mm | 50 |
| F Number | 2.0 |
| Distortion Model | pinhole |
| Microlens Pitch Um | 125 |
| Microlens Focal Length Mm | 0.5 |
| N Angular Samples | [9, 9] |
| Array Size | [512, 512] |
| Fill Factor | 0.98 |
| Pixel Size Um | 1.4 |
| Read Noise E | 2.0 |
| Quantum Efficiency | 0.8 |
| Full Well E | 10000 |
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
