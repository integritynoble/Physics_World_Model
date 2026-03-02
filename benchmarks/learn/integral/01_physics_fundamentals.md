# 01 — Physics Fundamentals: Integral Photography

## 1. Overview

Integral photography (also known as integral imaging) captures and reproduces three-dimensional scenes using a microlens array. Each microlens records an elemental image corresponding to a slightly different perspective of the scene, encoding depth information. The forward model is I(x,y) = integral of L(x,y,u,v) * T(u,v) dudv where L is the 4D light field, T is the microlens transmission function, and the integral is over the angular domain. Depth estimation from disparity analysis and depth-image-based rendering (DIBR) enable 3D visualization and computational refocusing from the captured elemental image array.

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
| Microlens Array | modulator | sampling | 0.72 | alignment, fixed_pattern |
| CMOS Image Sensor | detector | integration | 0.8 | shot_poisson, read_gaussian, fixed_pattern, quantization |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512, 64] |
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
| F Number | 2.8 |
| Distortion Model | pinhole |
| Microlens Pitch Um | 150 |
| Microlens Focal Length Mm | 0.7 |
| N Depth Planes | 64 |
| Array Size | [512, 512] |
| Fill Factor | 0.95 |
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
