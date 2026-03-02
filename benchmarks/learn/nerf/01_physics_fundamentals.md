# 01 — Physics Fundamentals: Neural Radiance Fields (NeRF)

## 1. Overview

Neural Radiance Fields represent a 3D scene as a continuous volumetric function mapping 3D coordinates and viewing direction to color and density, parameterized by a neural network. Multi-view posed images supervise training. The forward model integrates color and density along camera rays using volume rendering: C(r) = integral T(t) * sigma(t) * c(t) dt where T(t) is the transmittance. NeRF, Instant-NGP, and Mip-NeRF 360 are standard architectures.

**Category**: Neural Rendering
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

380 – 780 nm

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
| Multi-View Camera Rig | source | identity | 1.0 | — |
| Camera Lens | lens | projection | 0.9 | aberration |
| 3D Scene Volume | medium | nonlinear | 1.0 | — |
| Volume Rendering Integrator | modulator | integration | 1.0 | — |
| RGB Image Sensor | detector | sampling | 0.85 | shot_poisson, read_gaussian, quantization |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [128, 128, 64] |
| Measurement shape (y) | [10, 128, 128, 3] |
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| N Views | 10 |
| Image Resolution | [128, 128] |
| Field Of View Deg | 60 |
| Angular Coverage Deg | 360 |
| Focal Length Mm | 50 |
| Aperture | f/2.0 |
| Distortion Model | pinhole |
| Volume Resolution | [128, 128, 64] |
| Density Activation | softplus |
| Color Activation | sigmoid |
| Positional Encoding Levels | 10 |
| N Samples Per Ray | 64 |
| N Importance Samples | 128 |
| Near Plane | 0.1 |
| Far Plane | 10.0 |
| Stratified Sampling | True |
| Bit Depth | 8 |
| Color Channels | 3 |
| Gamma | 2.2 |


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
