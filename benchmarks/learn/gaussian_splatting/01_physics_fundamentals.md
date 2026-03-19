# 01 — Physics Fundamentals: 3D Gaussian Splatting (3DGS)

## 1. Overview

3D Gaussian Splatting represents a scene as a collection of oriented 3D Gaussians, each with position, covariance, opacity, and spherical harmonics color coefficients. Rendering projects each Gaussian onto the image plane as a 2D Gaussian via the EWA splatting equation, then alpha-composites them front to back. The differentiable rasterizer enables end-to-end optimization of Gaussian parameters from multi-view images with real-time rendering capability.

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
| 3D Gaussian Point Cloud | medium | nonlinear | 1.0 | — |
| Differentiable Tile-Based Rasterizer | modulator | projection | 1.0 | — |
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
| Focal Length Mm | 50 |
| Distortion Model | pinhole |
| N Gaussians Initial | 100000 |
| N Gaussians Max | 5000000 |
| Sh Degree | 3 |
| Opacity Activation | sigmoid |
| Scale Activation | exp |
| Tile Size Px | 16 |
| Sort Method | radix |
| Alpha Compositing | front_to_back |
| Splat Sigma | 2.0 |
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
