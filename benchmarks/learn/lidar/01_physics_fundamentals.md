# 01 — Physics Fundamentals: LiDAR Scanner

## 1. Overview

LiDAR (Light Detection And Ranging) measures distance by scanning a pulsed laser beam across the scene and recording the time of flight of reflected photons. A scan trajectory (raster or Lissajous) samples the scene sparsely. SPAD or APD detectors measure photon arrival times. The forward model is y = Poisson(alpha * T(S(x))) + n, where S is the scan trajectory sampling and T is ToF gating.

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

905 – 1550 nm

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
| Pulsed Laser Source | source | identity | 1.0 | — |
| Scanning Mirror | scanner | sampling | 0.9 | jitter |
| SPAD Detector | detector | integration | 0.25 | shot_poisson, dark_count |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256] |
| Measurement shape (y) | [256, 256] |
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | pulsed_laser |
| Wavelength Nm | 905 |
| Pulse Energy Uj | 10 |
| Rep Rate Khz | 100 |
| Scan Type | raster |
| Fov Deg | 60 |
| Angular Resolution Deg | 0.1 |
| N Bins | 64 |
| Bin Width Ns | 0.5 |
| Dead Time Ns | 20 |
| Qe | 0.25 |


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
