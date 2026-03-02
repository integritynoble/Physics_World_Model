# 01 — Physics Fundamentals: Diffuse Optical Tomography (DOT)

## 1. Overview

Diffuse Optical Tomography (DOT) reconstructs spatial maps of optical absorption (mu_a) and reduced scattering (mu_s') coefficients inside biological tissue from boundary measurements of near-infrared light. NIR laser sources illuminate the tissue surface at multiple positions, and photodetectors on the boundary measure transmitted and reflected light intensity. The forward model is governed by the diffusion equation: y = J(mu_a, mu_s') * x where J is the Jacobian (sensitivity matrix) mapping internal optical property changes to boundary measurement changes. Born approximation or L-BFGS with total variation regularization are standard reconstruction methods.

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

750 – 850 nm

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
| NIR Laser Sources | source | identity | 1.0 | — |
| Tissue Medium | medium | projection | 0.01 | shot_poisson |
| Boundary Photodetectors | detector | integration | 0.7 | shot_poisson, read_gaussian, thermal |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [64, 64, 64] |
| Measurement shape (y) | [256] |
| Forward model type | linear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | APD |
| Wavelengths Nm | [750, 800, 850] |
| Power Mw | 20 |
| N Sources | 16 |
| Modulation | continuous_wave |
| Background Mu A Per Mm | 0.01 |
| Background Mu S Prime Per Mm | 1.0 |
| Refractive Index | 1.37 |
| Diffusion Model | frequency_domain |
| Tissue Thickness Mm | 60 |
| N Detectors | 16 |
| Responsivity A Per W | 0.5 |
| Nep Pw Per Sqrt Hz | 10.0 |
| Bandwidth Mhz | 200 |
| Bit Depth | 16 |


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
