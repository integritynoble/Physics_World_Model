# 01 — Physics Fundamentals: Generic Matrix Sensing

## 1. Overview

A generic linear inverse problem defined by an explicit measurement matrix A. The forward model is y = A*x where A is an arbitrary M-by-N matrix. This modality serves as a universal fallback for any linear sensing system that can be expressed in matrix form, including random projections, subsampling, and learned measurement operators. FISTA-TV or LISTA are the default solvers.

**Category**: Compressive Imaging
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

0 – 0 nm

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
| Signal Source | source | identity | 1.0 | — |
| Linear Measurement Operator | modulator | projection | 1.0 | — |
| Additive Noise Channel | medium | identity | 1.0 | read_gaussian |
| Digital Readout | detector | sampling | 1.0 | quantization |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [64, 64] |
| Measurement shape (y) | [614] |
| Forward model type | explicit_matrix |
| Category module | compressive_mask |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | ideal_adc |
| Matrix Type | gaussian_random |
| M | 614 |
| N | 4096 |
| Sampling Rate | 0.15 |
| Noise Sigma | 0.01 |
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
