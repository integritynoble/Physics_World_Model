# 01 — Physics Fundamentals: Widefield Fluorescence Microscopy

## 1. Overview

Standard widefield epi-fluorescence microscopy where the entire field of view is illuminated simultaneously and the image is formed by convolution of the specimen fluorescence distribution with the system point spread function. Out-of-focus blur is the primary degradation. Deconvolution via Richardson-Lucy or learned priors (CARE) restores resolution.

**Category**: Microscopy
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
| Mercury / LED Source | source | identity | 1.0 | — |
| Excitation Filter | filter | modulation | 0.85 | — |
| Objective Lens (20x / 0.75 NA) | lens | convolution | 0.8 | aberration |
| Emission Filter | filter | modulation | 0.9 | — |
| sCMOS Detector | detector | integration | 0.82 | shot_poisson, read_gaussian, fixed_pattern |

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
| Power Mw | 50 |
| Spectral Width Nm | 30 |
| Center Nm | 525 |
| Bandwidth Nm | 50 |
| Magnification | 20 |
| Numerical Aperture | 0.75 |
| Psf Sigma Px | 2.0 |
| Immersion | air |
| Pixel Size Um | 6.5 |
| Read Noise E | 1.6 |
| Quantum Efficiency | 0.82 |
| Full Well E | 30000 |
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
