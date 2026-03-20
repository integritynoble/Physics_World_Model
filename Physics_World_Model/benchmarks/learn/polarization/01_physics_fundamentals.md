# 01 — Physics Fundamentals: Polarization Microscopy

## 1. Overview

Polarization microscopy measures the birefringence and anisotropy of transparent specimens by analyzing changes in the polarization state of transmitted light. The forward model involves the Mueller matrix or Jones calculus of the optical path, with retardance and orientation as the target parameters. Applications include mineralogy, liquid crystals, biological tissue organization, and stress analysis.

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
| Polarized Light Source | source | identity | 1.0 | — |
| Objective Lens (20x / 0.5 NA) | lens | convolution | 0.8 | aberration |
| Analyzer + sCMOS Detector | detector | integration | 0.85 | shot_poisson, read_gaussian |

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
| Wavelength Nm | 550 |
| Polarizer Type | linear |
| Magnification | 20 |
| Numerical Aperture | 0.5 |
| Psf Sigma Px | 1.8 |
| Immersion | air |
| Pixel Size Um | 6.5 |
| Read Noise E | 1.6 |
| Quantum Efficiency | 0.85 |
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
