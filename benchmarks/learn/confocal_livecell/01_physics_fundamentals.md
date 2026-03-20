# 01 — Physics Fundamentals: Confocal Live-Cell Microscopy

## 1. Overview

Laser scanning confocal microscopy for live-cell imaging. A focused laser scans the specimen point by point, and a pinhole rejects out-of-focus light. The image formation is modelled as convolution with the confocal PSF (product of excitation and detection PSFs). Fast acquisition rates for live cells often sacrifice SNR.

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

400 – 650 nm

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
| Laser Source (488 nm) | source | identity | 1.0 | — |
| Scanning Mirrors (Galvo) | modulator | sampling | 0.92 | alignment |
| Objective Lens (60x / 1.4 NA Oil) | lens | convolution | 0.75 | aberration |
| Confocal Pinhole (1 AU) | filter | modulation | 0.7 | alignment |
| PMT Detector | detector | integration | 0.25 | shot_poisson, read_gaussian, thermal |

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
| Type | PMT |
| Wavelength Nm | 488 |
| Power Mw | 5 |
| Line Width Nm | 0.1 |
| Scan Rate Hz | 8000 |
| Pixels Per Line | 512 |
| Bidirectional | True |
| Magnification | 60 |
| Numerical Aperture | 1.4 |
| Psf Sigma Lateral Px | 1.0 |
| Psf Sigma Axial Px | 3.5 |
| Immersion | oil |
| Pinhole Au | 1.0 |
| Pinhole Diameter Um | 60 |
| Quantum Efficiency | 0.25 |
| Dark Current Cps | 50 |
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
