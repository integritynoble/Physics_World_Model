# 01 — Physics Fundamentals: PALM/STORM Single-Molecule Localization

## 1. Overview

Photo-Activated Localization Microscopy (PALM) and Stochastic Optical Reconstruction Microscopy (STORM) achieve nanometer-scale resolution by imaging sparse subsets of stochastically activated fluorescent molecules over thousands of frames. Individual molecule positions are localized with sub-pixel precision from their diffraction-limited images, then combined into a super-resolved reconstruction.

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

400 – 750 nm

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
| Activation Laser (405 nm) | source | identity | 1.0 | — |
| Excitation Laser (647 nm) | source | identity | 1.0 | — |
| Objective Lens (100x / 1.49 NA Oil) | lens | convolution | 0.7 | aberration |
| EMCCD Detector | detector | integration | 0.9 | shot_poisson, read_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512] |
| Measurement shape (y) | [512, 512] |
| Forward model type | nonlinear_operator |
| Category module | compressive_mask |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | EMCCD |
| Wavelength Nm | 647 |
| Power Mw | 50 |
| Magnification | 100 |
| Numerical Aperture | 1.49 |
| Psf Sigma Px | 1.3 |
| Immersion | oil |
| Quantum Efficiency | 0.9 |
| Em Gain | 300 |
| Read Noise E | 50 |
| Pixel Size Um | 16 |
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
