# 01 — Physics Fundamentals: STED Microscopy

## 1. Overview

Stimulated Emission Depletion (STED) microscopy achieves sub-diffraction resolution by using a depletion donut beam to confine fluorescence emission to a sub-diffraction spot. The effective PSF is determined by both the excitation focus and the depletion beam profile. STED resolution scales with depletion laser power: d ~ lambda / (2*NA*sqrt(1 + I/I_sat)).

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

400 – 775 nm

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
| Excitation Laser (640 nm) | source | identity | 1.0 | — |
| Depletion Laser (775 nm) | source | modulation | 1.0 | — |
| Objective Lens (100x / 1.4 NA Oil) | lens | convolution | 0.7 | aberration |
| APD / Hybrid Detector | detector | integration | 0.45 | shot_poisson, read_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512] |
| Measurement shape (y) | [512, 512] |
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | hybrid_detector |
| Wavelength Nm | 775 |
| Pulse Width Ps | 100 |
| Power Mw | 200 |
| Beam Profile | donut |
| Magnification | 100 |
| Numerical Aperture | 1.4 |
| Effective Psf Fwhm Nm | 50 |
| Immersion | oil |
| Quantum Efficiency | 0.45 |
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
