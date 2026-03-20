# 01 — Physics Fundamentals: Fluorescence Lifetime Imaging (FLIM)

## 1. Overview

Fluorescence Lifetime Imaging Microscopy (FLIM) measures the fluorescence decay kinetics at each pixel, providing contrast based on the excited-state lifetime rather than intensity alone. A pulsed laser excites fluorophores, and the time-correlated single photon counting (TCSPC) detector records arrival-time histograms. The forward model is y(t) = IRF(t) * [sum_i a_i * exp(-t/tau_i)] + background, where IRF is the instrument response function and tau_i are the lifetime components. Phasor analysis or maximum likelihood fitting recover the lifetime parameters at each pixel.

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

405 – 488 nm

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
| Dichroic Mirror | filter | modulation | 0.95 | — |
| Objective Lens (60x / 1.4 NA Oil) | lens | convolution | 0.75 | aberration |
| Emission Filter | filter | modulation | 0.88 | — |
| TCSPC Detector | detector | integration | 0.2 | shot_poisson, thermal |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256, 2] |
| Measurement shape (y) | [256, 256, 256] |
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | SPAD |
| Wavelengths Nm | [405, 488] |
| Pulse Width Ps | 70 |
| Repetition Rate Mhz | 80 |
| Average Power Mw | 1 |
| Edge Wavelength Nm | 500 |
| Reflection Band Nm | [400, 490] |
| Transmission Band Nm | [510, 700] |
| Magnification | 60 |
| Numerical Aperture | 1.4 |
| Immersion | oil |
| Psf Sigma Px | 1.0 |
| Center Nm | 530 |
| Bandwidth Nm | 40 |
| Time Resolution Ps | 50 |
| N Time Bins | 256 |
| Time Range Ns | 12.5 |
| Dark Count Rate Cps | 25 |
| Dead Time Ns | 50 |
| Quantum Efficiency | 0.2 |
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
