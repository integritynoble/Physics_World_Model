# 01 — Physics Fundamentals: Structured Illumination Microscopy (SIM)

## 1. Overview

Structured Illumination Microscopy achieves approximately 2x lateral resolution improvement by illuminating the sample with sinusoidal patterns at multiple orientations and phases. Frequency mixing between the illumination pattern and sample structure shifts high-frequency information into the passband. Reconstruction (Wiener-SIM or DL-SIM) separates and reassembles frequency components.

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
| Coherent Laser (488 nm) | source | identity | 1.0 | — |
| Spatial Light Modulator (Pattern Generator) | modulator | modulation | 0.6 | alignment, fixed_pattern |
| Objective Lens (100x / 1.49 NA Oil) | lens | convolution | 0.7 | aberration |
| Emission Filter | filter | modulation | 0.9 | — |
| sCMOS Detector | detector | integration | 0.82 | shot_poisson, read_gaussian, fixed_pattern |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512] |
| Measurement shape (y) | [512, 512, 9] |
| Forward model type | linear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | laser |
| Wavelength Nm | 488 |
| Power Mw | 20 |
| Coherence Length Mm | 50 |
| N Angles | 3 |
| N Phases | 3 |
| Pattern Frequency Cycles Per Px | 0.1 |
| Pixel Pitch Um | 3.74 |
| Magnification | 100 |
| Numerical Aperture | 1.49 |
| Psf Sigma Px | 1.5 |
| Immersion | oil |
| Center Nm | 525 |
| Bandwidth Nm | 50 |
| Pixel Size Um | 6.5 |
| Read Noise E | 1.6 |
| Quantum Efficiency | 0.82 |
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
