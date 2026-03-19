# 01 — Physics Fundamentals: Single-Pixel Camera (SPC)

## 1. Overview

A single-pixel camera acquires an image by projecting a sequence of spatial patterns (Hadamard, random, or learned) onto the scene and measuring total intensity with a single photodetector. The measurement model is y = A*x where A is the pattern matrix and x is the vectorized image. Compressed sensing algorithms (FISTA, ISTA-Net) recover x from M << N measurements.

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

400 – 1000 nm

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
| Scene Illumination (Ambient or Active) | source | identity | 1.0 | — |
| Collection Lens | lens | convolution | 0.92 | aberration |
| Digital Micromirror Device (DMD) | modulator | modulation | 0.68 | fixed_pattern, alignment |
| Condensing Lens | lens | projection | 0.9 | — |
| Single-Pixel Photodiode | detector | integration | 0.9 | shot_poisson, read_gaussian, thermal |

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
| Type | InGaAs_photodiode |
| Spectral Range Nm | [400, 1000] |
| Focal Length Mm | 25 |
| F Number | 2.8 |
| Resolution | [1024, 768] |
| Mirror Pitch Um | 10.8 |
| Tilt Angle Deg | 12 |
| Pattern Type | hadamard |
| Sampling Rate | 0.15 |
| Responsivity A Per W | 0.95 |
| Nep Pw Per Sqrt Hz | 3.0 |
| Bandwidth Khz | 100 |
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
