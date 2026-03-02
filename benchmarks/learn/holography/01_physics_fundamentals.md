# 01 — Physics Fundamentals: Digital Holographic Microscopy

## 1. Overview

Digital holography records the interference pattern between an object wave and a reference wave on a 2D sensor. In off-axis geometry, a tilted reference separates the twin image and DC terms in Fourier space. Numerical reconstruction propagates the wavefield back to the sample plane using the angular spectrum method. Both amplitude and quantitative phase are recovered, enabling label-free cell imaging.

**Category**: Coherent Imaging
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
| Coherent Laser Source (633 nm) | source | identity | 1.0 | — |
| Beam Splitter | beamsplitter | interference | 0.5 | — |
| Microscope Objective (40x / 0.65 NA) | lens | convolution | 0.8 | aberration |
| Off-Axis Reference Beam | source | interference | 1.0 | alignment |
| CCD Detector | detector | integration | 0.7 | shot_poisson, read_gaussian, quantization |

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
| Type | cube |
| Wavelength Nm | 633 |
| Power Mw | 5 |
| Coherence Length M | 0.3 |
| Splitting Ratio | 0.5 |
| Magnification | 40 |
| Numerical Aperture | 0.65 |
| Immersion | air |
| Carrier Frequency Cycles Per Px | 0.2 |
| Reference Amplitude | 1.0 |
| Tilt Angle Deg | 2.5 |
| Pixel Size Um | 4.65 |
| Read Noise E | 8.0 |
| Quantum Efficiency | 0.7 |
| Full Well E | 18000 |
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
