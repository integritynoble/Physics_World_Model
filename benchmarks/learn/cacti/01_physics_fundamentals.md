# 01 — Physics Fundamentals: Coded Aperture Compressive Temporal Imaging (CACTI)

## 1. Overview

CACTI (also known as Snapshot Compressive Imaging for video) captures multiple video frames in a single 2D exposure using time-varying coded aperture masks. Each video frame is multiplied by a different shifted version of a base mask, and all frames sum on the detector. The forward model is y = sum_t mask_t * x_t. GAP-TV or EfficientSCI reconstructs the video sequence from the single snapshot.

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
| Scene Illumination | source | identity | 1.0 | — |
| Objective Lens | lens | convolution | 0.88 | aberration |
| Coded Aperture (Shifting Mask) | mask | modulation | 0.5 | fixed_pattern, alignment |
| Relay Optics | lens | convolution | 0.9 | — |
| CMOS Detector | detector | integration | 0.78 | shot_poisson, read_gaussian, quantization |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256, 8] |
| Measurement shape (y) | [256, 256] |
| Forward model type | linear_operator |
| Category module | compressive_mask |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | ambient |
| Frame Rate Equiv Fps | 240 |
| Focal Length Mm | 75 |
| F Number | 2.0 |
| Mask Type | binary_random |
| Density | 0.5 |
| N Frames | 8 |
| Shift Type | vertical |
| Shift Step Px | 1 |
| Pixel Size Um | 5.5 |
| Read Noise E | 3.0 |
| Quantum Efficiency | 0.78 |
| Exposure Ms | 33 |
| Bit Depth | 10 |


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
