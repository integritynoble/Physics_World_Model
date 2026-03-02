# 01 — Physics Fundamentals: Lensless (Diffuser Camera) Imaging

## 1. Overview

Lensless cameras replace the conventional lens with a thin optical element (diffuser, phase mask, or coded aperture) placed directly on the sensor. The captured image is a convolution of the scene with the system PSF, which is typically a large, structured caustic pattern. The forward model is y = PSF ** x + n. Because the PSF is known (measured via calibration), computational reconstruction via ADMM-TV or FlatNet recovers the scene image.

**Category**: Computational Photography
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
| Phase Diffuser | mask | convolution | 0.7 | fixed_pattern, alignment |
| Spacer / Air Gap | medium | propagation | 0.98 | — |
| CMOS Image Sensor | detector | integration | 0.78 | shot_poisson, read_gaussian, fixed_pattern, quantization |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256, 3] |
| Measurement shape (y) | [256, 256, 3] |
| Forward model type | linear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | random_phase_diffuser |
| Material | polycarbonate |
| Thickness Um | 500 |
| Feature Size Um | 100 |
| Psf Sigma Px | 10.0 |
| Psf Support Px | 256 |
| Distance Mm | 2.0 |
| Refractive Index | 1.0 |
| Pixel Size Um | 1.4 |
| Read Noise E | 2.5 |
| Quantum Efficiency | 0.78 |
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
