# 01 — Physics Fundamentals: Fourier Ptychographic Microscopy (FPM)

## 1. Overview

Fourier Ptychographic Microscopy (FPM) achieves high-resolution, wide field-of-view, and quantitative phase imaging by illuminating the sample with an LED array at variable angles and capturing low-resolution intensity images through a low-NA objective. Each LED angle shifts the sample's Fourier spectrum, and the low-NA objective acts as a pupil filter. The forward model for illumination angle j is y_j = |F^{-1}{P(k - k_j) * O(k)}|^2, where P is the pupil function, O(k) is the object's Fourier spectrum, and k_j is the illumination wave vector. Sequential phase retrieval or gradient descent algorithms stitch together the Fourier components to recover a high-resolution complex image with both amplitude and phase.

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

450 – 650 nm

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
| LED Array (Variable Angle Illumination) | source | modulation | 0.9 | alignment |
| Sample (Thin Specimen) | sample | modulation | 0.8 | — |
| Objective Lens (4x / 0.1 NA) | lens | convolution | 0.85 | aberration |
| CMOS Camera | detector | integration | 0.78 | shot_poisson, read_gaussian, quantization |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [1024, 1024] |
| Measurement shape (y) | [256, 256, 225] |
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | thin_sample |
| Array Size | [15, 15] |
| N Leds | 225 |
| Led Pitch Mm | 4.0 |
| Center Wavelength Nm | 530 |
| Spectral Width Nm | 20 |
| Array To Sample Distance Mm | 80 |
| Max Phase Shift Rad | 2.0 |
| Max Absorption | 0.5 |
| Magnification | 4 |
| Numerical Aperture | 0.1 |
| Immersion | air |
| Synthetic Na | 0.5 |
| Resolution Gain Factor | 5 |
| Pixel Size Um | 6.5 |
| Read Noise E | 2.0 |
| Quantum Efficiency | 0.78 |
| Full Well E | 30000 |
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
