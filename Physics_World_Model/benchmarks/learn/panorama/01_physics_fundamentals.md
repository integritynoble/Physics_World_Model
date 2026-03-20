# 01 — Physics Fundamentals: Panorama Multi-Focus Fusion

## 1. Overview

Panoramic multi-focus fusion combines multiple images captured at different focal distances into a single all-in-focus composite. Each input image has a different depth plane in focus while others are blurred by the defocus PSF. The forward model for each image is y_k = PSF(d_k) ** x + n where d_k is the defocus distance. Fusion algorithms (Laplacian pyramid, guided filter, IFCNN) detect sharp regions in each input and blend them seamlessly.

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
| Camera Lens (Variable Focus) | lens | convolution | 0.88 | aberration |
| Focus Bracketing Controller | modulator | modulation | 1.0 | alignment |
| RGB Image Sensor | detector | integration | 0.82 | shot_poisson, read_gaussian, quantization |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512, 3] |
| Measurement shape (y) | [512, 512, 3] |
| Forward model type | linear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | ambient |
| Focal Length Mm | 50 |
| Aperture | f/2.8 |
| N Focus Planes | 5 |
| Focus Range M | [0.3, 10.0] |
| N Images | 5 |
| Focus Step Diopters | 0.5 |
| Alignment Method | feature_based |
| Pixel Size Um | 4.0 |
| Read Noise E | 3.0 |
| Quantum Efficiency | 0.82 |
| Bit Depth | 14 |
| Color Filter | Bayer_RGGB |


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
