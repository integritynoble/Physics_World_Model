# 01 — Physics Fundamentals: Low-Dose Widefield Microscopy

## 1. Overview

Widefield fluorescence microscopy operated at very low illumination power or short exposure time to reduce phototoxicity and photobleaching. Images are dominated by shot noise and read noise. Reconstruction requires denoising (BM3D, Noise2Void, CARE) before or jointly with deconvolution.

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
| Attenuated LED Source | source | identity | 1.0 | — |
| Excitation Filter | filter | modulation | 0.85 | — |
| Objective Lens (40x / 1.3 NA Oil) | lens | convolution | 0.78 | aberration |
| Emission Filter | filter | modulation | 0.9 | — |
| sCMOS Detector (Low Exposure) | detector | integration | 0.82 | shot_poisson, read_gaussian, fixed_pattern |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512] |
| Measurement shape (y) | [512, 512] |
| Forward model type | linear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | LED |
| Power Mw | 2 |
| Spectral Width Nm | 30 |
| Attenuation Factor | 0.04 |
| Center Nm | 525 |
| Bandwidth Nm | 50 |
| Magnification | 40 |
| Numerical Aperture | 1.3 |
| Psf Sigma Px | 1.2 |
| Immersion | oil |
| Pixel Size Um | 6.5 |
| Read Noise E | 1.6 |
| Quantum Efficiency | 0.82 |
| Exposure Ms | 5 |
| Photon Count Mean | 20 |
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
