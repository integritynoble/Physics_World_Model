# 01 — Physics Fundamentals: Light-Sheet Fluorescence Microscopy (LSFM)

## 1. Overview

Light-sheet fluorescence microscopy (LSFM / SPIM) illuminates the sample with a thin sheet of light perpendicular to the detection axis, providing intrinsic optical sectioning. The primary artifacts are stripe patterns caused by absorption and scattering in the illumination path, plus anisotropic PSF blur. Reconstruction involves destriping followed by optional deconvolution.

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
| Laser Source (488 nm) | source | identity | 1.0 | — |
| Cylindrical Lens (Sheet Former) | lens | convolution | 0.9 | alignment |
| Sample Medium | medium | modulation | 0.85 | aberration |
| Detection Objective (20x / 1.0 NA Water) | lens | convolution | 0.78 | aberration |
| sCMOS Detector | detector | integration | 0.82 | shot_poisson, read_gaussian, fixed_pattern |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512, 128] |
| Measurement shape (y) | [512, 512, 128] |
| Forward model type | linear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | laser |
| Wavelength Nm | 488 |
| Power Mw | 10 |
| Sheet Thickness Um | 4.0 |
| Sheet Width Um | 500 |
| Rayleigh Length Um | 200 |
| Refractive Index | 1.33 |
| Scattering Mean Free Path Um | 100 |
| Stripe Strength | 0.2 |
| Attenuation Coef | 0.02 |
| Magnification | 20 |
| Numerical Aperture | 1.0 |
| Psf Sigma Lateral Px | 1.5 |
| Psf Sigma Axial Px | 1.0 |
| Immersion | water |
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
