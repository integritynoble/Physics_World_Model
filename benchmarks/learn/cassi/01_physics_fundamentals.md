# 01 — Physics Fundamentals: Coded Aperture Snapshot Spectral Imaging (CASSI)

## 1. Overview

CASSI compresses a 3D hyperspectral data cube (x, y, lambda) into a single 2D snapshot measurement. A binary coded aperture mask spatially modulates each spectral band, and a dispersive element (prism or grating) shifts bands laterally before they integrate on the detector. The forward model is y = sum_l mask(x,y) * x(x,y,l) shifted by s(l). Reconstruction via MST, GAP-TV, or HDNet recovers the full spectral cube.

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
| Broadband Illumination Source | source | identity | 1.0 | — |
| Coded Aperture Mask | mask | modulation | 0.5 | fixed_pattern, alignment |
| Relay Lens | lens | convolution | 0.9 | aberration |
| Dispersive Prism | disperser | dispersion | 0.88 | alignment |
| Imaging Lens | lens | convolution | 0.9 | aberration |
| CCD / CMOS Detector | detector | integration | 0.75 | shot_poisson, read_gaussian, quantization |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256, 28] |
| Measurement shape (y) | [256, 310] |
| Forward model type | linear_operator |
| Category module | compressive_mask |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | halogen |
| Spectral Range Nm | [450, 650] |
| N Bands | 28 |
| Band Spacing Nm | 7.14 |
| Mask Type | binary_random |
| Density | 0.5 |
| Feature Size Um | 50 |
| Substrate Material | glass |
| Focal Length Mm | 75 |
| F Number | 4.0 |
| Dispersion Step Px | 2 |
| Dispersion Direction Deg | 0.0 |
| Prism Material | SF11 |
| Total Shift Px | 54 |
| Pixel Size Um | 12.0 |
| Read Noise E | 5.0 |
| Quantum Efficiency | 0.75 |
| Full Well E | 20000 |
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
