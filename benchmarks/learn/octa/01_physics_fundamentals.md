# 01 — Physics Fundamentals: OCT Angiography (OCTA)

## 1. Overview

OCT Angiography (OCTA) generates en-face vascular maps of the retina by computing decorrelation or variance between repeated B-scans at the same location. Flowing blood causes signal fluctuation, while static tissue remains stable. The forward model combines OCT interferometry with temporal decorrelation: y = Var_t[OCT(x, t)] + n.

**Category**: Medical Imaging
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

840 – 1060 nm

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
| Swept-Source OCT | source | identity | 1.0 | — |
| Scanning + Objective Optics | lens | convolution | 0.75 | motion_artifact |
| Balanced Photodetector | detector | integration | 0.85 | shot_poisson, read_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512] |
| Measurement shape (y) | [512, 512] |
| Forward model type | nonlinear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | swept_source |
| Center Wavelength Nm | 840 |
| Bandwidth Nm | 50 |
| Sweep Rate Khz | 200 |
| Power Mw | 2 |
| Lateral Resolution Um | 15 |
| Scan Range Mm | 3 |
| N Bscans Per Location | 4 |
| Dynamic Range Db | 100 |
| Sensitivity Dbm | -95 |
| Bit Depth | 14 |


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
