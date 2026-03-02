# 01 — Physics Fundamentals: Optical Coherence Tomography (OCT)

## 1. Overview

Optical Coherence Tomography (OCT) is an interferometric imaging technique that produces high-resolution cross-sectional images of biological tissue. A broadband or swept-source light source is split into sample and reference arms. Backscattered light from the sample interferes with the reference beam, producing spectral fringes. The forward model is y(k) = |E_r + E_s(k)|^2 which expands to DC + cross-correlation + autocorrelation terms. The depth profile (A-scan) is obtained by Fourier transforming the spectral interferogram after DC removal and dispersion compensation. B-scans and volumetric C-scans are formed by lateral scanning.

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
| SLD / Swept Source | source | identity | 1.0 | — |
| Beam Splitter (Interferometer) | beamsplitter | interference | 0.5 | — |
| Sample Arm (Scanning + Objective) | lens | convolution | 0.75 | aberration, alignment |
| Reference Arm | source | interference | 0.9 | — |
| Spectrometer / Balanced Detector | detector | integration | 0.8 | shot_poisson, read_gaussian, thermal |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512, 512] |
| Measurement shape (y) | [512, 512, 1024] |
| Forward model type | nonlinear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | balanced_photodetector |
| Center Wavelength Nm | 1060 |
| Bandwidth Nm | 100 |
| Sweep Rate Khz | 100 |
| Power Mw | 20 |
| Coherence Length Mm | 6 |
| Splitting Ratio | 0.5 |
| Scan Type | galvo_xy |
| Numerical Aperture | 0.05 |
| Lateral Resolution Um | 15 |
| Scan Range Mm | 6 |
| Working Distance Mm | 25 |
| Reference Amplitude | 1.0 |
| Dispersion Compensation | True |
| Path Length Matching Um | 0.1 |
| N Spectral Samples | 1024 |
| Dynamic Range Db | 105 |
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
