# 01 — Physics Fundamentals: Solar EUV/X-ray Imaging

## 1. Overview

Solar EUV/X-ray Imaging imaging modality with DAG: M --> P --> D.

**Category**: Astronomy & Space Imaging
**Carrier**: Photon/EUV

---

## 2. Photon/EUV Physics

Extreme ultraviolet (EUV) and soft X-ray imaging captures radiation from hot plasmas (10⁶-10⁷ K). Solar EUV imaging reveals the Sun's corona, while EUV lithography uses 13.5 nm light for semiconductor patterning.

### Key Concepts

- EUV emission from hot plasmas
- Multilayer mirror optics
- Differential emission measure (DEM)
- Coronal temperature diagnostics
- EUV absorption in the atmosphere

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
I(λ) = ∫ G(T,λ) · DEM(T) dT
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| Coded Mask | mask | modulation | 0.5 | — |
| Detector | detector | integration | 0.8 | poisson, gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [64, 64] |
| Measurement shape (y) | [64, 64] |
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Density | 0.5 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Photon/EUV |
| Primary contrast | Determined by photon/euv-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
