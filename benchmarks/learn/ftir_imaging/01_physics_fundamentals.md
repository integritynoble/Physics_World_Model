# 01 — Physics Fundamentals: FTIR Spectroscopic Imaging

## 1. Overview

FTIR Spectroscopic Imaging imaging modality with DAG: M --> Sigma --> D.

**Category**: Spectroscopy & Spectral Imaging
**Carrier**: IR

---

## 2. IR Physics

Infrared imaging detects thermal radiation or IR absorption. Every object above absolute zero emits thermal radiation according to Planck's law. Active IR techniques use external illumination to probe material properties via absorption spectroscopy.

### Key Concepts

- Planck's law and blackbody radiation
- Emissivity and thermal contrast
- Atmospheric transmission windows
- Microbolometer and cooled detector arrays
- Thermal diffusivity and heat equation

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
L(λ,T) = (2hc²/λ⁵) · 1/(exp(hc/λkT) - 1)
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
| Forward model type | linear_operator |
| Category module | compressive_mask |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Density | 0.5 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | IR |
| Primary contrast | Determined by ir-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
