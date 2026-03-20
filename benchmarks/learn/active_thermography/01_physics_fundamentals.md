# 01 — Physics Fundamentals: Active Thermography (IR)

## 1. Overview

Active Thermography (IR) imaging modality with DAG: P --> D.

**Category**: Industrial Inspection
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
| Source/Emitter | source | propagation | 1.0 | — |
| Detector | detector | integration | 0.8 | gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [64, 64] |
| Measurement shape (y) | [64, 64] |
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

No specific physics parameters defined.


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
