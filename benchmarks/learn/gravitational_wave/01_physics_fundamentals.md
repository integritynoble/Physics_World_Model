# 01 — Physics Fundamentals: Gravitational Wave Detection

## 1. Overview

Gravitational Wave Detection imaging modality with DAG: P --> Sigma --> D.

**Category**: Broader Experimental Science
**Carrier**: Gravitational

---

## 2. Gravitational Physics

Gravitational wave detection measures spacetime strain caused by accelerating masses (merging black holes, neutron stars). Laser interferometers (LIGO, Virgo) detect differential arm length changes of ~10⁻²¹ m.

### Key Concepts

- General relativity and gravitational radiation
- Laser interferometry and Michelson configuration
- Strain sensitivity: h ~ ΔL/L ~ 10⁻²¹
- Matched filtering for signal extraction
- Noise: seismic, thermal, shot, quantum

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
h(t) = (ΔL₊ - ΔL₋) / L
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
| Physical probe | Gravitational |
| Primary contrast | Determined by gravitational-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
