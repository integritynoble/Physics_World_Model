# 01 — Physics Fundamentals: Full-Waveform Inversion (FWI)

## 1. Overview

Full-Waveform Inversion (FWI) imaging modality with DAG: P --> D.

**Category**: Broader Experimental Science
**Carrier**: Seismic/Acoustic

---

## 2. Seismic/Acoustic Physics

Seismic imaging uses low-frequency acoustic/elastic waves (1-100 Hz) to probe Earth's subsurface structure. Sources generate compressional (P) and shear (S) waves that reflect and refract at geological boundaries.

### Key Concepts

- P-waves and S-waves
- Reflection and refraction at interfaces
- Normal moveout (NMO) and stacking
- Migration: converting time to depth
- Full waveform inversion (FWI)

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
∇²u - (1/c²) ∂²u/∂t² = f(x,t)
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
| Physical probe | Seismic/Acoustic |
| Primary contrast | Determined by seismic/acoustic-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
