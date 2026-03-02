# 01 — Physics Fundamentals: Seismic Tomography

## 1. Overview

Seismic Tomography imaging modality with DAG: P --> D.

**Category**: Broader Experimental Science
**Carrier**: Seismic

---

## 2. Seismic Physics

Seismic imaging uses low-frequency elastic waves to probe Earth's interior. Travel times of reflected/refracted waves are inverted to reconstruct velocity and density structure.

### Key Concepts

- Seismic wave propagation
- Travel time tomography
- Ray theory and Fermat's principle
- Velocity model building
- Resolution kernels

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
t = ∫_ray 1/v(x) dl
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
| Forward model type | linear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

No specific physics parameters defined.


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Seismic |
| Primary contrast | Determined by seismic-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
