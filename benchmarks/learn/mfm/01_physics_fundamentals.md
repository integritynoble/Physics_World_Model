# 01 — Physics Fundamentals: Magnetic Force Microscopy (MFM)

## 1. Overview

Magnetic Force Microscopy (MFM) imaging modality with DAG: S --> M --> D.

**Category**: Scanning Probe Microscopy
**Carrier**: Magnetic

---

## 2. Magnetic Physics

Magnetic imaging detects local magnetic fields or magnetic properties of materials. Techniques range from magnetic force microscopy (MFM) to magnetic particle imaging (MPI). The signal depends on the spatial distribution of magnetic moments.

### Key Concepts

- Magnetic dipole fields and stray fields
- Lift-mode scanning (MFM)
- Langevin magnetisation curve (MPI)
- System function and calibration
- Spatial encoding via drive fields

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
u(t) = -dΦ/dt = -μ₀ · ∫ ∂M/∂H · dH/dt · S(r) dr
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
| Category module | scanning_probe |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Density | 0.5 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Magnetic |
| Primary contrast | Determined by magnetic-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
