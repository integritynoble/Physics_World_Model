# 01 — Physics Fundamentals: Eddy Current Imaging

## 1. Overview

Eddy Current Imaging imaging modality with DAG: F --> D.

**Category**: Industrial Inspection
**Carrier**: EM

---

## 2. EM Physics

Electromagnetic induction imaging uses alternating magnetic fields to induce eddy currents in conductive materials. Changes in impedance due to defects, cracks, or material variations are detected by the probe coil.

### Key Concepts

- Eddy current induction and skin depth
- Impedance plane analysis
- Probe design and lift-off effects
- Frequency selection and penetration depth
- Defect detection sensitivity

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
ΔZ = ΔR + jΔX = f(σ, μ, geometry, frequency)
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| Fourier Sampling | filter | modulation | 1.0 | — |
| Signal Encoding | medium | modulation | 0.9 | — |
| Detector | detector | integration | 0.85 | gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [64, 64] |
| Measurement shape (y) | [64, 64] |
| Forward model type | nonlinear_operator |
| Category module | medical_mri_kspace |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Acceleration | 4 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | EM |
| Primary contrast | Determined by em-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
