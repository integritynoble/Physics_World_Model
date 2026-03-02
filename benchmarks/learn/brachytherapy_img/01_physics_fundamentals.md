# 01 — Physics Fundamentals: Brachytherapy Imaging

## 1. Overview

Brachytherapy Imaging imaging modality with DAG: Pi --> D.

**Category**: Medical Imaging
**Carrier**: Gamma/X-ray

---

## 2. Gamma/X-ray Physics

This modality uses a specific physical probe to interact with the sample or scene. The interaction produces measurements that encode information about the object's internal structure or surface properties.

### Key Concepts

- Probe-sample interaction mechanism
- Forward model relating object to measurements
- Noise model (signal-dependent and independent)
- Spatial resolution and field of view
- Contrast mechanism and sensitivity

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
y = A(x) + noise
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| Source | source | identity | 1.0 | — |
| Projection | medium | projection | 0.9 | — |
| Detector | detector | integration | 0.85 | poisson |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [64, 64] |
| Measurement shape (y) | [64, 64] |
| Forward model type | nonlinear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| N Angles | 180 |
| Pixel Size Um | 50.0 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Gamma/X-ray |
| Primary contrast | Determined by gamma/x-ray-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
