# 01 — Physics Fundamentals: Portal Imaging (EPID)

## 1. Overview

Portal Imaging (EPID) imaging modality with DAG: Pi --> D.

**Category**: Medical Imaging
**Carrier**: MV

---

## 2. MV Physics

Megavoltage (MV) imaging uses the treatment beam of a linear accelerator (typically 6 MV) to create portal images for verification of radiation therapy positioning.

### Key Concepts

- MV beam characteristics and Compton scattering dominance
- Electronic portal imaging devices (EPID)
- Low contrast due to Compton dominance
- Patient positioning verification
- Dose calculation from portal images

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
I = I₀ · exp(-∫ μ_compton(E, ρ) dl) + scatter
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
| Forward model type | linear_operator |
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
| Physical probe | MV |
| Primary contrast | Determined by mv-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
