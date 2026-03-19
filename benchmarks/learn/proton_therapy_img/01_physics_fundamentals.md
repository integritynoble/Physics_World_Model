# 01 — Physics Fundamentals: Proton Therapy Imaging

## 1. Overview

Proton Therapy Imaging imaging modality with DAG: Pi --> D.

**Category**: Medical Imaging
**Carrier**: Proton

---

## 2. Proton Physics

Proton imaging uses proton beams (typically 100-250 MeV) that lose energy as they traverse matter via the Bethe-Bloch formula. Proton CT measures the residual energy or range to reconstruct the stopping power distribution.

### Key Concepts

- Bethe-Bloch energy loss formula
- Bragg peak and range
- Multiple Coulomb scattering
- Stopping power and water-equivalent path length
- Comparison with X-ray CT for treatment planning

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
-dE/dx = (4π e⁴ z² N_A Z ρ) / (m_e v²A) · ln(2m_e v²/I)
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
| Physical probe | Proton |
| Primary contrast | Determined by proton-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
