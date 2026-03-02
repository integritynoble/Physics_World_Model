# 01 — Physics Fundamentals: Electrical Impedance Tomography (EIT)

## 1. Overview

Electrical Impedance Tomography (EIT) imaging modality with DAG: M --> D.

**Category**: Broader Experimental Science
**Carrier**: Electric

---

## 2. Electric Physics

Electrical impedance imaging applies small alternating currents through electrodes and measures the resulting voltages to reconstruct the conductivity distribution inside the body or object.

### Key Concepts

- Ohm's law: V = I · Z
- Conductivity σ and permittivity ε
- Ill-posedness and regularisation
- Electrode models: complete, shunt, gap
- Temporal difference imaging

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
∇·(σ∇φ) = 0  with boundary conditions
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
| Physical probe | Electric |
| Primary contrast | Determined by electric-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
