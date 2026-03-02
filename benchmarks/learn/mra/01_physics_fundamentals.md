# 01 — Physics Fundamentals: MR Angiography (MRA)

## 1. Overview

MR Angiography (MRA) imaging modality with DAG: M --> F --> S --> D.

**Category**: Medical Imaging
**Carrier**: Spin/RF

---

## 2. Spin/RF Physics

Spin/RF imaging exploits nuclear magnetic resonance (NMR). Protons in a strong B₀ field precess at the Larmor frequency ω = γB₀. RF pulses tip the magnetisation into the transverse plane, and gradient fields provide spatial encoding via the Fourier relationship between k-space and image space.

### Key Concepts

- Larmor equation: ω = γ B₀ (γ/2π = 42.577 MHz/T for ¹H)
- T1 relaxation (spin-lattice), T2 relaxation (spin-spin)
- Gradient encoding: frequency, phase, slice selection
- k-space: 2D/3D Fourier relationship
- Multi-coil arrays and parallel imaging

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
s(t) = ∫∫ ρ(x,y) · S_c(x,y) · e^(-i2π k·r) dr
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
| Forward model type | linear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Density | 0.5 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Spin/RF |
| Primary contrast | Determined by spin/rf-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
