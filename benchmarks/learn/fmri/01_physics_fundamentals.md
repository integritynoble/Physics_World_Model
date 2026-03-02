# 01 — Physics Fundamentals: Functional MRI (BOLD fMRI)

## 1. Overview

Functional MRI detects brain activity via the blood-oxygen-level-dependent (BOLD) contrast mechanism. T2*-weighted EPI sequences are acquired rapidly (TR ~2s) while the subject performs tasks or rests. The BOLD signal reflects changes in deoxyhemoglobin concentration due to neurovascular coupling. The forward model involves k-space undersampling, T2* decay, and hemodynamic response function convolution.

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
| Main Magnet (3T) | source | identity | 1.0 | — |
| EPI Gradient Encoding | modulator | sampling | 1.0 | alignment |
| Receive Coil Array | detector | integration | 0.9 | thermal, read_gaussian |

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
| Field Strength T | 3.0 |
| Bore Diameter Cm | 60 |
| Tr Ms | 2000 |
| Te Ms | 30 |
| Echo Spacing Ms | 0.5 |
| Acceleration Factor | 2 |
| N Coils | 32 |
| Coil Type | head_array |
| Bandwidth Hz | 250000 |


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
