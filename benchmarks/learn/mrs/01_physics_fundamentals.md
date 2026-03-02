# 01 — Physics Fundamentals: MR Spectroscopy (MRS)

## 1. Overview

MR spectroscopy measures the chemical composition of tissue by detecting resonance frequency shifts of metabolites. Single-voxel spectroscopy (SVS) acquires a free induction decay (FID) or spin echo from a localized volume. The spectrum reveals peaks at characteristic chemical shifts for metabolites like N-acetylaspartate, choline, creatine, and lactate. The forward model involves spatial localization, k-space encoding, and spectral decomposition.

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
| Localization Sequence (PRESS/STEAM) | modulator | sampling | 0.8 | chemical_shift_displacement |
| Receive Coil | detector | integration | 0.9 | thermal, read_gaussian |

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
| Shimming | second_order |
| Sequence | PRESS |
| Tr Ms | 3000 |
| Te Ms | 135 |
| Voxel Size Mm | [20, 20, 20] |
| N Coils | 1 |
| Bandwidth Hz | 2000 |
| N Spectral Points | 2048 |


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
