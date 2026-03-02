# 01 — Physics Fundamentals: Diffusion MRI (DTI)

## 1. Overview

Diffusion MRI sensitizes the MR signal to water molecule diffusion by applying diffusion-encoding gradients. The Stejskal-Tanner equation describes signal attenuation: S = S_0 * exp(-b * D), where b is the diffusion weighting factor and D is the apparent diffusion coefficient. Diffusion tensor imaging (DTI) estimates a 3x3 diffusion tensor at each voxel, enabling fractional anisotropy mapping and white matter tractography.

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
| Diffusion-Encoding Gradients + EPI | modulator | sampling | 1.0 | eddy_current, motion |
| Receive Coil Array | detector | integration | 0.9 | thermal, read_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [128, 128] |
| Measurement shape (y) | [128, 128] |
| Forward model type | linear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Field Strength T | 3.0 |
| Max Gradient Mt Per M | 80 |
| B Values | [0, 1000] |
| N Directions | 30 |
| Tr Ms | 8000 |
| Te Ms | 80 |
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
