# 01 — Physics Fundamentals: Shear-Wave Elastography

## 1. Overview

Shear-wave elastography (SWE) measures tissue stiffness by generating shear waves via acoustic radiation force and tracking their propagation speed. The shear wave speed c_s is related to shear modulus by G = rho * c_s^2. The forward model combines acoustic push generation, shear wave propagation through viscoelastic tissue, and ultrasonic tracking of tissue displacement.

**Category**: Medical Imaging
**Carrier**: Acoustic

---

## 2. Acoustic Physics

Acoustic imaging uses sound waves (1-50 MHz for medical ultrasound, kHz for sonar). Waves are transmitted into the medium, and reflections from impedance boundaries are received and beamformed to create images. The speed of sound (~1540 m/s in tissue) and acoustic impedance Z = ρ·c determine contrast.

### Key Concepts

- Acoustic impedance Z = ρ·c and reflection coefficient
- Beamforming: delay-and-sum (DAS), adaptive methods
- Frequency-dependent attenuation
- Phased arrays and synthetic aperture focusing
- Doppler effect for flow measurement

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
y(t) = Σ_i  A_i · s(t - 2r_i/c) + noise
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| Piezoelectric Transducer (Push + Track) | source | identity | 1.0 | — |
| Viscoelastic Tissue | medium | propagation | 0.7 | speckle |
| Receive Array + Displacement Tracker | detector | integration | 0.9 | thermal_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256] |
| Measurement shape (y) | [128, 512] |
| Forward model type | nonlinear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| N Elements | 128 |
| Center Frequency Mhz | 5.0 |
| Push Duration Us | 100 |
| Tracking Prf Hz | 10000 |
| Shear Modulus Kpa | 10.0 |
| Density Kg M3 | 1000.0 |
| Viscosity Pa S | 0.5 |
| Sampling Rate Mhz | 40 |
| Displacement Sensitivity Um | 0.1 |
| Bit Depth | 12 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Acoustic |
| Primary contrast | Determined by acoustic-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
