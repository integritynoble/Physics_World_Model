# 01 — Physics Fundamentals: Doppler Ultrasound

## 1. Overview

Doppler ultrasound measures blood flow velocity by detecting frequency shifts in reflected ultrasound waves. Color Doppler maps 2D velocity fields, while spectral (pulsed-wave) Doppler provides quantitative velocity-time waveforms at a sample volume. The forward model combines acoustic propagation with Doppler frequency estimation from the autocorrelation of IQ data.

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
| Piezoelectric Transducer Array | source | identity | 1.0 | — |
| Tissue + Blood Flow | medium | projection | 0.5 | speckle |
| Receive Array + Doppler Processor | detector | integration | 0.9 | thermal_gaussian |

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
| Element Pitch Mm | 0.3 |
| Prf Hz | 10000 |
| Speed Of Sound M Per S | 1540.0 |
| Blood Velocity Range M Per S | [-1.0, 1.0] |
| Sampling Rate Mhz | 40 |
| Wall Filter Cutoff Hz | 50 |
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
