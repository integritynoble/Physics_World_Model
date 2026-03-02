# 01 — Physics Fundamentals: Ultrasound B-mode Imaging

## 1. Overview

Pulse-echo ultrasound imaging transmits acoustic pulses into tissue and records reflected echoes on a transducer array. Beamforming (delay-and-sum) converts RF channel data into a B-mode image. The forward model involves acoustic propagation, receive sensitivity, and thermal noise.

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

### Wavelength / Energy Range

0 – 0 nm

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
| Tissue Medium | medium | projection | 0.5 | speckle |
| Receive Array + ADC | detector | integration | 0.9 | thermal_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256] |
| Measurement shape (y) | [128, 512] |
| Forward model type | linear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| N Elements | 128 |
| Center Frequency Mhz | 5.0 |
| Element Pitch Mm | 0.3 |
| Fractional Bandwidth | 0.6 |
| Speed Of Sound M Per S | 1540.0 |
| Attenuation Db Per Cm Mhz | 0.5 |
| Sampling Rate Mhz | 40 |
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
