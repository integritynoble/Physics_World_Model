# 01 — Physics Fundamentals: Photoacoustic Imaging

## 1. Overview

Photoacoustic imaging (PAI) combines optical absorption contrast with ultrasonic detection resolution. A short-pulsed laser illuminates the tissue, and optical absorption generates an initial pressure distribution p0 = Gamma * mu_a * Phi, where Gamma is the Grueneisen parameter, mu_a is the absorption coefficient, and Phi is the local optical fluence. The initial pressure propagates as acoustic waves detected by an ultrasound transducer array. The forward model is y = R * p0 where R is the acoustic propagation operator. Back-projection or time-reversal algorithms reconstruct the initial pressure distribution from the measured RF data.

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

532 – 1064 nm

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
| Pulsed Laser Source | source | identity | 1.0 | — |
| Tissue Absorption | medium | modulation | 0.3 | — |
| Acoustic Propagation | medium | propagation | 0.85 | — |
| Ultrasound Transducer Array | detector | integration | 0.75 | shot_poisson, read_gaussian, thermal |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256] |
| Measurement shape (y) | [128, 2048] |
| Forward model type | linear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | linear_array |
| Wavelengths Nm | [532, 1064] |
| Pulse Energy Mj | 20 |
| Pulse Duration Ns | 5 |
| Repetition Rate Hz | 10 |
| Grueneisen Parameter | 0.8 |
| Background Mu A Per Cm | 0.1 |
| Fluence Model | diffusion |
| Tissue Depth Mm | 30 |
| Speed Of Sound M Per S | 1540 |
| Attenuation Db Per Cm Per Mhz | 0.5 |
| Medium | soft_tissue |
| N Elements | 128 |
| Element Pitch Mm | 0.3 |
| Center Frequency Mhz | 5.0 |
| Bandwidth Percent | 60 |
| Sampling Rate Mhz | 40 |
| Time Samples | 2048 |
| Bit Depth | 14 |


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
