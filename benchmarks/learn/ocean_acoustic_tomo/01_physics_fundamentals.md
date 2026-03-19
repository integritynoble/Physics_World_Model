# 01 — Physics Fundamentals: Ocean Acoustic Tomography

## 1. Overview

Ocean Acoustic Tomography imaging modality with DAG: P --> D.

**Category**: Broader Experimental Science
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
| Source/Emitter | source | propagation | 1.0 | — |
| Detector | detector | integration | 0.8 | gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [64, 64] |
| Measurement shape (y) | [64, 64] |
| Forward model type | linear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

No specific physics parameters defined.


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
