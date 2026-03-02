# 01 — Physics Fundamentals: Time-of-Flight Depth Camera

## 1. Overview

Time-of-flight depth cameras measure scene depth by emitting modulated near-infrared light and measuring the phase shift or time delay of the reflected signal. The sensor (typically SPAD or demodulation pixels) bins photon arrivals into time gates. The forward model is y = Poisson(alpha * G(x)) + N(0, sigma^2) where G is the ToF gating operator that converts depth to time-binned photon counts.

**Category**: Depth Imaging
**Carrier**: Photon/IR

---

## 2. Photon/IR Physics

Time-of-flight cameras measure the round-trip time of modulated light (IR LEDs or lasers) to compute depth. The phase shift of the reflected signal is proportional to distance.

### Key Concepts

- Amplitude-modulated continuous wave (AMCW)
- Phase-to-depth conversion: d = c·φ/(4πf_mod)
- Multi-frequency disambiguation
- Systematic errors: multipath, motion blur
- Depth resolution and range trade-off

### Wavelength / Energy Range

850 – 940 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
d = c · Δφ / (4π f_mod)
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| NIR Laser Diode | source | identity | 1.0 | — |
| Scene (Depth Object) | medium | projection | 0.3 | multipath |
| SPAD Array Sensor | detector | integration | 0.3 | shot_poisson, read_gaussian, dark_count |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256] |
| Measurement shape (y) | [256, 256] |
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | laser_diode |
| Wavelength Nm | 905 |
| Power Mw | 500 |
| Modulation Freq Mhz | 20 |
| Max Range M | 10.0 |
| Reflectivity | 0.3 |
| N Bins | 64 |
| Bin Width Ns | 1.0 |
| Dead Time Ns | 50 |
| Qe | 0.3 |
| Dark Count Rate Cps | 100 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Photon/IR |
| Primary contrast | Determined by photon/ir-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
