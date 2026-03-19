# 01 — Physics Fundamentals: Synthetic Aperture Radar (SAR)

## 1. Overview

Synthetic Aperture Radar (SAR) forms high-resolution images of terrain by coherently combining radar returns acquired along a flight path. The synthetic aperture created by platform motion yields azimuth resolution independent of range. The forward model involves a Radon-like projection with complex-valued returns: y = A * x + n.

**Category**: Remote Sensing
**Carrier**: RF

---

## 2. RF Physics

Radiofrequency (RF) imaging uses electromagnetic waves in the MHz-GHz range. Synthetic aperture radar (SAR) achieves high spatial resolution by synthesising a large antenna aperture from platform motion. The signal is coherent, enabling phase-based measurements like interferometry.

### Key Concepts

- Range resolution: Δr = c / (2B)
- Azimuth resolution and synthetic aperture
- Range-Doppler processing
- Phase coherence and interferometry
- Speckle noise (multiplicative)

### Wavelength / Energy Range

30000000.0 – 1000000000.0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
s(t) = Σ_n  σ_n · exp(-j4πf_c R_n(t)/c) · rect(t/T)
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| Radar Transmitter | source | identity | 1.0 | — |
| Synthetic Aperture Processor | modulator | projection | 0.9 | speckle |
| Radar Receiver | detector | integration | 0.85 | thermal, quantization |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512] |
| Measurement shape (y) | [512, 512] |
| Forward model type | linear_operator |
| Category module | medical_mri_kspace |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | microwave |
| Frequency Ghz | 9.6 |
| Bandwidth Mhz | 150 |
| Aperture Length M | 10 |
| Platform Velocity M Per S | 200 |
| Noise Figure Db | 3.0 |
| Adc Bits | 14 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | RF |
| Primary contrast | Determined by rf-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
