# 01 — Physics Fundamentals: Passive Microwave Radiometry

## 1. Overview

Passive Microwave Radiometry imaging modality with DAG: Sigma --> D.

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

0 – 0 nm

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
| Scattering | medium | modulation | 0.7 | — |
| Detector | detector | integration | 0.8 | poisson |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [64, 64] |
| Measurement shape (y) | [64, 64] |
| Forward model type | linear_operator |
| Category module | remote_sensing_sar |

---

## 5. Key Physics Parameters

No specific physics parameters defined.


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
