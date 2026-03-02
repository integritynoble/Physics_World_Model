# 01 — Physics Fundamentals: MALDI Mass Spectrometry Imaging

## 1. Overview

MALDI Mass Spectrometry Imaging imaging modality with DAG: S --> D.

**Category**: Scientific Instrumentation
**Carrier**: Ion

---

## 2. Ion Physics

Ion-based imaging uses focused ion beams or mass spectrometry to map elemental or molecular composition. Ions are extracted from the sample surface and analysed by mass-to-charge ratio, providing spatial chemical maps.

### Key Concepts

- Sputtering and ion yield
- Mass-to-charge ratio analysis
- Spatial resolution vs sensitivity trade-off
- Matrix effects and quantification
- Time-of-flight (ToF) mass analysis

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
I(m/z, x, y) = Y(m/z) · c(x,y) · primary_dose + noise
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
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

No specific physics parameters defined.


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Ion |
| Primary contrast | Determined by ion-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
