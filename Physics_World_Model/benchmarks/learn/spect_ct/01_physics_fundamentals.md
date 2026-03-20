# 01 — Physics Fundamentals: SPECT/CT Fusion

## 1. Overview

SPECT/CT Fusion imaging modality with DAG: Pi --> D (SPECT) + Pi --> D (CT) --> Fusion.

**Category**: Multi-Modal Fusion
**Carrier**: Gamma

---

## 2. Gamma Physics

Gamma-ray imaging detects high-energy photons (140 keV for ⁹⁹ᵐTc, 511 keV for PET) emitted by radioactive tracers inside the body. Collimation (SPECT) or coincidence detection (PET) provides directional information. The forward model is based on line integrals of the tracer distribution, similar to CT but in emission mode.

### Key Concepts

- Radioactive decay and tracer kinetics
- Collimation and coincidence detection
- Attenuation correction
- Scatter and randoms correction
- Resolution recovery and PSF modelling

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
y_i = ∫_Li  f(x) · a(x) dl + scatter + randoms
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| Source | source | identity | 1.0 | — |
| PSF Convolution | lens | convolution | 0.95 | — |
| Detector | detector | integration | 0.8 | poisson, gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [64, 64] |
| Measurement shape (y) | [64, 64] |
| Forward model type | linear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Sigma | 2.0 |
| Read Noise E | 5.0 |
| Pixel Size Um | 6.5 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Gamma |
| Primary contrast | Determined by gamma-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
