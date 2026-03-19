# 01 — Physics Fundamentals: Single Photon Emission CT (SPECT)

## 1. Overview

SPECT images the distribution of a gamma-emitting radiotracer by acquiring projections through a parallel-hole collimator rotated around the patient. The collimator response function (depth-dependent blur) is a key component. MLEM/OSEM with attenuation and collimator-detector response modelling are standard reconstruction methods.

**Category**: Medical Imaging
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
| Radiotracer Source (gamma emitter) | source | identity | 1.0 | — |
| Parallel-Hole Collimator | filter | modulation | 0.01 | — |
| NaI(Tl) Gamma Camera | detector | integration | 0.85 | shot_poisson |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [128, 128] |
| Measurement shape (y) | [120, 128] |
| Forward model type | linear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Isotope | Tc-99m |
| Half Life Hr | 6.0 |
| Activity Mbq | 740 |
| Gamma Energy Kev | 140 |
| Hole Diameter Mm | 1.5 |
| Septal Thickness Mm | 0.2 |
| Hole Length Mm | 25 |
| Crystal Thickness Mm | 9.5 |
| Pixel Size Mm | 4.4 |
| Energy Resolution Percent | 10 |


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
