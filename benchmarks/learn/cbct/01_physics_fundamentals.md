# 01 — Physics Fundamentals: Cone-Beam Computed Tomography (CBCT)

## 1. Overview

Cone-beam CT acquires a volumetric dataset from a single rotation using a 2D flat-panel detector and a cone-shaped X-ray beam. It trades off lower dose and compact geometry for increased scatter artifacts and reduced soft-tissue contrast compared to clinical fan-beam CT. Applications include dental, head-and-neck, and image-guided radiation therapy.

**Category**: Medical Imaging
**Carrier**: X-ray

---

## 2. X-ray Physics

X-rays are high-energy electromagnetic radiation (photon energies ~20-150 keV) that penetrate matter and are attenuated according to Beer-Lambert law: I = I₀ exp(-∫μ(x,y,z) dl). Different tissues have different linear attenuation coefficients μ, creating contrast.

### Key Concepts

- Beer-Lambert attenuation law
- Linear attenuation coefficient μ (energy-dependent)
- Polychromatic spectrum and beam hardening
- Detector types: scintillator + photodiode, photon-counting
- Dose considerations: ALARA principle

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
I(d) = I₀ · exp(-∫ μ(l) dl) + noise
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| X-ray Tube (Cone Beam) | source | identity | 1.0 | — |
| Patient (Attenuating Medium) | medium | projection | 0.12 | shot_poisson, scatter |
| Flat Panel Detector | detector | integration | 0.8 | shot_poisson, read_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256] |
| Measurement shape (y) | [180, 256] |
| Forward model type | linear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Tube Voltage Kvp | 110 |
| Tube Current Ma | 10 |
| Focal Spot Mm | 0.5 |
| Beam Geometry | cone |
| Mean Attenuation Coefficient | 0.2 |
| N Angles | 180 |
| Pixel Size Um | 200 |
| Scintillator | CsI |
| Bit Depth | 14 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | X-ray |
| Primary contrast | Determined by x-ray-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
