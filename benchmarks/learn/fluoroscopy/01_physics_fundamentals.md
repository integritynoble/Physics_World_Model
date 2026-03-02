# 01 — Physics Fundamentals: Fluoroscopy

## 1. Overview

Fluoroscopy provides real-time X-ray imaging by acquiring rapid sequences of low-dose radiographs. Temporal integration over multiple frames reduces noise at the expense of motion blur. The forward model is a Beer-Lambert projection with temporal frame averaging and Poisson-Gaussian noise.

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
| X-ray Tube (Continuous) | source | identity | 1.0 | — |
| Patient (Attenuating Medium) | medium | projection | 0.15 | shot_poisson |
| Image Intensifier / Flat Panel | detector | integration | 0.7 | shot_poisson, read_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [512, 512] |
| Measurement shape (y) | [512, 512] |
| Forward model type | nonlinear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Tube Voltage Kvp | 80 |
| Tube Current Ma | 3 |
| Focal Spot Mm | 0.6 |
| Mode | continuous |
| Mean Attenuation Coefficient | 0.2 |
| Pixel Size Um | 200 |
| Frame Rate Fps | 30 |
| Scintillator | CsI |
| Bit Depth | 12 |


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
