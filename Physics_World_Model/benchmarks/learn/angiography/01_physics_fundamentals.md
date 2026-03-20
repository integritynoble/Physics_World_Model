# 01 — Physics Fundamentals: X-ray Angiography

## 1. Overview

X-ray angiography is a real-time fluoroscopic technique using iodine-based contrast agents to visualize blood vessels. Digital subtraction angiography (DSA) subtracts a pre-contrast mask image from the contrast-enhanced frame to isolate vascular structures. The forward model combines Beer-Lambert attenuation with contrast agent dynamics and temporal integration.

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
| X-ray Tube (Pulsed) | source | identity | 1.0 | — |
| Tissue + Contrast Agent | medium | projection | 0.1 | shot_poisson |
| Flat Panel Detector | detector | integration | 0.8 | shot_poisson, read_gaussian |

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
| Tube Current Ma | 200 |
| Focal Spot Mm | 0.6 |
| Mode | pulsed |
| Mean Attenuation Coefficient | 0.2 |
| Contrast Agent | iodine |
| Contrast Concentration Mg Per Ml | 350 |
| Pixel Size Um | 200 |
| Frame Rate Fps | 30 |
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
