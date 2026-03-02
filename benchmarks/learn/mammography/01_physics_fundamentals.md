# 01 — Physics Fundamentals: Mammography

## 1. Overview

Mammography uses low-energy X-rays (typically 25-35 kVp) to image breast tissue. The low photon energy maximizes contrast between soft tissue types. Specialized flat-panel detectors with direct or indirect conversion provide high spatial resolution for detecting microcalcifications and masses.

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
| Molybdenum / Rhodium X-ray Tube | source | identity | 1.0 | — |
| Compressed Breast Tissue | medium | projection | 0.2 | shot_poisson |
| Digital Mammography Detector | detector | integration | 0.85 | shot_poisson, read_gaussian |

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
| Tube Voltage Kvp | 28 |
| Tube Current Ma | 100 |
| Anode Material | Mo/Rh |
| Focal Spot Mm | 0.3 |
| Compressed Thickness Mm | 50 |
| Mean Attenuation Coefficient | 0.6 |
| Pixel Size Um | 70 |
| Detector Type | amorphous_selenium |
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
