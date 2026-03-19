# 01 — Physics Fundamentals: Dual-Energy X-ray Absorptiometry (DEXA)

## 1. Overview

Dual-energy X-ray absorptiometry acquires projections at two X-ray energies to decompose tissue into bone mineral and soft tissue components. The dual-energy Beer-Lambert model separates the contributions of two material bases. DEXA is the clinical standard for bone mineral density measurement.

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
| Dual-Energy X-ray Source | source | identity | 1.0 | — |
| Patient (Bone + Soft Tissue) | medium | projection | 0.25 | shot_poisson |
| Multi-Element Detector Array | detector | integration | 0.85 | shot_poisson, read_gaussian |

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
| Low Energy Kvp | 40 |
| High Energy Kvp | 70 |
| Tube Current Ma | 5 |
| Bone Attenuation Low | 0.8 |
| Bone Attenuation High | 0.3 |
| Soft Tissue Attenuation | 0.2 |
| Pixel Size Um | 350 |
| Detector Type | CdZnTe |
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
