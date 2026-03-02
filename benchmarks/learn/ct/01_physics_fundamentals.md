# 01 — Physics Fundamentals: X-ray Computed Tomography (CT)

## 1. Overview

X-ray Computed Tomography acquires a sinogram of projection measurements at multiple angles around the patient. Each projection is a line integral of the attenuation coefficient along X-ray paths (Radon transform). The inverse problem recovers the 2D attenuation map from the sinogram. Filtered Back-Projection (FBP) is the classical solver; iterative methods (SART) and learned priors (RED-CNN) improve low-dose quality.

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

### Wavelength / Energy Range

0 – 0 nm

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
| X-ray Tube | source | identity | 1.0 | — |
| Beam Collimator | filter | modulation | 0.85 | — |
| Patient (Attenuating Medium) | medium | projection | 0.1 | shot_poisson |
| Anti-scatter Grid | filter | modulation | 0.7 | — |
| Scintillator + Photodiode Array Detector | detector | integration | 0.85 | shot_poisson, read_gaussian, quantization |

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
| Tube Voltage Kvp | 120 |
| Tube Current Ma | 200 |
| Focal Spot Mm | 0.6 |
| Spectrum | polychromatic |
| Type | fan_beam |
| Fan Angle Deg | 50 |
| Slice Thickness Mm | 1.0 |
| Mean Attenuation Coefficient | 0.2 |
| N Angles | 180 |
| Angular Range Deg | 180 |
| Grid Ratio | 12 |
| N Detectors | 512 |
| Detector Pitch Mm | 1.0 |
| Scintillator | CsI |
| Bit Depth | 20 |


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
