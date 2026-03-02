# 01 — Physics Fundamentals: Positron Emission Tomography (PET)

## 1. Overview

PET detects coincident 511 keV annihilation photon pairs from a positron-emitting radiotracer. The system matrix projects the activity distribution into sinogram space. Attenuation correction, scatter estimation, and MLEM/OSEM reconstruction are standard.

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
| Radiotracer Source (positron emitter) | source | identity | 1.0 | — |
| Patient (Attenuating Medium) | medium | projection | 0.3 | attenuation, scatter |
| Scintillator Ring Detector (LSO/LYSO) | detector | integration | 0.85 | shot_poisson, random_coincidences |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [128, 128] |
| Measurement shape (y) | [180, 128] |
| Forward model type | linear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Isotope | F-18 |
| Half Life Min | 109.77 |
| Activity Mbq | 370 |
| Mean Attenuation Coefficient | 0.096 |
| N Rings | 4 |
| N Detectors Per Ring | 512 |
| Crystal | LYSO |
| Timing Resolution Ps | 300 |


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
