# 01 — Physics Fundamentals: Neutron Radiography / Tomography

## 1. Overview

Neutron radiography and tomography use thermal or cold neutrons to image the attenuation distribution inside objects. Neutrons interact strongly with light elements (hydrogen, lithium, boron) and penetrate heavy metals. The forward model follows Beer-Lambert law.

**Category**: Scientific Instrumentation
**Carrier**: Neutron

---

## 2. Neutron Physics

Neutron imaging uses thermal or cold neutrons (wavelength ~0.1-1 nm) that interact with nuclei rather than electrons. This gives complementary contrast to X-rays — light elements (H, Li, B) are strong absorbers, while heavy metals may be transparent.

### Key Concepts

- Nuclear cross-sections vs photon cross-sections
- Complementary contrast to X-rays
- Neutron sources: reactors, spallation
- Scintillator-based neutron detection
- Bragg edge imaging for crystallography

### Wavelength / Energy Range

0.05 – 0.5 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
I = I₀ · exp(-Σ_t · d) + scatter
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| Neutron Source (Reactor/Spallation) | source | identity | 1.0 | — |
| Scintillator + CCD Detector | detector | integration | 0.3 | shot_poisson, read_gaussian, gamma_background |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256] |
| Measurement shape (y) | [256, 256] |
| Forward model type | nonlinear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | reactor |
| Flux N Per Cm2 Per S | 10000000.0 |
| Scintillator | LiF/ZnS |
| Pixel Size Um | 50 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Neutron |
| Primary contrast | Determined by neutron-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
