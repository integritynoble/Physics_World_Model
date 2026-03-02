# 01 — Physics Fundamentals: Neutron Diffraction

## 1. Overview

Neutron Diffraction imaging modality with DAG: R --> S --> D.

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

0 – 0 nm

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
| Scattering | medium | modulation | 0.7 | — |
| Detector | detector | integration | 0.8 | poisson |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [64, 64] |
| Measurement shape (y) | [64, 64] |
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

No specific physics parameters defined.


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
