# 01 — Physics Fundamentals: Atomic Force Microscopy (AFM)

## 1. Overview

Atomic Force Microscopy (AFM) imaging modality with DAG: S --> D.

**Category**: Scanning Probe Microscopy
**Carrier**: Mechanical

---

## 2. Mechanical Physics

Mechanical probe imaging uses physical contact between a sharp tip and the sample surface. Forces (van der Waals, electrostatic, magnetic) are measured as the tip scans across the surface. The tip-sample interaction provides topographic and material property information with nanometre resolution.

### Key Concepts

- Tip-sample interaction forces
- Contact, tapping, and non-contact modes
- Cantilever dynamics and resonance
- Piezoelectric scanning and feedback
- Tip convolution artefact

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
z(x,y) = h(x,y) ⊛ tip(x,y) + noise
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
| Physical probe | Mechanical |
| Primary contrast | Determined by mechanical-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
