# 01 — Physics Fundamentals: Terahertz Imaging (THz)

## 1. Overview

Terahertz Imaging (THz) imaging modality with DAG: P --> D.

**Category**: Industrial Inspection
**Carrier**: THz

---

## 2. THz Physics

Terahertz (THz) imaging uses electromagnetic waves at 0.1-10 THz, bridging the gap between microwave and infrared. THz waves penetrate many non-metallic materials (plastics, ceramics, textiles) while being strongly absorbed by water, providing unique contrast.

### Key Concepts

- THz generation: photoconductive, optical rectification
- Time-domain spectroscopy (THz-TDS)
- Material-specific absorption signatures
- Penetration depth and water sensitivity
- Sub-wavelength resolution techniques

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
E(t) = E₀ · h(t) ⊛ sample_response(t) + noise
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| Source/Emitter | source | propagation | 1.0 | — |
| Detector | detector | integration | 0.8 | gaussian |

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
| Physical probe | THz |
| Primary contrast | Determined by thz-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
