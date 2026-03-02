# 01 — Physics Fundamentals: Particle Calorimetry

## 1. Overview

Particle Calorimetry imaging modality with DAG: R --> Sigma --> D.

**Category**: Broader Experimental Science
**Carrier**: Particle

---

## 2. Particle Physics

Particle calorimetry measures the energy of particles (electrons, photons, hadrons) by total absorption. Electromagnetic and hadronic showers develop in dense materials, and the deposited energy is proportional to the incident particle energy.

### Key Concepts

- Electromagnetic and hadronic showers
- Radiation length and interaction length
- Energy resolution: σ_E/E = a/√E ⊕ b ⊕ c/E
- Sampling vs homogeneous calorimeters
- Particle identification via shower shape

### Wavelength / Energy Range

0 – 0 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
E_measured = Σ_cells  w_i · E_cell_i + noise
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
| Physical probe | Particle |
| Primary contrast | Determined by particle-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
