# 01 — Physics Fundamentals: Muon Tomography

## 1. Overview

Muon tomography uses cosmic-ray muons to image the density distribution inside large objects by measuring scattering angles related to the material radiation length.

**Category**: Scientific Instrumentation
**Carrier**: Muon

---

## 2. Muon Physics

Muon tomography uses cosmic-ray muons that scatter as they pass through dense materials. By tracking incoming and outgoing muon trajectories, the scattering density (related to atomic number Z) can be reconstructed.

### Key Concepts

- Cosmic ray muon flux (~1 muon/cm²/min at sea level)
- Multiple Coulomb scattering
- Scattering angle and radiation length
- Point of closest approach (POCA) reconstruction
- Maximum likelihood / expectation maximisation

### Wavelength / Energy Range

0.0001 – 0.001 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
θ_rms = (13.6 MeV / pβc) · √(x/X₀) · [1 + 0.038 ln(x/X₀)]
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| Cosmic Ray Muon Source | source | identity | 1.0 | — |
| Scintillator / Drift Tube Tracker | detector | integration | 0.95 | position_resolution, timing_jitter |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [64, 64, 64] |
| Measurement shape (y) | [1024] |
| Forward model type | nonlinear_operator |
| Category module | medical_ct_radon |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | cosmic_ray |
| Mean Energy Gev | 3.0 |
| N Layers | 8 |
| Detection Efficiency | 0.95 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Muon |
| Primary contrast | Determined by muon-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
