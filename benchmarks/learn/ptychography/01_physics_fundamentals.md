# 01 — Physics Fundamentals: Ptychographic Imaging

## 1. Overview

Ptychography is a coherent diffractive imaging technique that scans a localized probe beam across the sample with overlapping positions. At each position, the far-field diffraction intensity is recorded. The forward model for position j is I_j = |F(P * O_j)|^2 where P is the probe function, O_j is the sample exit wave at position j, and F is the Fourier transform. Iterative phase retrieval algorithms (ePIE, PIE) or neural networks (PtychoNN) recover both amplitude and phase of the sample.

**Category**: Coherent Imaging
**Carrier**: Electron/Photon

---

## 2. Electron/Photon Physics

Coherent diffractive imaging uses coherent beams (electrons or photons) and records the far-field diffraction pattern. Since detectors measure only intensity (not phase), phase retrieval algorithms are needed to reconstruct the complex object.

### Key Concepts

- Coherent illumination and Fraunhofer diffraction
- Phase problem: detectors measure |F{ψ}|² only
- Oversampling requirement (>2× Nyquist)
- Iterative phase retrieval: HIO, ER, RAAR
- Support constraint and positivity

### Wavelength / Energy Range

0.01 – 0.5 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
I(q) = |F{ψ(r)}|²  (phase lost)
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| Coherent X-ray or Electron Source | source | identity | 1.0 | — |
| Zone Plate / Fresnel Lens (Probe Former) | lens | convolution | 0.15 | aberration |
| Scanning Stage | modulator | sampling | 1.0 | alignment |
| Sample | sample | modulation | 0.8 | — |
| Pixel Array Detector | detector | integration | 0.9 | shot_poisson, read_gaussian |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256] |
| Measurement shape (y) | [16, 128, 128] |
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | photon_counting |
| Photon Energy Kev | 8.0 |
| Wavelength Nm | 0.155 |
| Coherence Length Um | 10 |
| Outermost Zone Nm | 50 |
| Diameter Um | 100 |
| Focal Length Mm | 25 |
| N Positions | 16 |
| Step Size Um | 2.0 |
| Overlap Fraction | 0.6 |
| Scan Pattern | raster |
| Max Phase Shift Rad | 1.5 |
| Max Absorption | 0.3 |
| Pixel Size Um | 75 |
| N Pixels | [128, 128] |
| Dynamic Range | 1000000 |
| Bit Depth | 24 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Electron/Photon |
| Primary contrast | Determined by electron/photon-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
