# 01 — Physics Fundamentals: Coherent Diffractive Imaging / Phase Retrieval

## 1. Overview

Coherent Diffractive Imaging (CDI) recovers both amplitude and phase of a specimen from a single far-field diffraction intensity pattern. A coherent source (X-ray or laser) illuminates the sample, and the scattered wavefield propagates to a detector in the far field without any intervening lens. The measured quantity is y = |F{x}|^2 where F is the Fourier transform; the phase information is lost. The phase retrieval problem is solved by iterative projection algorithms (Hybrid Input-Output, Error Reduction, RAAR) that alternate between enforcing the measured Fourier magnitude and a real-space support constraint. Sufficient oversampling (at least 2x Nyquist) is required for unique recovery.

**Category**: Coherent Imaging
**Carrier**: Photon/Electron

---

## 2. Photon/Electron Physics

This modality uses a specific physical probe to interact with the sample or scene. The interaction produces measurements that encode information about the object's internal structure or surface properties.

### Key Concepts

- Probe-sample interaction mechanism
- Forward model relating object to measurements
- Noise model (signal-dependent and independent)
- Spatial resolution and field of view
- Contrast mechanism and sensitivity

### Wavelength / Energy Range

0.01 – 700 nm

---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
y = A(x) + noise
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain

### Imaging Chain Elements

| Element | Type | Transfer | Throughput | Noise |
|---------|------|----------|------------|-------|
| Coherent Source (X-ray or Laser) | source | identity | 1.0 | — |
| Sample | sample | modulation | 0.8 | — |
| Free-Space Propagation (Far Field) | medium | propagation | 1.0 | — |
| Pixel Array Detector (No Lens) | detector | integration | 0.85 | shot_poisson, read_gaussian, quantization |

### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | [256, 256] |
| Measurement shape (y) | [256, 256] |
| Forward model type | nonlinear_operator |
| Category module | microscopy_psf |

---

## 5. Key Physics Parameters

| Parameter | Value |
|-----------|-------|
| Type | photon_counting |
| Wavelength Nm | 0.155 |
| Coherence Length Um | 50 |
| Beam Diameter Um | 10 |
| Source Options | ['synchrotron_xray', 'xfel', 'HeNe_laser'] |
| Max Phase Shift Rad | 2.0 |
| Max Absorption | 0.5 |
| Oversampling Ratio | 2.0 |
| Support Diameter Px | 128 |
| Propagation Type | far_field |
| Fresnel Number | 0.001 |
| Sample To Detector Mm | 500 |
| Pixel Size Um | 75 |
| N Pixels | [256, 256] |
| Dynamic Range | 1000000 |
| Bit Depth | 24 |


---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | Photon/Electron |
| Primary contrast | Determined by photon/electron-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
