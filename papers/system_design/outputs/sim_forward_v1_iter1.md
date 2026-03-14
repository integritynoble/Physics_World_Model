---
modality: sim
period: forward
version: 1
iteration: 1
---

# Task

Design a structured illumination microscopy (SIM) system with 3 orientations and 3 phase shifts per orientation. Specify the forward model, select appropriate reconstruction algorithm, and produce reconstructed images.

# Plan

1. Configure 488 nm laser source for structured illumination
2. Define sinusoidal illumination patterns: 3 orientations × 3 phases = 9 raw frames
3. Model fluorescent sample interaction with emission at 520 nm
4. Configure high-NA objective (1.4 NA, oil immersion) as PSF convolution
5. Configure sCMOS detector with Poisson + readout noise
6. Identify pattern frequency/phase errors as primary mismatch sources

# Action

## System Flowchart

```
[488nm Laser] → [SLM: 3 orient. × 3 phases] → [Fluorescent Sample]
                                                        ↓
                                              [1.4 NA Objective (PSF)]
                                                        ↓
                                               [sCMOS Detector] → [16-bit ADC] → y
                                                        ↓
                                                [Poisson + readout σ=1.5 e⁻]
```

### Element: Laser Source (`source`)

- **Type**: source
- **Parameters**:
  - `wavelength_nm`: 488
  - `power_mw`: 50
  - `coherence`: partially_coherent

### Element: Structured Illumination Patterns (`modulation`)

- **Type**: modulation
- **Parameters**:
  - `type`: sinusoidal_grating
  - `n_orientations`: 3
  - `n_phases`: 3
  - `pattern_period_nm`: 200
  - `modulation_depth`: 0.9
- **Forward model**: $y_k = \text{PSF} \ast (I_k \cdot x) + n_k$ for $k=1,...,9$
- **Mismatch sources**:
  - `pattern_frequency_error` [medium]: Grating frequency deviates from design → correction: Fourier peak fitting
  - `pattern_phase_error` [medium]: Phase steps deviate from 0°/120°/240° → correction: iterative parameter estimation

### Element: High-NA Objective (`objective`)

- **Type**: optics (convolution)
- **Parameters**:
  - `magnification`: 100
  - `NA`: 1.4
  - `immersion`: oil
  - `psf_model`: airy_disk
- **Mismatch sources**:
  - `optical_aberration` [low]: Residual spherical/chromatic aberration → correction: measured PSF deconvolution

### Element: sCMOS Camera (`detector`)

- **Type**: detector
- **Parameters**:
  - `pixels`: [256, 256]
  - `pixel_size_um`: 6.5
  - `quantum_efficiency`: 0.82
- **Noise**:
  - poisson: mean_photons=10000
  - gaussian: sigma_electrons=1.5

## Composite Forward Model

```
y_k = PSF * (I_k · x) + n_k,  k = 1,...,9
y_sum = sum(y_k)  (blurred sum image for simplified reconstruction)
```

where $I_k$ are sinusoidal illumination patterns, $\ast$ is 2D convolution with the system PSF, and $n_k$ is per-frame noise.

**Canonical chain**: $M \to C \to D$ (Modulation → Convolution → Detect)

**Measurement shape**: raw frames `(9, 256, 256)`, sum image `(256, 256)`

## spec.md

```
modality: sim
carrier: photon
geometry: multi_frame, 3 orientations × 3 phases = 9 frames
object: 2D fluorescent sample, 256x256
forward_model: M(sinusoidal) -> C(PSF) -> D(intensity)
noise: Poisson(10000) + Gaussian(sigma=1.5e-)
target: PSNR >= 25 dB, SSIM >= 0.7
system_elements: source=488nm laser 50mW, optics=1.4NA oil objective + SLM, detector=sCMOS 256x256 6.5um
```

# Demands

- **feasibility**: yes
- **budget_feasible**: yes (~$150k system)
- **algorithm_convergence**: N/A (forward period)

**Comments**: SIM achieves ~2× resolution improvement beyond diffraction limit. 9 raw frames provide sufficient information for frequency-domain reconstruction (Wiener-SIM) or iterative approaches (FISTA+TV, PnP+TV). Pattern parameter estimation is critical for artifact-free reconstruction.
