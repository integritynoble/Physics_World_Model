---
modality: lensless
period: forward
version: 1
iteration: 1
---

# Task

Design a lensless imaging system using a diffuser or coded mask. The camera captures a single measurement through a known PSF. Specify the forward model, select appropriate reconstruction algorithm, and produce reconstructed images.

# Plan

1. Configure broadband LED source for incoherent flood illumination
2. Model 2D planar object at fixed distance from sensor
3. Define phase diffuser as convolution element with calibrated PSF
4. Configure bare CMOS sensor with Poisson + readout noise
5. Identify PSF calibration error and depth-dependent PSF as mismatch sources

# Action

## System Flowchart

```
[LED Source] → [2D Object] → [Phase Diffuser (PSF)] → [Bare CMOS] → [12-bit ADC] → y
                                    ↓                      ↓
                              [Convolution            [Poisson noise
                               y = H * x]              + readout σ=3 e⁻]
```

### Element: Broadband LED (`source`)

- **Type**: source
- **Parameters**:
  - `wavelength_nm_range`: [400, 700]
  - `power_mw`: 100
  - `illumination`: incoherent_flood
- **Mismatch sources**: none

### Element: Phase Diffuser (`diffuser`)

- **Type**: optics (convolution)
- **Parameters**:
  - `type`: diffuser
  - `psf_type`: diffuse_caustic
  - `psf_support_px`: [256, 256]
  - `model`: convolution
- **Forward model**: $y = H \ast x + n$, where $H$ is the measured PSF
- **Mismatch sources**:
  - `psf_calibration_error` [medium]: Measured PSF differs from true PSF → correction: in-situ point source calibration
  - `depth_dependent_psf` [low]: PSF varies with object distance → correction: depth-sectioned reconstruction

### Element: Bare CMOS Sensor (`detector`)

- **Type**: detector
- **Parameters**:
  - `pixels`: [256, 256]
  - `pixel_pitch_um`: 5.5
  - `quantum_efficiency`: 0.6
- **Noise**:
  - poisson: mean_photons=5000
  - gaussian: sigma_electrons=3.0

### Element: 12-bit ADC (`adc`)

- **Type**: digitization
- **Parameters**:
  - `bit_depth`: 12

## Composite Forward Model

```
y = H * x + n
```

where $H$ is the system PSF (diffuser caustic pattern), $\ast$ denotes 2D convolution, and $n \sim \text{Poisson}(H \ast x) + \mathcal{N}(0, \sigma^2)$.

**Canonical chain**: $C \to D$ (Convolution → Detect)

**Measurement shape**: `(256, 256)` — same spatial dimensions as object

## spec.md

```
modality: lensless
carrier: photon
geometry: single_shot, distance=50mm
object: 2D planar scene, 256x256
forward_model: C(PSF) -> D(intensity)
noise: Poisson(5000) + Gaussian(sigma=3e-)
target: PSNR >= 8 dB, SSIM >= 0.1
system_elements: source=broadband LED 100mW, optics=phase diffuser, detector=CMOS 256x256 5.5um
```

# Demands

- **feasibility**: yes
- **budget_feasible**: yes (< $500 total system cost)
- **algorithm_convergence**: N/A (forward period)

**Comments**: Lensless imaging has extremely diffuse PSF — deconvolution is severely ill-conditioned. Expected SNR ~15 dB at detector but reconstructed PSNR limited to ~8 dB by PSF conditioning. Wiener deconvolution or FISTA+TV recommended.
