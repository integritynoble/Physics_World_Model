---
modality: lensless_3d
period: forward
version: 1
iteration: 1
---

# Task

Design a single-shot 3D lensless imaging system that recovers Nz=8 depth planes from a single 2D diffuser camera measurement. The system exploits the depth-dependent PSF of a phase diffuser to encode 3D information into a single 2D image.

# Plan

1. Configure broadband LED source for incoherent flood illumination
2. Model 3D volumetric object with Nz=8 depth planes
3. Model phase diffuser with depth-dependent PSFs (PSF changes with object distance)
4. Define depth accumulation (detector integrates light from all depth planes)
5. Configure bare CMOS sensor with Poisson + readout noise

# Action

## System Flowchart

```
[LED Source] -> [3D Object x_z] -> [Phase Diffuser H_z] -> [Accumulate] -> [CMOS] -> y
                 (Nz=8 planes)    (depth-dependent PSF)   (sum over z)
                                         |                      |
                                   [H_z * x_z]          [Sum_z + noise]
```

### Element: Broadband LED (`source`)

- **Type**: source
- **Parameters**:
  - `wavelength_range_nm`: [400, 700]
  - `power_mw`: 100
  - `illumination`: incoherent_flood

### Element: Phase Diffuser (`diffuser`)

- **Type**: optics (depth-varying convolution)
- **Parameters**:
  - `type`: diffuse_caustic
  - `psf_support_px`: [256, 256]
  - `model`: depth_varying_convolution
  - `n_depth_planes`: 8
  - `depth_range_mm`: [30, 100]
- **Forward model**: $u_z = H_z \ast x_z$ for each depth plane $z$
- **Key physics**: The diffuser PSF changes with object distance due to:
  - Defocus: PSF spread increases with distance from calibration plane
  - Magnification: lateral shift of caustic pattern
  - These depth-dependent changes provide the diversity needed for 3D recovery
- **Mismatch sources**:
  - `psf_calibration_error` [medium]: Per-depth PSF measurement error -> correction: per-depth point source calibration
  - `depth_discretization` [low]: Continuous depth approximated by discrete planes -> correction: finer depth sampling

### Element: Depth Accumulation (`accumulator`)

- **Type**: processing
- **Parameters**:
  - `type`: sum
  - `n_depth_planes`: 8
- **Forward model**: $y = \sum_{z=1}^{N_z} H_z \ast x_z$

### Element: Bare CMOS Sensor (`detector`)

- **Type**: detector
- **Parameters**:
  - `pixels`: [256, 256]
  - `pixel_pitch_um`: 5.5
  - `quantum_efficiency`: 0.6
- **Noise**:
  - poisson: mean_photons=5000
  - gaussian: sigma_electrons=3.0

## Composite Forward Model

```
y = sum_{z=1}^{Nz} H_z * x_z + n,  Nz = 8
```

where $H_z$ is the depth-dependent diffuser PSF for plane $z$, $x_z$ is the object at depth $z$, $\ast$ is 2D convolution, and $n \sim \text{Poisson} + \mathcal{N}(0, \sigma^2)$.

**Canonical chain**: $C \to \Sigma \to D$ (Convolve-per-depth -> Accumulate -> Detect)

**Measurement shape**: `(256, 256)` -- single 2D measurement

**Object shape**: `(8, 256, 256)` -- 8 depth planes to recover

**Compression ratio**: 8:1 (8 depth planes compressed into 1 measurement)

## spec.md

```
modality: lensless_3d
carrier: photon
geometry: single_shot_3d, Nz=8 depth planes
object: 3D volume, 8x256x256
forward_model: C(depth-varying PSF) -> Sigma(sum) -> D(intensity)
noise: Poisson(5000) + Gaussian(sigma=3e-)
target: PSNR >= 15 dB, SSIM >= 0.4
system_elements: source=broadband LED 100mW, optics=phase diffuser, detector=CMOS 256x256 5.5um
```

# Demands

- **feasibility**: yes
- **budget_feasible**: yes (< $500 total system cost -- simplest possible lensless camera)
- **algorithm_convergence**: N/A (forward period)

**Comments**: This is the simplest lensless system that captures genuinely new information -- 3D depth from a single 2D shot. The depth-dependent PSF variation provides the measurement diversity needed for volumetric recovery. Unlike coded lensless (M->C->D) which adds a mask but records the same information as plain lensless, 3D lensless exploits the inherent physics of the diffuser to encode an additional spatial dimension. The 8:1 compression ratio (8 depth planes -> 1 measurement) makes this a genuine compressive imaging problem, analogous to temporal-coded lensless (time) and spectral lensless (wavelength).
