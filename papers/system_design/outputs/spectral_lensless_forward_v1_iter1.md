---
modality: spectral_lensless
period: forward
version: 1
iteration: 1
---

# Task

Design a snapshot spectral lensless imaging system that recovers L=8 spectral bands from a single coded+dispersed lensless measurement. The system combines a binary coded aperture, a dispersive element (prism/grating), and a phase diffuser for single-shot hyperspectral imaging without any lens.

# Plan

1. Configure broadband LED source for incoherent flood illumination
2. Define binary random coded aperture (50% fill factor, 256x256)
3. Model dispersive element (prism) with wavelength-dependent lateral shift
4. Model phase diffuser as convolution element with calibrated PSF
5. Define spectral accumulation (detector integrates all L dispersed coded bands)
6. Configure bare CMOS sensor with Poisson + readout noise
7. Identify mask alignment, dispersion calibration, and PSF calibration as primary mismatch sources

# Action

## System Flowchart

```
[LED Source] -> [Spectral Object x_b] -> [Coded Mask M] -> [Prism W_b] -> [Diffuser H] -> [Accumulate] -> [CMOS] -> y
                  (L=8 bands)           (binary random)    (lateral shift) (PSF convolve)  (sum over b)
                                               |                |               |               |
                                         [M . x_b]      [W_b(M . x_b)]  [H * W_b(M . x_b)]  [Sum + noise]
```

### Element: Broadband LED (`source`)

- **Type**: source
- **Parameters**:
  - `wavelength_range_nm`: [400, 700]
  - `power_mw`: 150
  - `illumination`: incoherent_flood

### Element: Binary Coded Aperture (`modulation`)

- **Type**: modulation
- **Parameters**:
  - `type`: binary_random
  - `fill_factor`: 0.5
  - `pixel_pitch_um`: 5.5
  - `size`: [256, 256]
- **Forward model**: $x_b^{\text{coded}} = M \odot x_b$
- **Mismatch sources**:
  - `mask_misalignment` [medium]: Lateral shift between mask and sensor -> correction: cross-correlation registration

### Element: Dispersive Prism (`dispersion`)

- **Type**: optics (shift-variant)
- **Parameters**:
  - `type`: prism_BK7
  - `dispersion_nm_per_px`: 37.5
  - `max_shift_px`: 20
  - `n_bands`: 8
- **Forward model**: $z_b = W_b(M \odot x_b)$ where $W_b$ is a wavelength-dependent lateral shift
- **Mismatch sources**:
  - `dispersion_calibration_error` [medium]: Wavelength-to-shift mapping differs from nominal -> correction: spectral calibration with narrowband sources

### Element: Phase Diffuser (`diffuser`)

- **Type**: optics (convolution)
- **Parameters**:
  - `type`: diffuse_caustic
  - `psf_support_px`: [256, 256]
  - `model`: convolution
- **Forward model**: $u_b = H \ast W_b(M \odot x_b)$
- **Mismatch sources**:
  - `psf_calibration_error` [medium]: Measured PSF differs from true PSF -> correction: in-situ calibration

### Element: Spectral Accumulation (`accumulator`)

- **Type**: processing
- **Parameters**:
  - `type`: sum
  - `n_bands`: 8
- **Forward model**: $y = \sum_{b=1}^{L} H \ast W_b(M \odot x_b)$

### Element: Bare CMOS Sensor (`detector`)

- **Type**: detector
- **Parameters**:
  - `pixels`: [256, 256]
  - `pixel_pitch_um`: 5.5
  - `quantum_efficiency`: 0.55
- **Noise**:
  - poisson: mean_photons=2000
  - gaussian: sigma_electrons=3.0

## Composite Forward Model

```
y = sum_{b=1}^{L} H * W_b(M . x_b) + n,  L = 8
```

where $H$ is the diffuser PSF, $W_b$ is the wavelength-dependent dispersion shift for band $b$, $M$ is the binary coded aperture, $\odot$ is element-wise multiplication, $\ast$ is 2D convolution, and $n \sim \text{Poisson} + \mathcal{N}(0, \sigma^2)$.

**Canonical chain**: $M \to W \to C \to \Sigma \to D$ (Modulate -> Disperse -> Convolve -> Accumulate -> Detect)

**Measurement shape**: `(256, 256)` -- single 2D measurement

**Object shape**: `(8, 256, 256)` -- 8 spectral bands to recover

**Compression ratio**: 8:1 (8 bands compressed into 1 measurement)

## spec.md

```
modality: spectral_lensless
carrier: photon
geometry: single_shot_spectral, L=8 bands
object: hyperspectral cube, 8x256x256
forward_model: M(coded_aperture) -> W(dispersion) -> C(PSF) -> Sigma(sum) -> D(intensity)
noise: Poisson(2000) + Gaussian(sigma=3e-)
target: PSNR >= 15 dB, SSIM >= 0.4
system_elements: source=broadband LED 150mW, optics=binary mask + BK7 prism + phase diffuser, detector=CMOS 256x256 5.5um
```

# Demands

- **feasibility**: yes
- **budget_feasible**: yes (< $1200 total system cost)
- **algorithm_convergence**: N/A (forward period)

**Comments**: This system is a novel 5-primitive composition M->W->C->Sigma->D that has not appeared in prior literature as a single system. It combines three independently-studied concepts: coded aperture imaging (M), dispersive spectral encoding (W), and lensless diffuser imaging (C->D). The dispersion provides spectral diversity (each band shifts differently on the detector), the coded mask provides spatial diversity, and the diffuser eliminates the need for imaging optics. The 8:1 spectral compression makes recovery challenging but tractable with GAP-TV or FISTA methods. Expected PSNR: 15-20 dB per band.
