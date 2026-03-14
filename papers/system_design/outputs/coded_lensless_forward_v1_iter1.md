---
modality: coded_lensless
period: forward
version: 1
iteration: 1
---

# Task

Design a coded lensless imaging system that combines a binary coded aperture with a phase diffuser for single-shot computational imaging without any lens.

# Plan

1. Configure broadband LED source for incoherent flood illumination
2. Define binary random coded aperture (50% fill factor, 256×256)
3. Model phase diffuser as convolution element with calibrated PSF
4. Configure bare CMOS sensor with Poisson + readout noise
5. Identify mask alignment and PSF calibration as primary mismatch sources

# Action

## System Flowchart

```
[LED Source] → [2D Object] → [Coded Aperture M] → [Phase Diffuser (PSF)] → [Bare CMOS] → [12-bit ADC] → y
                                    ↓                      ↓                     ↓
                              [Binary mask         [Convolution            [Poisson noise
                               M ⊙ x]              y = H * (M⊙x)]         + readout σ=3 e⁻]
```

### Element: Broadband LED (`source`)

- **Type**: source
- **Parameters**:
  - `wavelength_range_nm`: [400, 700]
  - `power_mw`: 100
  - `illumination`: incoherent_flood

### Element: Binary Coded Aperture (`modulation`)

- **Type**: modulation
- **Parameters**:
  - `type`: binary_random
  - `fill_factor`: 0.5
  - `pixel_pitch_um`: 5.5
  - `size`: [256, 256]
- **Forward model**: $x_{\text{coded}} = M \odot x$, where $M$ is the binary mask
- **Mismatch sources**:
  - `mask_misalignment` [medium]: Lateral shift between mask and sensor → correction: cross-correlation registration

### Element: Phase Diffuser (`diffuser`)

- **Type**: optics (convolution)
- **Parameters**:
  - `type`: diffuse_caustic
  - `psf_support_px`: [256, 256]
  - `model`: convolution
- **Forward model**: $y = H \ast (M \odot x) + n$, where $H$ is the diffuser PSF
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

## Composite Forward Model

```
y = H * (M ⊙ x) + n
```

where $H$ is the diffuser PSF, $M$ is the binary coded aperture, $\odot$ is element-wise multiplication, $\ast$ is 2D convolution, and $n \sim \text{Poisson}(H \ast (M \odot x)) + \mathcal{N}(0, \sigma^2)$.

**Canonical chain**: $M \to C \to D$ (Modulate → Convolve → Detect)

**Measurement shape**: `(256, 256)` — same spatial dimensions as object

**Key advantage over plain lensless**: The coded aperture adds measurement diversity, improving the conditioning of the inverse problem. The combined operator $H \cdot \text{diag}(M)$ has better spectral properties than $H$ alone.

## spec.md

```
modality: coded_lensless
carrier: photon
geometry: single_shot, coded_aperture + diffuser
object: 2D planar scene, 256x256
forward_model: M(coded_aperture) -> C(PSF) -> D(intensity)
noise: Poisson(5000) + Gaussian(sigma=3e-)
target: PSNR >= 10 dB, SSIM >= 0.2
system_elements: source=broadband LED 100mW, optics=binary mask + phase diffuser, detector=CMOS 256x256 5.5um
```

# Demands

- **feasibility**: yes
- **budget_feasible**: yes (< $600 total system cost)
- **algorithm_convergence**: N/A (forward period)

**Comments**: Coded lensless improves on plain lensless by adding a binary coded aperture before the diffuser. The mask modulation provides measurement diversity that improves the conditioning of the deconvolution problem. Expected PSNR improvement of 4-7 dB over plain lensless (from ~8 dB to ~12-15 dB). The same chain (M→C→D) as SIM but with completely different physics: detection-side mask modulation vs illumination-side structured patterns.
