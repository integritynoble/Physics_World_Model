---
modality: temporal_coded_lensless
period: forward
version: 1
iteration: 1
---

# Task

Design a temporal-coded lensless video system that recovers T=8 video frames from a single coded lensless exposure. The system combines CACTI-style temporal coding with lensless (diffuser-based) imaging.

# Plan

1. Configure broadband LED source for incoherent flood illumination
2. Define temporal binary mask with T=8 switching patterns
3. Model phase diffuser as convolution element with calibrated PSF
4. Define temporal accumulation (detector integrates all T coded frames)
5. Configure bare CMOS sensor with Poisson + readout noise
6. Identify temporal synchronization and PSF calibration as primary mismatch sources

# Action

## System Flowchart

```
[LED Source] → [Moving Scene x_t] → [Temporal Mask m_t] → [Phase Diffuser] → [Accumulate] → [CMOS] → y
                  (T=8 frames)       (binary temporal)    (PSF convolution)   (sum over t)
                                            ↓                    ↓                 ↓
                                      [m_t ⊙ x_t]        [H * (m_t ⊙ x_t)]  [Σ_t + noise]
```

### Element: Broadband LED (`source`)

- **Type**: source
- **Parameters**:
  - `wavelength_range_nm`: [400, 700]
  - `power_mw`: 100
  - `illumination`: incoherent_flood

### Element: Temporal Binary Mask (`modulation`)

- **Type**: modulation
- **Parameters**:
  - `type`: binary_temporal
  - `n_frames`: 8
  - `fill_factor`: 0.5
  - `modulation_rate_hz`: 1000
- **Forward model**: $x_t^{\text{coded}} = m_t \odot x_t$ for $t=1,...,8$
- **Mismatch sources**:
  - `temporal_jitter` [medium]: Timing uncertainty in mask switching → correction: synchronization calibration
  - `mask_pattern_error` [low]: Actual pattern deviates from design → correction: measurement-based calibration

### Element: Phase Diffuser (`diffuser`)

- **Type**: optics (convolution)
- **Parameters**:
  - `type`: diffuse_caustic
  - `psf_support_px`: [256, 256]
  - `model`: convolution
- **Forward model**: $y_t = H \ast (m_t \odot x_t)$
- **Mismatch sources**:
  - `psf_calibration_error` [medium]: Measured PSF differs from true PSF → correction: in-situ calibration

### Element: Temporal Accumulation (`accumulator`)

- **Type**: processing
- **Parameters**:
  - `type`: sum
  - `n_frames`: 8
- **Forward model**: $y = \sum_{t=1}^{8} H \ast (m_t \odot x_t)$

### Element: Bare CMOS Sensor (`detector`)

- **Type**: detector
- **Parameters**:
  - `pixels`: [256, 256]
  - `pixel_pitch_um`: 5.5
  - `quantum_efficiency`: 0.6
- **Noise**:
  - poisson: mean_photons=3000
  - gaussian: sigma_electrons=3.0

## Composite Forward Model

```
y = sum_{t=1}^{T} H * (m_t ⊙ x_t) + n,  T = 8
```

where $H$ is the diffuser PSF, $m_t$ are binary temporal mask patterns, $x_t$ are video frames, $\odot$ is element-wise multiplication, $\ast$ is 2D convolution, and $n \sim \text{Poisson} + \mathcal{N}(0, \sigma^2)$.

**Canonical chain**: $M \to C \to \Sigma \to D$ (Modulate → Convolve → Accumulate → Detect)

**Measurement shape**: `(256, 256)` — single 2D measurement

**Object shape**: `(8, 256, 256)` — 8 video frames to recover

**Compression ratio**: 8:1 (8 frames compressed into 1 measurement)

## spec.md

```
modality: temporal_coded_lensless
carrier: photon
geometry: single_shot_temporal, T=8 frames
object: video sequence, 8x256x256
forward_model: M(temporal) -> C(PSF) -> Sigma(sum) -> D(intensity)
noise: Poisson(3000) + Gaussian(sigma=3e-)
target: PSNR >= 18 dB, SSIM >= 0.5
system_elements: source=broadband LED 100mW, optics=DMD temporal modulator + phase diffuser, detector=CMOS 256x256 5.5um
```

# Demands

- **feasibility**: yes
- **budget_feasible**: yes (< $2000 total system cost with DMD)
- **algorithm_convergence**: N/A (forward period)

**Comments**: This system combines two previously independent concepts: CACTI-style temporal coding (M→Σ→D) and lensless imaging (C→D), yielding a novel chain M→C→Σ→D. The temporal mask provides frame-to-frame diversity while the diffuser eliminates the need for an imaging lens. The 8:1 compression ratio makes recovery challenging but tractable with GAP-TV or PnP methods. Expected PSNR: 18-22 dB per frame (comparable to CACTI but degraded by diffuser ill-conditioning).
