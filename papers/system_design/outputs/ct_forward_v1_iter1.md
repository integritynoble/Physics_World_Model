---
modality: ct
period: forward
version: 1
iteration: 1
---

# Task

Design a complete forward model for sparse-view X-ray CT with only 60 projection angles and low photon flux (I0=1e4) for pediatric chest imaging. The system must model all physical elements from X-ray source through detector digitization, including realistic noise and calibration mismatch sources.

# Plan

1. Configure polychromatic X-ray tube source at 80 kVp with 1.5mm Al filtration
2. Model Beer-Lambert attenuation through soft tissue phantom
3. Define parallel-beam acquisition geometry with 60 projection angles over 180°
4. Simulate flat-panel CsI:Tl detector with Poisson noise (I0=1e4) and Gaussian readout noise
5. Apply 12-bit ADC digitization with dark current
6. Identify beam hardening, scatter, and center-of-rotation mismatch sources

# Action

## System Flowchart

```
[X-ray Tube 80kVp] → [Soft Tissue Phantom] → [Parallel-Beam 60 angles]
       ↓                      ↓                        ↓
  [Polychromatic         [Beer-Lambert           [CoR offset
   beam hardening]        attenuation]            mismatch]
                                                       ↓
                              → [CsI:Tl Flat Panel Detector] → [12-bit ADC] → y
                                        ↓
                                  [Poisson I0=1e4]
                                  [Gaussian σ=3 e⁻]
                                  [Dark current 0.05 e⁻/s]
```

### Element: X-ray Tube Source (80 kVp) (`xray_source`)

- **Type**: source
- **Parameters**:
  - `energy_kVp`: 80
  - `flux_photons_per_s`: 500000.0
  - `focal_spot_mm`: 0.4
  - `filtration`: 1.5mm Al
  - `spectrum`: polychromatic
- **Mismatch sources**:
  - `beam_hardening` [high]: Polychromatic spectrum causes cupping artifacts in soft tissue → correction: 2nd-order polynomial linearization from water phantom calibration
- **Connects to**: tissue_attenuation

### Element: Soft Tissue Attenuation (`tissue_attenuation`)

- **Type**: interaction
- **Parameters**:
  - `model`: beer_lambert
  - `mu_water_cm`: 0.184
  - `material`: pediatric_soft_tissue
- **Mismatch sources**:
  - `scatter` [medium]: Compton scatter adds low-frequency background (SPR ~0.3 for pediatric chest) → correction: Scatter kernel estimation with 1D convolution correction
- **Connects to**: geometry

### Element: Parallel-Beam Acquisition (60 angles) (`geometry`)

- **Type**: geometry
- **Parameters**:
  - `scan_type`: parallel_beam
  - `num_angles`: 60
  - `angular_range_deg`: 180
  - `detector_pixels`: 256
  - `pixel_pitch_mm`: 0.4
- **Mismatch sources**:
  - `center_of_rotation_offset` [medium]: Mechanical misalignment causes ring artifacts (estimated ±0.5 px) → correction: Cross-correlation of 0°/180° projection pair
- **Connects to**: detector

### Element: CsI:Tl Flat Panel Detector (`detector`)

- **Type**: detector
- **Parameters**:
  - `scintillator`: CsI:Tl
  - `pixels`: [256, 256]
  - `pixel_pitch_mm`: 0.4
  - `quantum_efficiency`: 0.75
- **Noise**:
  - poisson: I0=10000.0
  - gaussian: sigma_electrons=3.0
  - dark_current: electrons_per_s=0.05, exposure_s=0.02
- **Mismatch sources**:
  - `detector_response_nonuniformity` [low]: Per-pixel gain variations up to ±2% → correction: Flat-field correction with air scan
- **Connects to**: adc

### Element: 12-bit ADC (`adc`)

- **Type**: digitization
- **Parameters**:
  - `bit_depth`: 12
  - `dynamic_range_db`: 72

## Composite Noise Model

```
y ~ Poisson(I0 * exp(-H*x)) + N(0, σ_readout²) + Poisson(dark * t_exp)
```

**Measurement shape**: `(256, 60)`

# Demands

- **feasibility**: yes
- **budget_feasible**: yes
- **algorithm_convergence**: N/A

**Comments**: Low-dose (I0=1e4) will produce noisy sinograms (estimated SNR ~17 dB). Sparse view (60 angles) will cause streak artifacts in FBP but is recoverable with iterative reconstruction.
