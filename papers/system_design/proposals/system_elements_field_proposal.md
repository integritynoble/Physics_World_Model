# Proposal: Adding `system_elements` as the 8th Field to spec.md

## Status
- **Authors**: Chengshuai Yang
- **Date**: 2026-03-14
- **Related papers**:
  - *Designing Any Imaging System from Natural Language* (system design paper)
  - *Eleven Primitives and Three Gates* (flagship paper, Yang & Yuan)

---

## 1. Motivation

The current 7-field `spec.md` schema answers **"what is the forward model A?"** -- sufficient for reconstruction. But it does not answer three questions critical for real system design:

| Question | Current spec.md | Gap |
|----------|----------------|-----|
| Can this system actually be built? | No answer | No hardware feasibility check |
| What components are needed and what do they cost? | No answer | No cost estimation |
| What calibration tolerances are required for deployment? | No answer | No deployment guidance |

The flagship paper (*Eleven Primitives and Three Gates*) proved that every reconstruction failure decomposes into exactly three independent root causes -- **information deficiency (Gate 1)**, **carrier noise (Gate 2)**, and **operator mismatch (Gate 3)**. Each gate depends on specific physical parameters of the hardware elements. The current spec.md captures the mathematical operators but not the physical elements that instantiate them.

**The proposed 8th field bridges this gap**: it provides the hardware-level parameters that the Judge Agent needs to evaluate all three gates, turning the Judge from a math validator into a full system feasibility checker.

---

## 2. The Gap: From Operator Chain to Physical System

Consider the CASSI forward model chain: `M -> W -> Sigma -> D`

The current 7 fields tell us:
- M is a binary coded aperture modulation
- W is a spectral disperser with step = 2 px/band
- Sigma sums 28 bands
- D detects intensity with Gaussian noise sigma = 0.01

But they do NOT tell us:
- **Gate 1**: What is the effective compression ratio? Is rank(H) sufficient for the target PSNR? -- Needs mask fill factor, spatial/spectral dimensions
- **Gate 2**: How many photons reach each detector pixel? Is SNR above the noise floor? -- Needs source power, quantum efficiency, exposure time, read noise
- **Gate 3**: How precisely must the mask be aligned? What is the calibration tolerance? -- Needs mask fabrication tolerance, dispersion calibration precision, mechanical stability

These are properties of the **physical elements**, not the mathematical operators.

---

## 3. Proposed Schema Extension

### 3.1 The 8-Field spec.md

```yaml
# Fields 1-7: Physics level (what A does) -- UNCHANGED
modality:       # What system?
carrier:        # What physical carrier?
geometry:       # What measurement geometry?
object:         # What are we imaging?
forward_model:  # What is the operator chain?
noise:          # What corrupts measurements?
target:         # What quality do we need?

# Field 8: Engineering level (what builds A) -- NEW
system_elements:
  source:       # What generates the carrier
  optics:       # What shapes/modulates the carrier
  detector:     # What measures the signal
  calibration:  # What tolerances are required for deployment
```

### 3.2 Sub-field Definitions

#### `source` -- Carrier Generation Element

Provides parameters needed for **Gate 2 (Carrier Budget)** evaluation.

```yaml
source:
  type: <string>           # e.g., x_ray_tube, broadband_lamp, laser, rf_coil, transducer_array
  power: <float with unit>  # e.g., 50mW, 120kVp, 3T
  spectral_range: [min, max] <unit>  # e.g., [400, 700] nm
  coherence: <string>      # incoherent | partially_coherent | coherent
  pulse_mode: <string>     # continuous | pulsed
  repetition_rate: <float with unit>  # for pulsed sources
```

**Gate 2 link**: Source power and spectral range determine the photon/carrier flux incident on the object, which sets the fundamental SNR floor.

#### `optics` -- Carrier Modulation Elements

Provides parameters needed for **Gate 1 (Recoverability)** and **Gate 3 (Operator Mismatch)** evaluation.

Each primitive in the `forward_model` chain maps to a physical optical element:

```yaml
optics:
  - primitive: M            # Which primitive this element instantiates
    element: coded_aperture  # Physical component
    specs:
      type: binary_mask
      pixel_pitch: 10um
      fill_factor: 0.5
      size: [256, 256]
      fabrication_tolerance: 0.1um  # Gate 3 parameter
  - primitive: W
    element: prism_disperser
    specs:
      material: BK7
      dispersion: 2px/band
      spectral_resolution: 10nm
      alignment_tolerance: 0.05deg  # Gate 3 parameter
  - primitive: C             # If present in chain
    element: relay_lens
    specs:
      f_number: 2.8
      magnification: 1x
      aberration_budget: lambda/4  # Gate 3 parameter
```

**Gate 1 link**: Mask fill factor, spatial dimensions, and compression ratio determine information capacity.
**Gate 3 link**: Fabrication tolerances, alignment tolerances, and aberration budgets determine calibration sensitivity.

#### `detector` -- Signal Measurement Element

Provides parameters needed for **Gate 2 (Carrier Budget)** evaluation.

```yaml
detector:
  type: <string>           # CMOS, CCD, photon_counter, rf_receiver, piezo_array
  pixel_size: <float with unit>
  resolution: [nx, ny]
  quantum_efficiency: <float>   # 0 to 1
  read_noise: <float with unit> # e.g., 2e-
  dark_current: <float with unit>  # e.g., 0.1e-/s
  well_depth: <float with unit>    # e.g., 30000e-
  bit_depth: <int>                 # e.g., 16
  frame_rate: <float with unit>    # e.g., 30fps
```

**Gate 2 link**: Quantum efficiency, read noise, dark current, and well depth determine the measurement SNR via:

```
SNR = QE * N_photon / sqrt(QE * N_photon + sigma_read^2 + I_dark * t_exp)
```

#### `calibration` -- Deployment Tolerance Specification

Provides parameters needed for **Gate 3 (Operator Mismatch)** evaluation. This sub-field is unique to `system_elements` -- it has no counterpart in the physics-level fields.

```yaml
calibration:
  parameters:              # What can drift or be misaligned
    - name: mask_shift_xy
      tolerance: 0.25px
      sensitivity: 13.98dB/px  # From flagship Gate 3 analysis
      correction_method: cross_correlation
      correction_frequency: per_session
    - name: dispersion_step
      tolerance: 0.1px/band
      sensitivity: 3.2dB/(px/band)
      correction_method: spectral_lamp_calibration
      correction_frequency: daily
    - name: detector_gain
      tolerance: 2%
      sensitivity: 0.5dB/%
      correction_method: flat_field
      correction_frequency: daily
  autonomous_recovery_rate: 0.85  # Expected rho from flagship paper
```

**Gate 3 link**: Each calibration parameter maps to a term in the mismatch sensitivity:

```
Delta_PSNR ≈ -(10/ln10) * (delta_theta^T J^T J delta_theta) / MSE_0
```

where J is the parameter Jacobian from the flagship paper's Calibration Sensitivity Theorem.

---

## 4. Judge Agent: Three-Gate Feasibility Evaluation

With the 8th field, the Judge Agent performs a complete three-gate evaluation. The gate scoring functions are taken directly from the flagship paper.

### 4.1 Gate 1: Recoverability Check

**Input**: `geometry` + `optics` (compression ratio, mask properties, sampling pattern)

**Computation**:
```
gamma = m / n                                    # effective compression ratio
  where m = number of independent measurements
        n = signal dimension (from object field)

PSNR_max_G1 = lookup(modality, gamma)            # from compression_db.yaml
```

**Decision**:
```
IF PSNR_max_G1 < target.PSNR:
    REJECT "Information deficiency: compression ratio gamma={gamma}
            insufficient for target {target.PSNR} dB.
            Minimum gamma = {gamma_min} (from compression table).
            Action: increase n_angles / n_coils / reduce compression."
ELSE:
    PASS with margin = PSNR_max_G1 - target.PSNR
```

**Example (CT)**:
- geometry: 60-view fan-beam, 736 detectors
- object: 362x362 image (n = 131,044)
- measurements: 60 x 736 = 44,160 (m)
- gamma = 44,160 / 131,044 = 0.337
- compression_db lookup: PSNR_max_G1 ~ 22 dB at gamma = 0.34 for CT
- target: PSNR >= 20 dB -> PASS (margin = 2 dB)

### 4.2 Gate 2: Carrier Budget Check

**Input**: `source` + `detector` + `noise` (photon flux, QE, read noise, exposure)

**Computation**:
```
N_photon = source.power * detector.QE * t_exposure / E_photon  # per pixel

SNR_measurement = N_photon / sqrt(N_photon + sigma_read^2 + I_dark * t_exp)

PSNR_max_G2 = 10 * log10(SNR) + C_M
  where C_M = 10 * log10( (n * ||x||_inf^2 / ||x||^2) * kappa(H)^-2 )
```

**Decision**:
```
IF PSNR_max_G2 < target.PSNR:
    REJECT "Carrier budget insufficient: SNR = {SNR_measurement:.1f} dB,
            noise floor exceeds target by {deficit:.1f} dB.
            Action: increase source power / exposure time / detector QE."
ELSE:
    PASS with noise_regime = {shot_limited | read_limited | dark_limited}
```

**Example (CASSI)**:
- source: 50mW broadband, 400-700nm
- detector: CMOS, QE=0.7, read_noise=2e-, well_depth=30ke-
- Per-band photon count: ~5000 photons/pixel
- SNR = 5000 / sqrt(5000 + 4) ~ 70.7 -> 37 dB
- target: PSNR >= 28 dB -> PASS (margin = 9 dB)

### 4.3 Gate 3: Operator Mismatch Check

**Input**: `calibration` (tolerances, sensitivities, correction methods)

**Computation**:
```
For each calibration parameter theta_k:
    Delta_PSNR_k = sensitivity_k * tolerance_k    # linear approximation

Delta_PSNR_total = sqrt( sum_k Delta_PSNR_k^2 )  # RSS for independent errors

PSNR_deployment = PSNR_max_G2 - Delta_PSNR_total
```

**Decision**:
```
IF PSNR_deployment < target.PSNR:
    WARNING "Gate 3 risk: expected deployment degradation = {Delta_PSNR_total:.1f} dB.
             Dominant parameter: {dominant_param} (sensitivity = {s:.1f} dB/unit).
             Autonomous recovery rate: {rho:.0%}.
             Action: tighten {dominant_param} tolerance or increase calibration frequency."
ELSE:
    PASS with deployment_margin = PSNR_deployment - target.PSNR
```

**Example (CASSI)**:
- mask_shift: tolerance 0.25px, sensitivity 13.98 dB/px -> Delta = 3.5 dB
- dispersion: tolerance 0.1px/band, sensitivity 3.2 dB/(px/band) -> Delta = 0.32 dB
- gain: tolerance 2%, sensitivity 0.5 dB/% -> Delta = 1.0 dB
- Total: sqrt(3.5^2 + 0.32^2 + 1.0^2) = 3.6 dB
- PSNR_deployment = 37 - 3.6 = 33.4 dB > 28 dB target -> PASS
- With autonomous recovery (rho = 0.85): recovers 0.85 * 3.6 = 3.1 dB -> effective loss = 0.5 dB

### 4.4 Cost Estimation

**Input**: All `system_elements` sub-fields

**Computation**: Look up component costs from a hardware database:

```yaml
cost_estimate:
  source: {item: broadband_LED, cost: $300}
  optics:
    - {item: coded_aperture_mask, cost: $500, note: "lithographic binary mask"}
    - {item: prism_disperser, cost: $800, note: "BK7 equilateral prism"}
    - {item: relay_lens, cost: $500, note: "f/2.8 telecentric"}
  detector: {item: CMOS_camera, cost: $2500, note: "Hamamatsu ORCA-Flash"}
  calibration: {item: spectral_lamp, cost: $200, note: "HgAr wavelength reference"}
  total: $4800
  confidence: estimated  # or quoted | measured
```

### 4.5 Complete Judge Report

```
========================================
JUDGE REPORT: CASSI Spectral Imager
========================================

Gate 1 (Recoverability):          PASS  [margin: +6 dB]
  Compression ratio: 28:1
  Information capacity: sufficient (rank(H) > n_bands * n_spatial)

Gate 2 (Carrier Budget):          PASS  [margin: +9 dB]
  Per-band photons: ~5000/pixel
  SNR floor: 37 dB > 28 dB target
  Noise regime: shot-limited

Gate 3 (Operator Mismatch):       PASS  [margin: +5.4 dB, with recovery]
  Expected degradation: 3.6 dB (mask alignment dominant)
  Autonomous recovery: 85% (rho = 0.85)
  Residual loss: 0.5 dB after calibration
  Dominant parameter: mask_shift_xy (sensitivity 14 dB/px)

Cost Estimate:                    $4,800
  Source: $300 | Optics: $1,800 | Detector: $2,500 | Calibration: $200

VERDICT: FEASIBLE
  Expected deployment PSNR: 33.4 dB (target: 28 dB)
  Critical path: mask alignment (tightest tolerance)
========================================
```

---

## 5. Worked Examples: Three Modalities

### 5.1 Clinical CT

```yaml
modality: computed_tomography
carrier: xray
geometry: fan_beam, n_angles=60, n_det=736
object: 362x362 image, non-negative
forward_model: Pi -> D
noise: Poisson, I_0=1e4
target: PSNR >= 20dB

system_elements:
  source:
    type: x_ray_tube
    voltage: 120kVp
    current: 200mA
    focal_spot: 0.6mm
    filtration: 2.5mm_Al
  optics:
    - primitive: Pi
      element: fan_beam_geometry
      specs:
        source_to_isocenter: 800mm
        isocenter_to_detector: 568mm
        detector_pitch: 1.496mm
        n_detector_channels: 736
        rotation_precision: 0.01deg   # Gate 3
  detector:
    type: scintillator_photodiode
    pixel_size: 1.0mm
    resolution: [1, 736]
    quantum_efficiency: 0.85
    read_noise: 50e-
    dynamic_range: 20bit
  calibration:
    parameters:
      - name: center_of_rotation
        tolerance: 2px
        sensitivity: 1.0dB/px         # From flagship: CoR ~2px for <1dB
        correction_method: sinogram_symmetry
        correction_frequency: per_scan
      - name: detector_gain_nonuniformity
        tolerance: 1%
        sensitivity: 0.3dB/%
        correction_method: air_scan_normalization
        correction_frequency: daily
    autonomous_recovery_rate: 0.97     # CT: 95-100% in flagship paper
```

**Judge Gate Evaluation**:
- Gate 1: gamma = 44,160/131,044 = 0.34; PSNR_max ~ 22 dB > 20 dB target -> PASS
- Gate 2: I_0 = 1e4 photons; SNR ~ 40 dB >> 20 dB target -> PASS
- Gate 3: CoR tolerance 2px at 1.0 dB/px -> 2.0 dB degradation; recovery 97% -> residual 0.06 dB -> PASS
- Cost: ~$150k (clinical) or ~$5k (benchtop micro-CT)

### 5.2 Accelerated MRI

```yaml
modality: mri
carrier: spin
geometry: cartesian_kspace, acceleration=4x, n_coils=4
object: 256x256 complex image
forward_model: M(coil) -> F(kspace) -> S -> D
noise: Gaussian, SNR=30dB
target: SSIM >= 0.9

system_elements:
  source:
    type: superconducting_magnet
    field_strength: 3T
    homogeneity: 1ppm_over_40cm
    gradient_strength: 40mT/m
    gradient_slew_rate: 200T/m/s
  optics:
    - primitive: M
      element: receive_coil_array
      specs:
        n_coils: 4
        coil_type: surface_loop
        coil_diameter: 15cm
        sensitivity_uniformity: 95%    # Gate 3
        coupling_isolation: -15dB
    - primitive: F
      element: gradient_system
      specs:
        encoding: cartesian
        matrix: [256, 256]
        fov: 220mm
        eddy_current_compensation: true  # Gate 3
    - primitive: S
      element: undersampling_pattern
      specs:
        type: uniform_cartesian
        acceleration: 4x
        acs_lines: 24              # auto-calibration signal lines
  detector:
    type: quadrature_receiver
    bandwidth: 125kHz
    noise_figure: 1.0dB
    adc_bits: 16
    n_channels: 4
  calibration:
    parameters:
      - name: coil_sensitivity_map
        tolerance: 5%
        sensitivity: 3.5dB/5%        # From flagship: 1.75-7.14dB at 5%
        correction_method: ESPIRiT
        correction_frequency: per_scan
      - name: b0_field_inhomogeneity
        tolerance: 50Hz
        sensitivity: 0.8dB/50Hz
        correction_method: field_map
        correction_frequency: per_scan
      - name: gradient_delay
        tolerance: 1us
        sensitivity: 0.2dB/us
        correction_method: navigator_echo
        correction_frequency: per_sequence
    autonomous_recovery_rate: 0.95
```

### 5.3 Snapshot CASSI

```yaml
modality: cassi
carrier: photon
geometry: coded_aperture + disperser, lambda=[400,700]nm
object: 256x256x28 spectral cube
forward_model: M -> W -> Sigma -> D
noise: Gaussian, sigma=0.01
target: PSNR >= 28dB

system_elements:
  source:
    type: broadband_illumination
    power: 50mW
    spectral_range: [400, 700]nm
    uniformity: 90%_over_fov
  optics:
    - primitive: M
      element: coded_aperture_mask
      specs:
        type: binary_random
        pixel_pitch: 10um
        fill_factor: 0.5
        size: [256, 256]
        fabrication_method: photolithography
        fabrication_tolerance: 0.1um       # Gate 3
        minimum_feature: 10um
    - primitive: W
      element: amici_prism
      specs:
        material: BK7/SF2
        dispersion: 2px/band
        spectral_range: [400, 700]nm
        alignment_tolerance: 0.05deg       # Gate 3
    - primitive: Sigma
      element: single_shot_integration
      specs:
        note: "implicit -- detector integrates all bands in one exposure"
  detector:
    type: CMOS
    model: Hamamatsu_ORCA_Flash4
    pixel_size: 6.5um
    resolution: [512, 512]
    quantum_efficiency: 0.7
    read_noise: 1.6e-
    dark_current: 0.06e-/s
    well_depth: 30000e-
    bit_depth: 16
    frame_rate: 100fps
  calibration:
    parameters:
      - name: mask_shift_xy
        tolerance: 0.25px
        sensitivity: 13.98dB/px           # Flagship: 13.98 +/- 0.62
        correction_method: cross_correlation
        correction_frequency: per_session
      - name: mask_rotation
        tolerance: 0.05deg
        sensitivity: 2.1dB/deg
        correction_method: registration
        correction_frequency: per_session
      - name: dispersion_step
        tolerance: 0.1px/band
        sensitivity: 3.2dB/(px/band)
        correction_method: spectral_lamp
        correction_frequency: weekly
      - name: flat_field
        tolerance: 2%
        sensitivity: 0.5dB/%
        correction_method: uniform_target
        correction_frequency: daily
    autonomous_recovery_rate: 0.85         # Flagship: CASSI 85% recovery
```

---

## 6. Connection to the Two Papers

### 6.1 Architecture: How the Three Papers Fit Together

```
Flagship Paper (Paper I)                System Design Paper (Paper II)
"11 Primitives and 3 Gates"             "Designing Any Imaging System
                                         from Natural Language"

PROVES:                                  PROVIDES:
- 11 primitives are sufficient           - spec.md 7-field schema
  and minimal                            - Plan/Judge/Performance agents
- 3 gates are complete and               - Natural language -> spec.md
  independent failure modes              - Automated reconstruction
- Gate scoring functions
- Recovery protocol

         ╲                              ╱
          ╲                            ╱
           ╲                          ╱
            ▼                        ▼
      8th field: system_elements
      BRIDGES:
      - Maps each primitive to its physical element
      - Provides gate-specific parameters
      - Enables Judge Agent to perform 3-gate evaluation
      - Adds cost estimation and feasibility checking
```

### 6.2 Registry Connection

The flagship paper already maintains YAML registries that directly feed the `system_elements` field:

| Flagship Registry | Lines | Feeds | system_elements Sub-field |
|---|---|---|---|
| `compression_db.yaml` | 1,186 | Gate 1 thresholds | `optics` (compression ratio from geometry) |
| `photon_db.yaml` | -- | Gate 2 thresholds | `source` + `detector` (photon budget) |
| `mismatch_db.yaml` | -- | Gate 3 sensitivities | `calibration` (per-parameter sensitivity) |
| `graph_templates.yaml` | 170 entries | DAG templates | `optics` (primitive-to-element mapping) |
| `modalities.yaml` | 170 entries | Default specs | All sub-fields (default values) |

The `system_elements` field provides the **instance-specific values** that the registries provide **default ranges** for. A user can accept defaults (auto-populated from registry) or override with actual hardware specifications.

### 6.3 Gate Scoring: Flagship Equations in the Judge Agent

The Judge Agent implements the flagship paper's gate scoring directly:

**Gate 1** (Compression Bound Theorem, Supplementary Note 1):
```
MSE_min >= (1/n) * sum_{i=1}^{n-m} sigma_i^2(x)
PSNR_max_G1 = 10 * log10(||x||_inf^2 / MSE_min)
```

**Gate 2** (Noise Bound Theorem, Supplementary Note 1):
```
PSNR_max_G2 = 10 * log10(SNR) + C_M
where C_M = 10 * log10( (n * ||x||_inf^2 / ||x||^2) * kappa(H)^-2 )
```

**Gate 3** (Calibration Sensitivity Theorem, Supplementary Note 1):
```
Delta_PSNR ≈ -(10/ln10) * (delta_theta^T J^T J delta_theta) / MSE_0
```

**Gate dominance** (methods.tex):
```
C_mismatch = PSNR_I - PSNR_II
C_noise    = PSNR_ideal - PSNR_noisy
C_recover  = PSNR_limit - PSNR_I
dominant_gate = argmax(C_recover, C_noise, C_mismatch)
```

**Recovery ratio** (methods.tex):
```
rho = (PSNR_IV - PSNR_II) / (PSNR_I - PSNR_II)
```

---

## 7. Design Principles

### 7.1 Backward Compatibility

The 8th field is **optional**. Existing 7-field spec.md files remain valid. When `system_elements` is absent:
- The Judge Agent performs math-only validation (current behavior)
- No gate evaluation, cost estimation, or feasibility check
- Reconstruction proceeds as before

When `system_elements` is present:
- The Judge Agent additionally performs 3-gate evaluation
- Cost estimation is generated
- Deployment guidance (Gate 3 tolerances) is provided

### 7.2 Auto-Population from Registries

For common modalities, `system_elements` can be auto-populated from the flagship paper's YAML registries:

```python
def auto_populate_elements(spec: dict) -> dict:
    """Fill system_elements defaults from flagship registries."""
    modality = spec["modality"]

    # Gate 1: compression defaults
    spec["system_elements"]["compression"] = compression_db[modality]

    # Gate 2: photon budget defaults
    spec["system_elements"]["source"] = photon_db[modality]["source"]
    spec["system_elements"]["detector"] = photon_db[modality]["detector"]

    # Gate 3: mismatch sensitivities
    spec["system_elements"]["calibration"] = mismatch_db[modality]

    return spec
```

Users override defaults with actual hardware specs for their specific system.

### 7.3 Primitive-to-Element Mapping

Each primitive in the `forward_model` chain maps to one or more physical elements in `optics`. This mapping is explicit:

```
forward_model: M -> W -> Sigma -> D
                |    |     |      |
                v    v     v      v
optics:     mask  prism  (implicit)  detector
```

The mapping enables the Judge to:
1. Verify every primitive has a physical instantiation (completeness)
2. Extract Gate 3 parameters from the element that implements each primitive
3. Trace reconstruction failures back to specific hardware components

### 7.4 Scope Boundary

`system_elements` captures **gate-relevant engineering parameters**, not exhaustive hardware specifications. The principle is:

- **Include**: Any parameter that affects Gate 1, 2, or 3 scoring
- **Include**: Any parameter needed for cost estimation
- **Exclude**: Mechanical housing, power supply, software interface, cable routing
- **Exclude**: Parameters that don't affect reconstruction quality

---

## 8. Implementation Plan

### Phase 1: Schema Definition
- Define `system_elements` YAML schema with JSON Schema validation
- Add to `spec.md` parser with optional field handling
- Write schema documentation with all allowed values

### Phase 2: Registry Integration
- Connect to flagship paper's `compression_db.yaml`, `photon_db.yaml`, `mismatch_db.yaml`
- Implement auto-population for 170 registered modalities
- Add hardware cost database (initially for 12 validated modalities)

### Phase 3: Judge Agent Extension
- Implement Gate 1 scoring (compression bound check)
- Implement Gate 2 scoring (carrier budget check)
- Implement Gate 3 scoring (mismatch sensitivity analysis)
- Implement cost estimation
- Generate structured `JudgeReport` with gate verdicts

### Phase 4: Validation
- Validate on 12 flagship modalities (CT, MRI, CASSI, CACTI, SPC, lensless, holography, fluorescence, CBCT, cryo-EM, electron ptychography, ultrasound)
- Verify gate scores match flagship paper's published thresholds
- Verify cost estimates against known system prices
- Test auto-population accuracy

---

## 9. Summary

| Aspect | Without system_elements | With system_elements |
|---|---|---|
| **Scope** | Forward model only | Forward model + physical system |
| **Judge checks** | Math consistency (6 gates) | Math + 3 physical gates + cost |
| **Feasibility** | Unknown | Pass/Fail per gate with margin |
| **Cost** | Unknown | Itemized estimate |
| **Deployment guidance** | None | Gate 3 tolerances + calibration schedule |
| **Connection to flagship** | Implicit (shared primitives) | Explicit (gate scores from flagship equations) |
| **Backward compatible** | N/A | Yes (8th field is optional) |

The 8th field transforms spec.md from a **reconstruction specification** into a **complete system design specification** -- answering not just "what is A?" but "can A be built, at what cost, and what calibration does it need?"
