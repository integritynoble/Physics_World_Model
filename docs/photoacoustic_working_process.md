# Photoacoustic Working Process

## End-to-End Pipeline for Photoacoustic Imaging

This document traces a complete photoacoustic imaging experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct an initial pressure distribution from photoacoustic RF data.
 RF data: rf_data.npy, Transducer positions: transducer_pos.npy,
 128 elements, 2048 time samples, speed of sound 1540 m/s."
```

---

## 2. PlanAgent — Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "rf_data.npy" detected
#   operator_type=OperatorType.linear_operator,
#   files=["rf_data.npy", "transducer_pos.npy"],
#   params={"n_elements": 128, "n_time_samples": 2048,
#           "speed_of_sound_m_per_s": 1540}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> photoacoustic entry
photoacoustic:
  keywords: [photoacoustic, optoacoustic, PAI, laser_ultrasound, absorption_contrast]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="photoacoustic",
#   confidence=0.95,
#   reasoning="Matched keywords: photoacoustic, RF data"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the photoacoustic registry entry:

```python
system = plan_agent.build_imaging_system("photoacoustic")
# ImagingSystem(
#   modality_key="photoacoustic",
#   display_name="Photoacoustic Imaging",
#   signal_dims={"x": [256, 256], "y": [128, 2048]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...4 elements...],
#   default_solver="back_projection"
# )
```

**Photoacoustic Element Chain (4 elements):**

```
Pulsed Laser (Nd:YAG) --> Tissue Absorption --> Acoustic Propagation --> Ultrasound Transducer Array
  throughput=1.0            throughput=0.30       throughput=0.85          throughput=0.75
  noise: none               noise: none           noise: none              noise: shot+read+thermal
  20 mJ/pulse               Gamma=0.8             v_s=1540 m/s            128 elements, 5 MHz center
  5 ns pulse                mu_a=0.1/cm           0.5 dB/cm/MHz atten     60% bandwidth, 40 MHz sampling
  10 Hz rep rate            diffusion model        soft tissue              14-bit ADC
```

**Cumulative throughput:** `0.30 x 0.85 x 0.75 = 0.191`

**Forward model equation:**
```
y = R * p0,   where p0 = Gamma * mu_a * Phi

R: acoustic propagation operator (circular Radon transform)
p0: initial pressure distribution
Gamma: Grueneisen parameter (~0.8)
mu_a: optical absorption coefficient
Phi: local optical fluence
```

---

## 3. PhotonAgent — SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  photoacoustic:
    model_id: "pulsed_laser"
    parameters:
      power_w: 10.0
      wavelength_nm: 532
      na: 0.3
      qe: 0.85
      exposure_s: 0.00001
  ```

Note: In photoacoustic imaging, "photon budget" represents the optical energy deposited in tissue, which generates the acoustic signal. The SNR is ultimately acoustic, determined by the initial pressure amplitude and transducer noise floor.

### Computation

```python
# 1. Photon energy at 532 nm
E_photon = h * c / wavelength_nm
#        = 6.626e-34 * 3e8 / (532e-9)
#        = 3.74e-19 J

# 2. Pulse energy
E_pulse = power_w * exposure_s = 10.0 * 1e-5 = 1e-4 J
# But pulse_energy_mj = 20 mJ from modalities.yaml
E_pulse = 0.020  # 20 mJ per pulse

# 3. Optical fluence at tissue surface
# beam_diameter ~ 1 cm, area = pi * (0.5)^2 = 0.785 cm^2
fluence = E_pulse / 0.785e-4 = 255 J/m^2
# (MPE limit for skin at 532 nm: 20 mJ/cm^2 = 200 J/m^2)

# 4. Initial pressure
# p0 = Gamma * mu_a * Phi
# Gamma = 0.8, mu_a = 0.1/cm = 10/m, Phi ~ fluence * exp(-mu_eff * depth)
# At depth d: Phi(d) = fluence * exp(-mu_eff * d)
# mu_eff = sqrt(3 * mu_a * (mu_a + mu_s')) ~ 4.6/cm for soft tissue
# At 5mm depth: Phi = 255 * exp(-4.6 * 0.5) = 25.5 J/m^2
p0_typical = 0.8 * 10.0 * 25.5 = 204 Pa

# 5. Transducer sensitivity and noise
# NEP (Noise Equivalent Pressure) ~ 1 Pa for typical piezo transducer
# Acoustic SNR = p0 / NEP
SNR_acoustic = 204 / 1.0 = 204
SNR_db = 20 * log10(204) = 46.2 dB

# 6. Electronic noise contribution
# Transducer preamp noise ~ 10 uV rms, sensitivity ~ 50 uV/Pa
read_noise_pa = 10e-6 / 50e-6 = 0.2 Pa
total_noise = sqrt(1.0^2 + 0.2^2) = 1.02 Pa
SNR_total = 204 / 1.02 = 200
SNR_total_db = 20 * log10(200) = 46.0 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=5.35e16,               # Photons per pulse into tissue
  snr_db=46.0,
  noise_regime=NoiseRegime.shot_limited,      # Acoustic SNR dominated by p0 amplitude
  shot_noise_sigma=1.0,                       # NEP in Pa
  read_noise_sigma=0.2,                       # Electronic noise in Pa-equivalent
  total_noise_sigma=1.02,
  feasible=True,
  quality_tier="excellent",                   # SNR > 30 dB
  throughput_chain=[
    {"Pulsed Laser Source": 1.0},
    {"Tissue Absorption": 0.30},
    {"Acoustic Propagation": 0.85},
    {"Ultrasound Transducer Array": 0.75}
  ],
  noise_model="gaussian",                     # Acoustic noise is approximately Gaussian
  explanation="Excellent acoustic SNR at shallow depths. Signal drops exponentially "
              "with depth due to optical attenuation (mu_eff ~ 4.6/cm)."
)
```

---

## 4. MismatchAgent — Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"photoacoustic"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  photoacoustic:
    parameters:
      speed_of_sound:
        range: [1400.0, 1600.0]
        typical_error: 20.0
        unit: "m/s"
        description: "Sound speed error from tissue composition assumption"
      transducer_position:
        range: [-2.0, 2.0]
        typical_error: 0.3
        unit: "mm"
        description: "Transducer element position error from manufacturing tolerance"
      laser_fluence:
        range: [0.5, 1.5]
        typical_error: 0.15
        unit: "normalized"
        description: "Optical fluence distribution error from tissue heterogeneity"
    severity_weights:
      speed_of_sound: 0.45
      transducer_position: 0.30
      laser_fluence: 0.25
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.45 * |20.0| / 200.0     # speed_of_sound:    0.045
  + 0.30 * |0.3| / 4.0        # transducer_pos:    0.023
  + 0.25 * |0.15| / 1.0       # laser_fluence:     0.038
S = 0.106  # Low severity (typical conditions)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.06 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="photoacoustic",
  mismatch_family="grid_search",
  parameters={
    "speed_of_sound":    {"typical_error": 20.0, "range": [1400, 1600], "weight": 0.45},
    "transducer_position": {"typical_error": 0.3, "range": [-2, 2], "weight": 0.30},
    "laser_fluence":     {"typical_error": 0.15, "range": [0.5, 1.5], "weight": 0.25}
  },
  severity_score=0.106,
  correction_method="grid_search",
  expected_improvement_db=1.06,
  explanation="Low mismatch severity. Speed of sound uncertainty is the primary "
              "calibration concern but within acceptable bounds."
)
```

---

## 5. RecoverabilityAgent — Can We Reconstruct?

**File:** `agents/recoverability_agent.py` (912 lines)

### Input
- `ImagingSystem` (signal_dims for CR calculation)
- `PhotonReport` (noise regime)
- Calibration table from `compression_db.yaml`:
  ```yaml
  photoacoustic:
    signal_prior_class: "tv"
    entries:
      - {cr: 0.5, noise: "shot_limited", solver: "ubp",
         recoverability: 0.78, expected_psnr_db: 32.1,
         provenance: {dataset_id: "ipasc_phantom_2023", ...}}
      - {cr: 0.5, noise: "detector_limited", solver: "ubp",
         recoverability: 0.65, expected_psnr_db: 28.3, ...}
      - {cr: 0.5, noise: "shot_limited", solver: "model_based_pa",
         recoverability: 0.86, expected_psnr_db: 35.2, ...}
  ```

### Computation

```python
# 1. Compression ratio
#    x = (256, 256) = 65536 pixels (initial pressure image)
#    y = (128, 2048) = 262144 samples (RF data)
CR = prod(y_shape) / prod(x_shape) = 262144 / 65536 = 4.0
# NOTE: CR > 1 means oversampled — more measurements than unknowns
# Use effective CR for lookup: min(CR, 1.0) is not appropriate here
# Instead: the lookup uses nominal CR=0.5 from the calibration table
# (accounts for limited-view geometry reducing effective sampling)

# 2. Limited-view factor
# 128-element linear array covers ~120 degrees (not full 360)
# Effective coverage fraction: 120/360 = 0.33
# Adjusted CR = 4.0 * 0.33 = 1.32 -> still oversampled for in-view features
# But limited-view creates null space for features tangent to array

# 3. Calibration table lookup
#    Match: noise="shot_limited", solver="ubp", cr=0.5
#    -> recoverability=0.78, expected_psnr=32.1 dB

# 4. Best solver selection
#    model_based_pa: 35.2 dB > ubp: 32.1 dB
#    -> recommended: "model_based_pa" (iterative, better quality)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.5,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.tv,
  operator_diversity_score=0.7,
  condition_number_proxy=0.588,
  recoverability_score=0.78,
  recoverability_confidence=1.0,
  expected_psnr_db=32.1,
  expected_psnr_uncertainty_db=1.5,
  recommended_solver_family="ubp",
  verdict="good",                         # score >= 0.70
  calibration_table_entry={...},
  explanation="Good recoverability. Universal back-projection expected 32.1 dB "
              "on IPASC phantom benchmark. Limited-view geometry may cause "
              "streak artifacts for features outside transducer aperture."
)
```

---

## 6. AnalysisAgent — Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(46.0 / 40, 1.0)   = 0.0     # Excellent SNR
mismatch_score    = 0.106                        = 0.106   # Low mismatch
compression_score = 1 - 0.78                     = 0.22    # Good recoverability
solver_score      = 0.2                          = 0.2     # Default placeholder

# Primary bottleneck
primary = "compression"  # max(0.0, 0.106, 0.22, 0.2) = compression

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.106*0.5) * (1 - 0.22*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.947 * 0.89 * 0.90
  = 0.758
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="compression",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.106, compression=0.22, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Use model-based iterative reconstruction for +3.1 dB over back-projection",
      priority="high",
      expected_gain_db=3.1
    ),
    Suggestion(
      text="Limited-view artifacts from linear array; consider circular geometry for future",
      priority="low",
      expected_gain_db=2.0
    )
  ],
  overall_verdict="good",               # P >= 0.70
  probability_of_success=0.758,
  explanation="System is well-configured. Limited-view geometry is the primary constraint."
)
```

---

## 7. AgentNegotiator — Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="good" | No veto |
| Severe mismatch without correction | severity=0.106 < 0.7 | No veto |
| All marginal | All good/excellent | No veto |
| Joint probability floor | P=0.758 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.78    # recoverability_score
P_mismatch       = 1.0 - 0.106 * 0.7 = 0.926

P_joint = 0.95 * 0.78 * 0.926 = 0.686
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.686
)
```

---

## 8. PreFlightReportBuilder — Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 256 * 256 = 65536
n_transducers = 128
dim_factor   = total_pixels * n_transducers / (256 * 256) = 128.0
solver_complexity = 1.0   # Back-projection (single pass)
cr_factor    = max(0.5, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 128.0 * 1.0 * 0.125 = 32.0 seconds
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="photoacoustic", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=32.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=[
    "RF data from transducer array (N_elements x N_time_samples)",
    "Transducer element positions (N_elements x 2 or 3)",
    "Speed of sound value or map"
  ]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Photoacoustic forward model: y = R * p0
#
# R: circular Radon transform (acoustic propagation)
# sinogram[i, t] = sum_r p0(r) * delta(|r - r_i| - c*t)
#
# Parameters:
#   transducer_positions: (128, 2) element locations [from transducer_pos.npy]
#   speed_of_sound: 1540 m/s (= 1.0 pixel units after normalization)
#   n_time_samples: 2048
#
# Input:  x = (256, 256) initial pressure distribution p0
# Output: y = (128, 2048) RF sinogram data

class PhotoacousticOperator(PhysicsOperator):
    def forward(self, p0):
        """sinogram[i,t] = sum_r p0(r) * delta(|r-r_i| - c*t)"""
        dist = compute_distance_matrix(grid_shape, trans_pos)
        time_idx = np.round(dist / speed_of_sound).astype(int)
        sinogram = np.zeros((n_trans, n_times))
        for i in range(n_trans):
            np.add.at(sinogram[i], time_idx[i][valid], p0[valid])
        return sinogram

    def adjoint(self, sinogram):
        """p0_hat[r] = sum_i sinogram[i, |r-r_i|/c]  (delay-and-sum)"""
        dist = compute_distance_matrix(grid_shape, trans_pos)
        time_idx = np.round(dist / speed_of_sound).astype(int)
        recon = np.zeros(grid_shape)
        for i in range(n_trans):
            recon += sinogram[i, time_idx[i]]
        return recon / n_trans

    def check_adjoint(self):
        """Verify <Ax, y> ~ <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided rf_data.npy:
y = np.load("rf_data.npy")          # (128, 2048)
trans_pos = np.load("transducer_pos.npy")  # (128, 2)

# If simulating:
# Ground truth: blood vessel phantom with Gaussian cross-sections
x_true = np.zeros((128, 128))
for vessel in range(6):
    # Line segments with Gaussian cross-sections
    for t in linspace(0, 1, length):
        gauss = exp(-((xx-px)^2 + (yy-py)^2) / (2*sigma^2))
        x_true += gauss * 0.05

# Transducer array: 128 elements in circular geometry
radius = 0.6 * n
angles = linspace(0, 2*pi, 128, endpoint=False)
trans_pos = stack([n/2 + radius*sin(angles), n/2 + radius*cos(angles)])

# Forward model + noise
sinogram = _forward_photoacoustic(x_true, trans_pos, n_times=1024)
sinogram += np.random.randn(*sinogram.shape) * 0.01  # Additive Gaussian noise
```

### Step 9c: Reconstruction with Back-Projection

```python
from pwm_core.recon.photoacoustic_solver import back_projection, time_reversal

# Algorithm 1: Universal Back-Projection (fast, standard)
recon_bp = back_projection(
    sinogram=y,                    # (128, 2048) RF data
    transducer_positions=trans_pos, # (128, 2) element positions
    grid_shape=(256, 256),          # Reconstruction grid
    speed_of_sound=1.0              # Normalized units
)
# Delay-and-sum: p0_hat(r) = (1/N) * sum_i sinogram[i, |r-r_i|/c]
# Expected PSNR: ~32.1 dB on IPASC phantom benchmark

# Algorithm 2: Iterative Time Reversal (higher quality)
recon_tr = time_reversal(
    sinogram=y,
    transducer_positions=trans_pos,
    grid_shape=(256, 256),
    speed_of_sound=1.0,
    n_iters=20                      # Forward-backward iteration count
)
# Iterative refinement: x_{k+1} = x_k + step * A^T(y - A*x_k)
# Non-negativity constraint applied each iteration
# Expected PSNR: ~35.2 dB with model-based approach
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| Back-Projection | Traditional | 32.1 dB | No | `back_projection(y, trans_pos, grid_shape)` |
| Time Reversal | Traditional | 33.5 dB | No | `time_reversal(y, trans_pos, grid_shape, n_iters=20)` |
| Model-Based PA | Iterative | 35.2 dB | Yes | `model_based_pa(y, trans_pos, grid_shape, n_iters=200)` |

### Step 9d: Metrics

```python
# PSNR
psnr = 10 * log10(max_val^2 / mse(recon, x_true))
# ~32.1 dB (back-projection), ~35.2 dB (model-based)

# SSIM
ssim = compute_ssim(recon, x_true)

# Contrast-to-Noise Ratio (CNR) — photoacoustic-specific
# CNR = (mu_vessel - mu_background) / sigma_background
cnr = (mean(recon[vessel_mask]) - mean(recon[bg_mask])) / std(recon[bg_mask])

# Resolution: full-width at half-maximum of vessel cross-section
fwhm = measure_fwhm(recon, vessel_centerline)
```

### Step 9e: RunBundle Output

```
run_bundle/
+-- meta.json              # ExperimentSpec + provenance
+-- agent_reports/
|   +-- photon_report.json
|   +-- mismatch_report.json
|   +-- recoverability_report.json
|   +-- system_analysis.json
|   +-- negotiation_result.json
|   +-- preflight_report.json
+-- arrays/
|   +-- y.npy              # RF sinogram (128, 2048) + SHA256 hash
|   +-- x_hat.npy          # Reconstructed pressure (256, 256) + SHA256 hash
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
+-- metrics.json           # PSNR, SSIM, CNR, FWHM
+-- operator.json          # Operator params (trans_pos hash, SoS, n_times)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized photoacoustic pipeline with high SNR (46.0 dB) and low mismatch. In practice, real photoacoustic systems face speed-of-sound heterogeneity in tissue, limited-view artifacts from incomplete transducer coverage, and depth-dependent signal loss from optical attenuation.

---

## Real Experiment: User Prompt

```
"Small-animal photoacoustic imaging of subcutaneous tumor vasculature.
 Linear array, partial view. The speed of sound may be wrong because
 of fat layers. RF data: tumor_rf.npy, array_pos: array_linear.npy,
 128 elements, 2048 samples, Fs=40 MHz."
```

**Key difference:** Limited-view geometry (linear array, not circular), heterogeneous tissue, and speed-of-sound mismatch from fat layers.

---

## R1. PlanAgent — Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.operator_correction,   # "speed of sound may be wrong"
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["tumor_rf.npy", "array_linear.npy"],
#   params={"n_elements": 128, "n_time_samples": 2048,
#           "sampling_rate_mhz": 40}
# )
```

---

## R2. PhotonAgent — Depth-Limited SNR

### Real tissue parameters

```python
# Deep tumor imaging: signal attenuated by tissue
# mu_eff at 532 nm in mouse tissue: ~7 cm^-1
# Tumor depth: 8 mm
# Fluence at depth: fluence_surface * exp(-mu_eff * d)
#                 = 255 * exp(-7 * 0.8) = 0.94 J/m^2

p0_deep = 0.8 * 10.0 * 0.94 = 7.52 Pa  # Much weaker than surface (204 Pa)

# Acoustic attenuation: 0.5 dB/cm/MHz, 5 MHz center, 8 mm depth
acoustic_atten = 10^(-0.5 * 0.8 * 5 / 20) = 0.794

p0_detected = 7.52 * 0.794 = 5.97 Pa

SNR_acoustic = 5.97 / 1.02 = 5.85
SNR_db = 20 * log10(5.85) = 15.3 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=5.35e16,
  snr_db=15.3,
  noise_regime=NoiseRegime.detector_limited,    # Acoustic noise floor dominates
  feasible=True,
  quality_tier="poor",                           # SNR < 20 dB
  explanation="Low acoustic SNR at 8mm depth due to optical attenuation "
              "(mu_eff=7/cm) and acoustic loss. Surface signals much stronger."
)
```

---

## R3. MismatchAgent — Speed-of-Sound Heterogeneity

```python
# Fat layer above tumor: v_fat = 1450 m/s, v_tissue = 1540 m/s
# Assumed: 1540 m/s (uniform), Actual: layered medium
psi_true = {
    "speed_of_sound": -40.0,           # 40 m/s off (fat layer)
    "transducer_position": +0.5,       # 0.5 mm registration error
    "laser_fluence": 0.25,             # Fluence 25% off from heterogeneity
}

# Severity
S = 0.45 * |40.0| / 200.0     # speed_of_sound:    0.090
  + 0.30 * |0.5| / 4.0        # transducer_pos:    0.038
  + 0.25 * |0.25| / 1.0       # laser_fluence:     0.063
S = 0.191  # Low-moderate severity

improvement_db = clip(10 * 0.191, 0, 20) = 1.91 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  severity_score=0.191,
  correction_method="grid_search",
  expected_improvement_db=1.91,
  explanation="Low-moderate mismatch. Speed-of-sound error from fat layer "
              "causes geometric distortion and defocusing. Grid search "
              "over SoS values recommended."
)
```

---

## R4. RecoverabilityAgent — Detector-Limited Regime

```python
# Calibration table: noise="detector_limited", solver="ubp"
# -> recoverability=0.65, expected_psnr=28.3 dB
RecoverabilityReport(
  recoverability_score=0.65,
  expected_psnr_db=28.3,
  verdict="sufficient",
  explanation="Recoverability limited by detector noise at depth and "
              "limited-view geometry (linear array, ~120 deg coverage)."
)
```

---

## R5. AnalysisAgent — Photon Budget is Bottleneck

```python
photon_score      = 1 - min(15.3 / 40, 1.0)   = 0.618
mismatch_score    = 0.191
compression_score = 1 - 0.65                     = 0.35
solver_score      = 0.2

primary = "photon"  # max(0.618, 0.191, 0.35, 0.2)

P = (1-0.618*0.5) * (1-0.191*0.5) * (1-0.35*0.5) * (1-0.2*0.5)
  = 0.691 * 0.904 * 0.825 * 0.90
  = 0.464
```

```python
SystemAnalysis(
  primary_bottleneck="photon",
  probability_of_success=0.464,
  overall_verdict="marginal",
  suggestions=[
    Suggestion(text="Average multiple pulses (N=64) for +9 dB acoustic SNR", priority="critical"),
    Suggestion(text="Use 1064 nm excitation for deeper penetration (mu_eff ~1/cm)", priority="high"),
    Suggestion(text="Apply model-based reconstruction with TV regularization", priority="high"),
  ]
)
```

---

## R6. AgentNegotiator — Conditional Proceed

```python
P_joint = 0.65 * 0.65 * (1 - 0.191*0.7) = 0.65 * 0.65 * 0.866 = 0.366

NegotiationResult(
  vetoes=[],
  proceed=True,         # P > 0.15
  probability_of_success=0.366
)
```

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=180.0,     # Includes SoS grid search
  proceed_recommended=True,
  warnings=[
    "Low acoustic SNR at depth (15.3 dB) — deep features may be unresolvable",
    "Speed-of-sound mismatch expected — grid search over [1400, 1600] m/s",
    "Limited-view linear array: structures tangent to array surface invisible"
  ]
)
```

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real In-Vivo |
|--------|-----------|--------------|
| **Photon Agent** | | |
| Acoustic SNR | 46.0 dB | 15.3 dB |
| Quality tier | excellent | poor |
| Depth | surface | 8 mm |
| **Mismatch Agent** | | |
| Severity | 0.106 (low) | 0.191 (low-moderate) |
| Dominant error | none | SoS heterogeneity |
| **Recoverability Agent** | | |
| Score | 0.78 (good) | 0.65 (sufficient) |
| Expected PSNR | 32.1 dB | 28.3 dB |
| **Analysis Agent** | | |
| Primary bottleneck | compression | **photon** |
| P(success) | 0.758 | 0.464 |
| **Negotiator** | | |
| P_joint | 0.686 | 0.366 |
| **PreFlight** | | |
| Runtime | 32s | 180s |
| Warnings | 0 | 3 |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (back-projection -> time-reversal -> model-based) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Adaptive:** Depth-dependent SNR analysis; system recommends pulse averaging or wavelength switching for deep imaging.
