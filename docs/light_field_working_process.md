# Light Field Imaging Working Process

## End-to-End Pipeline for Plenoptic Light Field Reconstruction

This document traces a complete light field imaging experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct a 4D light field from this plenoptic camera capture.
 Measurement: lf_raw.npy, 9x9 angular views, microlens pitch 125 um."
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "lf_raw.npy" detected
#   operator_type=OperatorType.linear_operator,
#   files=["lf_raw.npy"],
#   params={"n_angular_u": 9, "n_angular_v": 9, "microlens_pitch_um": 125}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> light_field entry
light_field:
  keywords: [light_field, plenoptic, microlens_array, 4D, refocusing, depth_estimation]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="light_field",
#   confidence=0.96,
#   reasoning="Matched keywords: light field, plenoptic, 4D, angular views"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the light_field registry entry:

```python
system = plan_agent.build_imaging_system("light_field")
# ImagingSystem(
#   modality_key="light_field",
#   display_name="Light Field Imaging",
#   signal_dims={"x": [512, 512, 9, 9], "y": [512, 512]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...4 elements...],
#   default_solver="shift_and_sum"
# )
```

**Light Field Element Chain (4 elements):**

```
Scene Illumination --> Main Lens (f/2.0) --> Microlens Array (MLA) --> CMOS Image Sensor
  throughput=1.0       throughput=0.90       throughput=0.75          throughput=0.80
  noise: none          noise: aberration     noise: alignment         noise: shot+read+fixed+quant
                       f=50mm, f/2.0         pitch=125um              pixel_size=1.4um
                                             9x9 angular              read_noise=2.0 e-
                                             fill_factor=0.98         bit_depth=12
```

**Cumulative throughput:** `1.0 x 0.90 x 0.75 x 0.80 = 0.540`

**Forward model:**
```
y(x, y) = sum_{u,v} L(x, y, u, v) * MLA(x, y, u, v)

The microlens array (MLA) maps the 4D light field L(x,y,u,v) to a 2D sensor:
- Spatial resolution: (512, 512) microlens positions
- Angular resolution: (9, 9) pixels under each microlens
- Sensor image: 512*9 x 512*9 = 4608 x 4608 raw pixels
- Each microlens integrates rays from its 9x9 sub-aperture

Compression: (512*512*9*9) / (512*512) = 81:1 angular compression per pixel
```

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  light_field:
    model_id: "generic_detector"
    parameters:
      power_w: 0.1
      wavelength_nm: 550
      na: 0.125
      qe: 0.70
      exposure_s: 0.033
  ```

### Computation

```python
# 1. Photon energy
E_photon = h * c / wavelength_nm = 6.626e-34 * 3e8 / 550e-9 = 3.61e-19 J

# 2. Collection solid angle
solid_angle = (na / n_medium)^2 / (4 * pi)
#           = (0.125 / 1.0)^2 / (4 * pi) = 0.00124

# 3. Raw photon count
N_raw = power_w * qe * solid_angle * exposure_s / E_photon
#     = 0.1 * 0.70 * 0.00124 * 0.033 / 3.61e-19
#     = 7.94e11 photons

# 4. Apply cumulative throughput
N_effective = N_raw * 0.540 = 4.29e11 photons

# 5. Per sub-aperture photon count (divided among 81 angular views)
N_per_view = N_effective / 81 = 5.30e9 photons/pixel/view

# 6. Noise variances (per sub-aperture pixel)
shot_var   = N_per_view = 5.30e9
read_var   = read_noise^2 = 4.0        # 2.0 e- read noise
total_var  = 5.30e9 + 4.0 = 5.30e9

# 7. SNR (per sub-aperture)
SNR = N_per_view / sqrt(total_var) = sqrt(5.30e9) = 72,800
SNR_db = 20 * log10(72800) = 97.2 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=5.30e9,              # Per sub-aperture pixel
  snr_db=97.2,
  noise_regime=NoiseRegime.shot_limited,    # shot_var/total_var > 0.99
  shot_noise_sigma=72800.0,
  read_noise_sigma=2.0,
  total_noise_sigma=72800.0,
  feasible=True,
  quality_tier="excellent",                 # SNR > 30 dB
  throughput_chain=[
    {"Scene Illumination": 1.0},
    {"Main Lens": 0.90},
    {"Microlens Array": 0.75},
    {"CMOS Image Sensor": 0.80}
  ],
  noise_model="poisson",
  explanation="Shot-limited regime. Excellent SNR per sub-aperture despite 81-way angular splitting."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"light_field"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  light_field:
    parameters:
      microlens_pitch_error:
        range: [-2.0, 2.0]
        typical_error: 0.3
        unit: "um"
        description: "Microlens array pitch calibration error"
      rotation:
        range: [-1.0, 1.0]
        typical_error: 0.15
        unit: "degrees"
        description: "In-plane rotation of MLA relative to sensor pixel grid"
      vignetting:
        range: [0.0, 0.35]
        typical_error: 0.08
        unit: "normalized"
        description: "Intensity falloff at sub-aperture boundaries"
    severity_weights:
      microlens_pitch_error: 0.45
      rotation: 0.30
      vignetting: 0.25
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.45 * |0.3| / 4.0      # microlens_pitch: 0.034
  + 0.30 * |0.15| / 2.0     # rotation: 0.023
  + 0.25 * |0.08| / 0.35    # vignetting: 0.057
S = 0.114  # Low severity (factory-calibrated MLA)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.14 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="light_field",
  mismatch_family="grid_search",
  parameters={
    "microlens_pitch_error": {"typical_error": 0.3, "range": [-2.0, 2.0], "weight": 0.45},
    "rotation":             {"typical_error": 0.15, "range": [-1.0, 1.0], "weight": 0.30},
    "vignetting":           {"typical_error": 0.08, "range": [0.0, 0.35], "weight": 0.25}
  },
  severity_score=0.114,
  correction_method="grid_search",
  expected_improvement_db=1.14,
  explanation="Low mismatch severity. Factory-calibrated MLA alignment. Vignetting is the largest single contributor."
)
```

---

## 5. RecoverabilityAgent -- Can We Reconstruct?

**File:** `agents/recoverability_agent.py` (912 lines)

### Input
- `ImagingSystem` (signal_dims for CR calculation)
- `PhotonReport` (noise regime)
- Calibration table from `compression_db.yaml`:
  ```yaml
  light_field:
    signal_prior_class: "low_rank"
    entries:
      - {cr: 0.012, noise: "shot_limited", solver: "learning_based_lf",
         recoverability: 0.82, expected_psnr_db: 33.5,
         provenance: {dataset_id: "stanford_lytro_lf_2023", ...}}
      - {cr: 0.012, noise: "shot_limited", solver: "tv_lf",
         recoverability: 0.68, expected_psnr_db: 28.1, ...}
      - {cr: 0.012, noise: "read_limited", solver: "learning_based_lf",
         recoverability: 0.73, expected_psnr_db: 30.2, ...}
  ```

### Computation

```python
# 1. Compression ratio
CR = prod(y_shape) / prod(x_shape)
#  = (512 * 512) / (512 * 512 * 9 * 9)
#  = 1 / 81 = 0.012

# 2. Operator diversity
#    Each microlens captures a distinct angular perspective
#    With 9x9 views, the angular sampling provides high diversity
diversity = 0.90  # High angular diversity (81 views)

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.526

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="shift_and_sum", cr=0.012
#    Closest: learning_based_lf at cr=0.012
#    -> recoverability=0.82, expected_psnr=33.5 dB

# 5. Best solver selection
#    learning_based_lf: 33.5 dB > tv_lf: 28.1 dB
#    Default: shift_and_sum (simple but effective baseline)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.012,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.low_rank,
  operator_diversity_score=0.90,
  condition_number_proxy=0.526,
  recoverability_score=0.82,
  recoverability_confidence=1.0,
  expected_psnr_db=33.5,
  expected_psnr_uncertainty_db=1.2,
  recommended_solver_family="shift_and_sum",
  verdict="good",                  # 0.70 <= score < 0.85
  calibration_table_entry={...},
  explanation="Good recoverability despite 81:1 compression. Light fields have intrinsic "
              "low-rank structure (angular smoothness) that makes recovery tractable. "
              "Learning-based methods expected 33.5 dB on Stanford/Lytro benchmark."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(97.2 / 40, 1.0)  = 0.0     # Excellent SNR
mismatch_score    = 0.114                      = 0.114   # Low
compression_score = 1 - 0.82                   = 0.18    # Good recoverability
solver_score      = 0.2                        = 0.2     # Default placeholder

# Primary bottleneck
primary = "solver"  # max(0.0, 0.114, 0.18, 0.2) = solver

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.114*0.5) * (1 - 0.18*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.943 * 0.91 * 0.90
  = 0.772
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="solver",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.114, compression=0.18, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Use learning-based LF reconstruction for +5.4 dB over shift-and-sum",
      priority="medium",
      expected_gain_db=5.4
    ),
    Suggestion(
      text="LFBM5D denoising can improve angular consistency by +2 dB",
      priority="medium",
      expected_gain_db=2.0
    )
  ],
  overall_verdict="good",             # 0.60 <= P < 0.80
  probability_of_success=0.772,
  explanation="System is well-configured. Solver choice is the primary bottleneck. "
              "Shift-and-sum is fast but does not exploit angular coherence."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="good" | No veto |
| Severe mismatch without correction | severity=0.114 < 0.7 | No veto |
| All marginal | All excellent/good | No veto |
| Joint probability floor | P=0.772 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95   # tier_prob["excellent"]
P_recoverability = 0.82   # recoverability_score
P_mismatch       = 1.0 - 0.114 * 0.7 = 0.920

P_joint = 0.95 * 0.82 * 0.920 = 0.717
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.717
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 512 * 512 * 9 * 9 = 21,233,664
dim_factor   = total_pixels / (512 * 512) = 81.0
solver_complexity = 1.0  # shift_and_sum (simple summation)
cr_factor    = max(0.012, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 81.0 * 1.0 * 0.125 = 20.25 seconds
# shift_and_sum is O(n_views), LFBM5D is much slower (~300s)
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="light_field", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=20.25,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=["raw plenoptic sensor image (2D, H_sensor x W_sensor)",
                  "optional: microlens center calibration, white image"]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Light field forward model:
#   y(x, y) = sum_{u,v} L(x, y, u, v) * MLA(x, y, u, v)
#
# The microlens array (MLA) maps the 4D light field to a 2D sensor:
#   - Each microlens at position (x, y) captures a 9x9 sub-image
#   - Pixel (u, v) under microlens (x, y) records L(x, y, u, v)
#   - Total sensor: 4608 x 4608 pixels (512*9 x 512*9)
#
# Parameters:
#   microlens_pitch: 125 um (9 pixels @ 1.4 um/pixel = 12.6 um effective pitch)
#   n_angular: [9, 9]
#   spatial_res: [512, 512]
#
# Input:  L = (512, 512, 9, 9) 4D light field (full plenoptic function)
# Output: y = (512, 512) 2D sensor image (angular integration per microlens)

class LightFieldOperator(PhysicsOperator):
    def forward(self, L):
        """y(x,y) = sum_{u,v} L(x,y,u,v)"""
        return np.sum(L, axis=(2, 3)) / (self.nu * self.nv)

    def adjoint(self, y):
        """L_hat(x,y,u,v) = y(x,y) for all u,v (replicate)"""
        L = np.zeros((self.sx, self.sy, self.nu, self.nv))
        L[:, :, :, :] = y[:, :, np.newaxis, np.newaxis]
        return L

    def check_adjoint(self):
        """Verify <Ax, y> ~= <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided lf_raw.npy:
lf_raw = np.load("lf_raw.npy")              # (4608, 4608) raw sensor
light_field = demux_lf(lf_raw, nu=9, nv=9)  # -> (512, 512, 9, 9)

# If simulating from depth + texture:
x_true_texture = load_texture()               # (512, 512) base image
depth_map = load_depth()                      # (512, 512) depth values
light_field = synthesize_lf(x_true_texture, depth_map, nu=9, nv=9)
# Each view shifted by: du = (u - cu) * depth(x,y) * baseline
y = operator.forward(light_field)             # (512, 512)
y += np.random.poisson(y)                     # Shot noise
```

### Step 9c: Reconstruction with Shift-and-Sum

```python
from pwm_core.recon.light_field_solver import shift_and_sum, lfbm5d

# Shift-and-Sum: refocus at a target depth by shifting and averaging views
x_hat = shift_and_sum(light_field)
# x_hat shape: (64, 64) -- central view refocused image
# Expected PSNR: ~28.0 dB (benchmark reference)
```

**Shift-and-Sum Algorithm:**

```python
def shift_and_sum(light_field, refocus_depth=0.0):
    """Refocus light field at target depth via disparity-compensated sum.

    For each angular view (u, v):
      1. Compute disparity: d(u,v) = (u - cu, v - cv) * refocus_depth
      2. Shift view by -d(u,v) to align at target depth
      3. Accumulate shifted views

    All-in-focus: sum all views with zero shift (refocus_depth=0)
    """
    sx, sy, nu, nv = light_field.shape
    cu, cv = nu // 2, nv // 2
    result = np.zeros((sx, sy))
    for u in range(nu):
        for v in range(nv):
            du = (u - cu) * refocus_depth
            dv = (v - cv) * refocus_depth
            shifted = ndi_shift(light_field[:, :, u, v], [-du, -dv])
            result += shifted
    return result / (nu * nv)
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| Shift-and-Sum | Traditional | ~28.0 dB | No | `shift_and_sum(light_field)` |
| LFBM5D | Traditional | ~31.0 dB | No | `lfbm5d(light_field, sigma=0.02)` |
| Learning-Based LF | Deep Learning | ~33.5 dB | Yes | `learning_based_lf(light_field)` |
| TV-LF (ADMM) | Traditional | ~28.1 dB | No | `tv_lf(light_field, lam=0.01, n_iters=300)` |

### Step 9d: Metrics

```python
# PSNR (central view refocused)
psnr = 10 * log10(max_val^2 / mse(x_hat, x_true))  # ~28.0 dB

# SSIM (structural similarity of refocused view)
ssim = structural_similarity(x_hat, x_true, data_range=1.0)

# Angular Consistency (light-field-specific)
# Measures smoothness across angular dimension via EPI slope
epi_consistency = compute_epi_slope_error(light_field_recon, light_field_gt)

# Depth Estimation Error (derived metric)
depth_recon = estimate_depth_from_lf(light_field_recon)
depth_mae = mean(abs(depth_recon - depth_gt))
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
|   +-- light_field.npy    # 4D light field (512, 512, 9, 9) + SHA256 hash
|   +-- y.npy              # Raw sensor image (512, 512) + SHA256 hash
|   +-- x_hat.npy          # Refocused reconstruction (512, 512) + SHA256 hash
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
+-- metrics.json           # PSNR, SSIM, angular consistency, depth MAE
+-- operator.json          # Operator params (MLA pitch, angular grid, spatial res)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized light field pipeline with factory-calibrated microlens array alignment. In practice, real plenoptic cameras have microlens array (MLA) misalignment from thermal expansion, in-plane rotation from imperfect mounting, and vignetting at sub-aperture boundaries from chief ray angle mismatch.

This section traces the same pipeline with realistic parameters from a laboratory Lytro-style plenoptic camera.

---

## Real Experiment: User Prompt

```
"I have a raw capture from our custom plenoptic camera. The microlens
 array was repositioned after maintenance. Sub-images look slightly
 rotated. Measurement: raw_capture.npy, 9x9 angular, pitch=125um."
```

**Key difference:** MLA repositioned after maintenance, visible rotation in sub-images. Microlens center calibration is stale.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["raw_capture.npy"],
#   params={"n_angular_u": 9, "n_angular_v": 9, "microlens_pitch_um": 125}
# )
```

---

## R2. PhotonAgent -- Lab Prototype Conditions

### Real detector parameters

```yaml
# Lab plenoptic camera: lower light, manual exposure
light_field_lab:
  power_w: 0.01                  # 10x dimmer (indoor LED illumination)
  wavelength_nm: 550
  na: 0.125
  qe: 0.65                      # Slightly degraded sensor
  exposure_s: 0.033
```

### Computation

```python
N_raw = 0.01 * 0.65 * 0.00124 * 0.033 / 3.61e-19 = 7.37e9

N_effective = 7.37e9 * 0.540 = 3.98e9
N_per_view = 3.98e9 / 81 = 4.91e7

shot_var   = 4.91e7
read_var   = 4.0
total_var  = 4.91e7

SNR = sqrt(4.91e7) = 7007
SNR_db = 20 * log10(7007) = 76.9 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=4.91e7,
  snr_db=76.9,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="excellent",
  explanation="Shot-limited. Excellent SNR even with indoor illumination."
)
```

---

## R3. MismatchAgent -- MLA Repositioning Error

```python
# Actual errors from MLA repositioning
mismatch_actual = {
    "microlens_pitch_error": 1.2,    # um (4x typical, from thermal shift)
    "rotation": 0.65,                 # degrees (4x typical, from mounting)
    "vignetting": 0.20,              # normalized (2.5x typical)
}

# Severity computation
S = 0.45 * |1.2| / 4.0        # pitch: 0.135
  + 0.30 * |0.65| / 2.0       # rotation: 0.098
  + 0.25 * |0.20| / 0.35      # vignetting: 0.143
S = 0.376  # MODERATE-HIGH severity

improvement_db = clip(10 * 0.376, 0, 20) = 3.76 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="light_field",
  severity_score=0.376,
  correction_method="grid_search",
  expected_improvement_db=3.76,
  explanation="Moderate-high mismatch. MLA pitch error (1.2 um) and rotation (0.65 deg) "
              "cause sub-image misalignment. Recalibration of microlens centers required."
)
```

---

## R4. RecoverabilityAgent -- Degraded by MLA Error

```python
# Calibration table lookup (read_limited proxy for structured noise from misalignment)
# -> recoverability=0.73, expected_psnr=30.2 dB
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.012,
  recoverability_score=0.73,               # Down from 0.82
  expected_psnr_db=30.2,                   # Down from 33.5
  verdict="good",                          # Still good
  explanation="Recoverability reduced by MLA misalignment. Sub-image demultiplexing "
              "errors degrade angular resolution."
)
```

---

## R5. AnalysisAgent -- MLA Alignment is the Bottleneck

```python
# Bottleneck scores
photon_score      = 0.0        # Excellent
mismatch_score    = 0.376      # Moderate-high
compression_score = 0.27       # Good
solver_score      = 0.2

primary = "mismatch"  # 0.376 dominates

P = (1 - 0.0*0.5) * (1 - 0.376*0.5) * (1 - 0.27*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.812 * 0.865 * 0.90
  = 0.632
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="mismatch",
  probability_of_success=0.632,
  overall_verdict="good",
  suggestions=[
    Suggestion(text="Recalibrate microlens center positions from white image", priority="critical", expected_gain_db=3.0),
    Suggestion(text="Apply sub-pixel MLA rotation correction", priority="high", expected_gain_db=1.5),
    Suggestion(text="Flat-field vignetting correction from white calibration image", priority="medium", expected_gain_db=0.8)
  ]
)
```

---

## R6. AgentNegotiator -- Conditional Proceed

```python
P_photon         = 0.95
P_recoverability = 0.73
P_mismatch       = 1.0 - 0.376 * 0.7 = 0.737

P_joint = 0.95 * 0.73 * 0.737 = 0.511
```

| Condition | Check | Result |
|-----------|-------|--------|
| Severe mismatch | severity=0.376 < 0.7 | No veto |
| Joint probability floor | P=0.511 > 0.15 | No veto |

```python
NegotiationResult(
  vetoes=[],
  proceed=True,
  probability_of_success=0.511
)
```

---

## R7. PreFlightReportBuilder -- Warnings Raised

```python
PreFlightReport(
  estimated_runtime_s=45.0,
  proceed_recommended=True,
  warnings=[
    "Mismatch severity 0.376 -- MLA alignment may be compromised",
    "Recalibrate microlens centers from white image before reconstruction",
    "Expected PSNR degraded from 33.5 to 30.2 dB without correction"
  ]
)
```

---

## R8. Pipeline Runner -- With MLA Recalibration

### Step R8a: Reconstruct with Stale Calibration

```python
light_field_wrong = demux_lf(raw_capture, centers=stale_centers, nu=9, nv=9)
x_wrong = shift_and_sum(light_field_wrong)
# PSNR = 22.5 dB  <-- visible sub-image ghosting from misaligned demux
```

### Step R8b: Recalibrate from White Image

```python
# White image: uniform scene captured through MLA reveals microlens centers
white_img = np.load("white_calibration.npy")
new_centers = find_microlens_centers(white_img, pitch_um=125, pixel_size_um=1.4)
# Detected rotation: 0.63 degrees (close to actual 0.65)
# Pitch correction: +1.1 um (close to actual 1.2)

light_field_cal = demux_lf(raw_capture, centers=new_centers, nu=9, nv=9)
x_cal = shift_and_sum(light_field_cal)
# PSNR = 27.8 dB  <-- +5.3 dB from correct demultiplexing
```

### Step R8c: LFBM5D with Calibrated Demux

```python
x_bm5d = lfbm5d(light_field_cal, sigma=0.02)
# PSNR = 30.8 dB  <-- exploits angular redundancy for denoising
```

### Step R8d: Final Comparison

| Configuration | Shift-and-Sum | LFBM5D | Learning-Based |
|---------------|---------------|--------|----------------|
| Ideal (factory calibrated) | 28.0 dB | 31.0 dB | 33.5 dB |
| Stale calibration (wrong centers) | 22.5 dB | 24.0 dB | 26.5 dB |
| Recalibrated (white image) | 27.8 dB | 30.8 dB | 32.8 dB |

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Experiment |
|--------|-----------|-----------------|
| **Photon Agent** | | |
| N_per_view | 5.30e9 | 4.91e7 |
| SNR | 97.2 dB | 76.9 dB |
| Quality tier | excellent | excellent |
| **Mismatch Agent** | | |
| Severity | 0.114 (low) | 0.376 (moderate-high) |
| Dominant error | vignetting | **MLA pitch + rotation** |
| Correction needed | No | **Yes** |
| **Recoverability Agent** | | |
| Score | 0.82 (good) | 0.73 (good) |
| Expected PSNR | 33.5 dB | 30.2 dB |
| **Analysis Agent** | | |
| Primary bottleneck | solver | **mismatch** |
| P(success) | 0.772 | 0.632 |
| **Negotiator** | | |
| P_joint | 0.717 | 0.511 |
| **Pipeline** | | |
| Without recalibration | 28.0 dB | 22.5 dB |
| With recalibration | -- | **27.8 dB** |
| With recalibration + LFBM5D | -- | **30.8 dB** |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (Shift-and-Sum -> LFBM5D -> Learning-Based) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **4D-aware:** The pipeline correctly handles the 4D->2D light field compression, where angular structure (low-rank prior) enables reconstruction from extreme compression ratios (81:1).
