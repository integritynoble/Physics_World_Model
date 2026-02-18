# Integral Photography Working Process

## End-to-End Pipeline for Integral Imaging 3D Reconstruction

This document traces a complete integral photography experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct a 3D depth volume from this integral imaging capture.
 Elemental images: elemental_array.npy, microlens pitch 150 um,
 16 depth planes, 512x512 spatial resolution."
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "elemental_array.npy" detected
#   operator_type=OperatorType.linear_operator,
#   files=["elemental_array.npy"],
#   params={"microlens_pitch_um": 150, "n_depths": 16, "spatial_res": [512, 512]}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> integral entry
integral:
  keywords: [integral_photography, elemental_images, microlens, 3D_display, depth_estimation]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="integral",
#   confidence=0.95,
#   reasoning="Matched keywords: integral, elemental images, depth, 3D"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the integral registry entry:

```python
system = plan_agent.build_imaging_system("integral")
# ImagingSystem(
#   modality_key="integral",
#   display_name="Integral Photography",
#   signal_dims={"x": [512, 512, 64], "y": [512, 512]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...4 elements...],
#   default_solver="depth_estimation"
# )
```

**Integral Photography Element Chain (4 elements):**

```
Scene Illumination --> Main Lens (f/2.8) --> Microlens Array --> CMOS Image Sensor
  throughput=1.0       throughput=0.90       throughput=0.72     throughput=0.80
  noise: none          noise: aberration     noise: alignment    noise: shot+read+fixed+quant
                       f=50mm, f/2.8         pitch=150um         pixel_size=1.4um
                                             64 depth planes     read_noise=2.0 e-
                                             fill_factor=0.95    bit_depth=12
```

**Cumulative throughput:** `1.0 x 0.90 x 0.72 x 0.80 = 0.518`

**Forward model:**
```
I(x, y) = integral L(x, y, u, v) * T(u, v) dudv

Equivalently, as a depth-weighted sum with defocus:
  y(x, y) = sum_d  w_d * conv(x_d, G_{sigma_d})(x, y) + noise

where:
  x_d      = scene intensity at depth plane d
  w_d      = depth weight (Gaussian centered at focus depth)
  G_{sigma_d} = defocus PSF (Gaussian with sigma proportional to |d - d_focus|)
  sigma_d  = |d - d_focus| * 0.8 + 0.5 pixels

Input:  x = (512, 512, 64) depth volume
Output: y = (512, 512) 2D elemental composite image
Compression: 512*512*64 / (512*512) = 64:1
```

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  integral:
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
E_photon = h * c / wavelength_nm = 3.61e-19 J

# 2. Collection solid angle
solid_angle = (na / n_medium)^2 / (4 * pi)
#           = (0.125 / 1.0)^2 / (4 * pi) = 0.00124

# 3. Raw photon count
N_raw = power_w * qe * solid_angle * exposure_s / E_photon
#     = 0.1 * 0.70 * 0.00124 * 0.033 / 3.61e-19
#     = 7.94e11 photons

# 4. Apply cumulative throughput
N_effective = N_raw * 0.518 = 4.11e11 photons/pixel

# 5. Noise variances
shot_var   = N_effective = 4.11e11
read_var   = read_noise^2 = 4.0           # 2.0 e-
total_var  = 4.11e11 + 4.0 = 4.11e11

# 6. SNR
SNR = N_effective / sqrt(total_var) = sqrt(4.11e11) = 641,093
SNR_db = 20 * log10(641093) = 116.1 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=4.11e11,
  snr_db=116.1,
  noise_regime=NoiseRegime.shot_limited,     # shot_var/total_var > 0.99
  shot_noise_sigma=641093.0,
  read_noise_sigma=2.0,
  total_noise_sigma=641093.0,
  feasible=True,
  quality_tier="excellent",                  # SNR > 30 dB
  throughput_chain=[
    {"Scene Illumination": 1.0},
    {"Main Lens": 0.90},
    {"Microlens Array": 0.72},
    {"CMOS Image Sensor": 0.80}
  ],
  noise_model="poisson",
  explanation="Shot-limited regime. Excellent SNR. Photon budget is not a concern for integral imaging."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"integral"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  integral:
    parameters:
      microlens_pitch:
        range: [-2.0, 2.0]
        typical_error: 0.25
        unit: "um"
        description: "Microlens pitch error from array fabrication tolerance"
      rotation:
        range: [-1.0, 1.0]
        typical_error: 0.1
        unit: "degrees"
        description: "In-plane rotation of MLA relative to display pixel grid"
      depth_range:
        range: [-50.0, 50.0]
        typical_error: 10.0
        unit: "mm"
        description: "Depth reconstruction range error from incorrect focal length"
    severity_weights:
      microlens_pitch: 0.40
      rotation: 0.30
      depth_range: 0.30
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.40 * |0.25| / 4.0      # microlens_pitch: 0.025
  + 0.30 * |0.1|  / 2.0      # rotation: 0.015
  + 0.30 * |10.0| / 100.0    # depth_range: 0.030
S = 0.070  # Very low severity (well-calibrated system)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 0.70 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="integral",
  mismatch_family="grid_search",
  parameters={
    "microlens_pitch": {"typical_error": 0.25, "range": [-2.0, 2.0], "weight": 0.40},
    "rotation":        {"typical_error": 0.1, "range": [-1.0, 1.0], "weight": 0.30},
    "depth_range":     {"typical_error": 10.0, "range": [-50.0, 50.0], "weight": 0.30}
  },
  severity_score=0.070,
  correction_method="grid_search",
  expected_improvement_db=0.70,
  explanation="Very low mismatch severity. Well-calibrated integral imaging system."
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
  integral:
    signal_prior_class: "low_rank"
    entries:
      - {cr: 0.03, noise: "shot_limited", solver: "learning_based_integral",
         recoverability: 0.78, expected_psnr_db: 31.5,
         provenance: {dataset_id: "hci_4d_lf_benchmark_2023", ...}}
      - {cr: 0.03, noise: "shot_limited", solver: "tv_integral",
         recoverability: 0.65, expected_psnr_db: 27.0, ...}
      - {cr: 0.03, noise: "read_limited", solver: "learning_based_integral",
         recoverability: 0.68, expected_psnr_db: 28.3, ...}
  ```

### Computation

```python
# 1. Compression ratio
CR = prod(y_shape) / prod(x_shape)
#  = (512 * 512) / (512 * 512 * 64)
#  = 1 / 64 = 0.016
#  (benchmark reference uses 0.03 for the calibration table)

# 2. Operator diversity
#    Each elemental image captures a different angular perspective
#    With 64 depth planes, the depth diversity is high
diversity = 0.80  # Good diversity (depth-dependent PSF provides structure)

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.556

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="depth_estimation", cr~0.03
#    Closest: learning_based_integral at cr=0.03
#    -> recoverability=0.78, expected_psnr=31.5 dB

# 5. Best solver selection
#    learning_based_integral: 31.5 dB > tv_integral: 27.0 dB
#    Default: depth_estimation (traditional, fast)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.016,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.low_rank,
  operator_diversity_score=0.80,
  condition_number_proxy=0.556,
  recoverability_score=0.78,
  recoverability_confidence=1.0,
  expected_psnr_db=31.5,
  expected_psnr_uncertainty_db=1.5,
  recommended_solver_family="depth_estimation",
  verdict="good",                  # 0.70 <= score < 0.85
  calibration_table_entry={...},
  explanation="Good recoverability. Integral imaging has inherent depth-angular structure "
              "that enables 3D reconstruction from a single 2D capture. Learning-based methods "
              "expected 31.5 dB on HCI 4D LF benchmark."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(116.1 / 40, 1.0) = 0.0     # Excellent SNR
mismatch_score    = 0.070                      = 0.070   # Very low
compression_score = 1 - 0.78                   = 0.22    # Good recoverability
solver_score      = 0.2                        = 0.2     # Default placeholder

# Primary bottleneck
primary = "compression"  # max(0.0, 0.070, 0.22, 0.2) = compression

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.070*0.5) * (1 - 0.22*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.965 * 0.89 * 0.90
  = 0.773
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="compression",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.070, compression=0.22, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Use learning-based integral reconstruction for +4.5 dB over depth estimation",
      priority="medium",
      expected_gain_db=4.5
    ),
    Suggestion(
      text="DIBR iterative refinement may improve depth consistency by +2 dB",
      priority="medium",
      expected_gain_db=2.0
    )
  ],
  overall_verdict="good",             # 0.60 <= P < 0.80
  probability_of_success=0.773,
  explanation="System is well-configured. 64:1 compression is the primary limiting factor. "
              "Depth-angular low-rank structure makes reconstruction tractable."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="good" | No veto |
| Severe mismatch without correction | severity=0.070 < 0.7 | No veto |
| All marginal | All excellent/good | No veto |
| Joint probability floor | P=0.773 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95   # tier_prob["excellent"]
P_recoverability = 0.78   # recoverability_score
P_mismatch       = 1.0 - 0.070 * 0.7 = 0.951

P_joint = 0.95 * 0.78 * 0.951 = 0.704
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.704
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 512 * 512 * 64 = 16,777,216
dim_factor   = total_pixels / (512 * 512) = 64.0
solver_complexity = 1.5  # Depth estimation (deconvolution per plane)
cr_factor    = max(0.016, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 64.0 * 1.5 * 0.125 = 24.0 seconds
# Depth estimation: ~24s, DIBR iterative: ~120s (50 iterations)
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="integral", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=24.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=["elemental image array (2D sensor capture)",
                  "optional: microlens calibration, depth range"]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Integral imaging forward model:
#   y(x,y) = sum_d w_d * conv(x_d, G_{sigma_d})(x,y) + noise
#
# Parameters:
#   n_depths: 16 depth planes (benchmark) or 64 (full system)
#   focus_depth: center plane (d = n_depths // 2)
#   psf_sigmas: |d - d_focus| * 0.8 + 0.5 (depth-dependent defocus)
#   depth_weights: Gaussian centered at 0.5, sigma=0.15
#
# Input:  x = (H, W, n_depths) depth-weighted volume
# Output: y = (H, W) 2D integrated image

class IntegralOperator(PhysicsOperator):
    def forward(self, x):
        """y(r,c) = sum_d w_d * conv(x[:,:,d], G_{sigma_d})"""
        y = np.zeros((self.H, self.W))
        for d in range(self.n_depths):
            blurred = gaussian_filter(x[:, :, d], sigma=self.psf_sigmas[d])
            y += self.depth_weights[d] * blurred
        return y

    def adjoint(self, y):
        """x_hat(r,c,d) = w_d * conv(y, G_{sigma_d})"""
        x = np.zeros((self.H, self.W, self.n_depths))
        for d in range(self.n_depths):
            x[:, :, d] = self.depth_weights[d] * gaussian_filter(y, sigma=self.psf_sigmas[d])
        return x

    def check_adjoint(self):
        """Verify <Ax, y> ~= <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-8)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided elemental_array.npy:
y = np.load("elemental_array.npy")    # (512, 512) integrated image

# If simulating:
base_image = load_ground_truth()       # (64, 64) scene texture
depth_weights = gaussian_weights(n_depths=16, center=0.5, sigma=0.15)
x_true = np.zeros((64, 64, 16))
for d in range(16):
    x_true[:, :, d] = depth_weights[d] * base_image

y = operator.forward(x_true)           # (64, 64)
y += np.random.randn(64, 64) * 0.01   # Gaussian noise
```

### Step 9c: Reconstruction with Depth Estimation

```python
from pwm_core.recon.integral_solver import depth_estimation, dibr

# Depth estimation: deconvolution per depth plane using known PSF
x_hat = depth_estimation(
    y=y,                           # (64, 64) measurement
    depth_weights=depth_weights,   # (16,) Gaussian weights
    psf_sigmas=psf_sigmas,         # (16,) per-plane defocus
    regularization=0.001,
)
# x_hat shape: (64, 64, 16) -- reconstructed depth volume
# Expected PSNR: ~27.0 dB (benchmark reference)
```

**Depth Estimation Algorithm:**

```python
def depth_estimation(y, depth_weights, psf_sigmas, regularization=0.001):
    """Per-plane Wiener deconvolution from integrated measurement.

    For each depth d:
      1. Compute defocused version: y_d = conv(y, G_{sigma_d})
      2. Apply Wiener filter in Fourier domain:
         X_d(f) = W_d(f) * Y(f) * conj(H_d(f)) / (|H_d(f)|^2 + lambda)
      3. Apply depth weight: x_d = w_d * IFFT(X_d)

    where H_d is the Fourier transform of PSF at depth d.
    """
    h, w = y.shape
    Y = np.fft.fft2(y)
    x = np.zeros((h, w, len(depth_weights)))
    for d, (w_d, sigma_d) in enumerate(zip(depth_weights, psf_sigmas)):
        H_d = gaussian_ft(sigma_d, h, w)          # PSF in Fourier domain
        X_d = w_d * Y * np.conj(H_d) / (np.abs(H_d)**2 + regularization)
        x[:, :, d] = np.real(np.fft.ifft2(X_d))
    return np.clip(x, 0, None)
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| Depth Estimation | Traditional | ~27.0 dB | No | `depth_estimation(y, weights, sigmas)` |
| DIBR Iterative | Traditional | ~29.0 dB | No | `dibr(y, weights, sigmas, n_iters=50)` |
| Learning-Based | Deep Learning | ~31.5 dB | Yes | `learning_based_integral(y, weights)` |
| TV-Integral | Traditional | ~27.0 dB | No | `tv_integral(y, weights, sigmas, lam=0.01)` |

### Step 9d: Metrics

```python
# Volumetric PSNR (across all depth planes)
psnr = 10 * log10(max_val^2 / mse(x_hat, x_true))  # ~27.0 dB

# SSIM (average over depth-collapsed views)
ssim_2d = structural_similarity(x_hat.mean(axis=2), x_true.mean(axis=2))

# Depth estimation accuracy (integral-specific)
depth_pred = argmax(x_hat, axis=2) / n_depths
depth_true = argmax(x_true, axis=2) / n_depths
depth_mae = mean(abs(depth_pred - depth_true))

# Angular consistency (cross-depth smoothness)
angular_smooth = mean([ssim(x_hat[:,:,d], x_hat[:,:,d+1]) for d in range(n_depths-1)])
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
|   +-- y.npy              # Integrated image (512, 512) + SHA256 hash
|   +-- x_hat.npy          # Depth volume (512, 512, 64) + SHA256 hash
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
|   +-- depth_map.npy      # Estimated depth map (512, 512) + SHA256 hash
+-- metrics.json           # PSNR, SSIM, depth MAE, angular consistency
+-- operator.json          # Operator params (MLA pitch, n_depths, PSF sigmas, weights)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized integral imaging pipeline with perfect calibration. In practice, real integral imaging systems suffer from microlens pitch fabrication errors, MLA-to-sensor rotation from mounting tolerance, and incorrect depth range assumptions from imprecise focal length knowledge.

This section traces the same pipeline with realistic parameters from a custom laboratory integral camera.

---

## Real Experiment: User Prompt

```
"I have an integral image from our custom microlens array camera. The
 array was replaced with a new batch and the pitch may differ slightly
 from the previous one. Also the depth range might be off since we changed
 the main lens. Measurement: integral_lab.npy, pitch~150um, 16 depths."
```

**Key difference:** New MLA batch (pitch uncertainty), new main lens (depth range uncertainty). Both manufacturing tolerance and system reconfiguration introduce mismatch.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["integral_lab.npy"],
#   params={"microlens_pitch_um": 150, "n_depths": 16}
# )
```

---

## R2. PhotonAgent -- Lab Conditions

### Real detector parameters

```yaml
# Lab integral camera: dimmer illumination, shorter exposure
integral_lab:
  power_w: 0.02                  # 5x dimmer (indoor setting)
  wavelength_nm: 550
  na: 0.125
  qe: 0.65                      # Older sensor
  exposure_s: 0.016              # Faster exposure to avoid vibration
```

### Computation

```python
N_raw = 0.02 * 0.65 * 0.00124 * 0.016 / 3.61e-19 = 7.14e8

N_effective = 7.14e8 * 0.518 = 3.70e8

shot_var   = 3.70e8
read_var   = 4.0
total_var  = 3.70e8

SNR = sqrt(3.70e8) = 19,235
SNR_db = 20 * log10(19235) = 85.7 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=3.70e8,
  snr_db=85.7,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="excellent",
  explanation="Shot-limited. Excellent SNR despite reduced illumination."
)
```

---

## R3. MismatchAgent -- New MLA Batch + New Lens

```python
# Actual errors from hardware changes
mismatch_actual = {
    "microlens_pitch": 0.8,       # um (3.2x typical, new batch tolerance)
    "rotation": 0.45,              # degrees (4.5x typical, manual mounting)
    "depth_range": 30.0,           # mm (3x typical, new lens focal length)
}

# Severity computation
S = 0.40 * |0.8|  / 4.0        # pitch: 0.080
  + 0.30 * |0.45| / 2.0        # rotation: 0.068
  + 0.30 * |30.0| / 100.0      # depth_range: 0.090
S = 0.238  # MODERATE severity

improvement_db = clip(10 * 0.238, 0, 20) = 2.38 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="integral",
  severity_score=0.238,
  correction_method="grid_search",
  expected_improvement_db=2.38,
  explanation="Moderate mismatch. Depth range error (30mm) from new lens is the largest contributor, "
              "followed by MLA pitch shift from new batch. Grid search correction recommended."
)
```

---

## R4. RecoverabilityAgent -- Degraded by Depth Range Error

```python
# Calibration table lookup (read_limited proxy for model mismatch)
# -> recoverability=0.68, expected_psnr=28.3 dB
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.016,
  recoverability_score=0.68,               # Down from 0.78
  expected_psnr_db=28.3,                   # Down from 31.5
  verdict="sufficient",                    # 0.60 <= score < 0.70
  explanation="Recoverability degraded by depth range error. Incorrect PSF model "
              "produces out-of-focus artifacts in depth reconstruction."
)
```

---

## R5. AnalysisAgent -- Depth Range is the Bottleneck

```python
# Bottleneck scores
photon_score      = 0.0        # Excellent
mismatch_score    = 0.238      # Moderate
compression_score = 0.32       # Sufficient
solver_score      = 0.2

primary = "compression"  # max(0.0, 0.238, 0.32, 0.2)
# Root cause: depth range error inflates compression bottleneck

P = (1 - 0.0*0.5) * (1 - 0.238*0.5) * (1 - 0.32*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.881 * 0.84 * 0.90
  = 0.666
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="compression",
  probability_of_success=0.666,
  overall_verdict="good",
  suggestions=[
    Suggestion(text="Recalibrate depth range using known calibration target", priority="high", expected_gain_db=2.0),
    Suggestion(text="Measure actual microlens pitch from microscope image", priority="medium", expected_gain_db=0.8),
    Suggestion(text="Use DIBR iterative solver for depth consistency", priority="medium", expected_gain_db=2.0)
  ]
)
```

---

## R6. AgentNegotiator -- Conditional Proceed

```python
P_photon         = 0.95
P_recoverability = 0.68
P_mismatch       = 1.0 - 0.238 * 0.7 = 0.833

P_joint = 0.95 * 0.68 * 0.833 = 0.538
```

| Condition | Check | Result |
|-----------|-------|--------|
| Severe mismatch | severity=0.238 < 0.7 | No veto |
| Joint probability floor | P=0.538 > 0.15 | No veto |

```python
NegotiationResult(
  vetoes=[],
  proceed=True,
  probability_of_success=0.538
)
```

---

## R7. PreFlightReportBuilder -- Warnings Raised

```python
PreFlightReport(
  estimated_runtime_s=35.0,
  proceed_recommended=True,
  warnings=[
    "Mismatch severity 0.238 -- depth range may be miscalibrated",
    "New MLA batch: verify pitch before high-precision depth estimation",
    "Expected PSNR degraded from 31.5 to 28.3 dB without correction"
  ]
)
```

---

## R8. Pipeline Runner -- With Calibration Correction

### Step R8a: Reconstruct with Nominal Parameters

```python
x_wrong = depth_estimation(y_lab, depth_weights, psf_sigmas_nominal, regularization=0.001)
# PSNR = 23.5 dB  <-- depth planes misaligned from incorrect PSF model
```

### Step R8b: Grid Search for Correct Depth Range

```python
# Sweep depth range: [d_min, d_max] over grid
# For each candidate, compute reconstruction + cost function
best_depth_range = grid_search_depth_range(
    y_lab, candidates=np.linspace(-50, 50, 21),  # 21 depth offsets
    solver=depth_estimation, metric="sharpness"
)
# Found: depth_offset = +28mm (actual: 30mm, error: 2mm)

x_corrected = depth_estimation(y_lab, depth_weights, psf_sigmas_corrected, regularization=0.001)
# PSNR = 26.8 dB  <-- +3.3 dB from corrected depth model
```

### Step R8c: DIBR with Corrected Parameters

```python
x_dibr = dibr(y_lab, depth_weights, psf_sigmas_corrected, n_iters=50, regularization=0.001)
# PSNR = 28.5 dB  <-- +1.7 dB from iterative refinement
```

### Step R8d: Final Comparison

| Configuration | Depth Est. | DIBR | Learning-Based |
|---------------|------------|------|----------------|
| Ideal (calibrated) | 27.0 dB | 29.0 dB | 31.5 dB |
| Nominal (wrong depth) | 23.5 dB | 24.8 dB | 27.0 dB |
| Grid-search corrected | 26.8 dB | 28.5 dB | 30.8 dB |

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Experiment |
|--------|-----------|-----------------|
| **Photon Agent** | | |
| N_effective | 4.11e11 | 3.70e8 |
| SNR | 116.1 dB | 85.7 dB |
| Quality tier | excellent | excellent |
| **Mismatch Agent** | | |
| Severity | 0.070 (very low) | 0.238 (moderate) |
| Dominant error | depth_range | **depth_range (30mm)** |
| Correction needed | No | **Yes** |
| **Recoverability Agent** | | |
| Score | 0.78 (good) | 0.68 (sufficient) |
| Expected PSNR | 31.5 dB | 28.3 dB |
| **Analysis Agent** | | |
| Primary bottleneck | compression | **compression** |
| P(success) | 0.773 | 0.666 |
| **Negotiator** | | |
| P_joint | 0.704 | 0.538 |
| **Pipeline** | | |
| Without correction | 27.0 dB | 23.5 dB |
| With correction | -- | **26.8 dB** |
| With correction + DIBR | -- | **28.5 dB** |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (Depth Est. -> DIBR -> Learning-Based) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Depth-aware:** The pipeline correctly handles the depth-volume inversion where per-plane deconvolution with depth-dependent PSFs enables 3D reconstruction from a single 2D integral image.
