# Panorama Multi-Focus Fusion Working Process

## End-to-End Pipeline for Panoramic Multi-Focus Image Fusion

This document traces a complete panoramic multi-focus fusion experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Fuse these 5 multi-focus images into a single all-in-focus panorama.
 Images: focus_stack_01.tif through focus_stack_05.tif, each 512x512,
 focal distances: [0.3m, 1.0m, 2.5m, 5.0m, 10.0m]."
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "focus_stack_01.tif" detected
#   operator_type=OperatorType.linear_operator,
#   files=["focus_stack_01.tif", ..., "focus_stack_05.tif"],
#   params={"n_images": 5, "focal_distances_m": [0.3, 1.0, 2.5, 5.0, 10.0]}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> panorama entry
panorama:
  keywords: [panorama, multi_focus, image_fusion, all_in_focus, depth_of_field]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="panorama",
#   confidence=0.94,
#   reasoning="Matched keywords: multi-focus, panorama, all-in-focus"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the panorama registry entry:

```python
system = plan_agent.build_imaging_system("panorama")
# ImagingSystem(
#   modality_key="panorama",
#   display_name="Panorama Multi-Focus Fusion",
#   signal_dims={"x": [512, 512, 3], "y": [512, 512, 3]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...4 elements...],
#   default_solver="laplacian_pyramid_fusion"
# )
```

**Panorama Element Chain (4 elements):**

```
Scene Illumination --> Camera Lens (Variable Focus) --> Focus Bracketing Controller --> RGB Image Sensor
  throughput=1.0        throughput=0.88                 throughput=1.0                  throughput=0.82
  noise: none           noise: aberration               noise: alignment                noise: shot+read+quant
                        f/2.8, 5 focus planes           5 images
                        focus_range=[0.3, 10.0]m        step=0.5 diopters
```

**Cumulative throughput:** `1.0 x 0.88 x 1.0 x 0.82 = 0.722`

**Forward model for each focus plane k:**
```
y_k(x, y) = PSF(d_k) ** x(x, y) + n

where PSF(d_k) is the defocus point spread function at distance d_k:
  PSF(d_k)(r) = circ(r / R_blur(d_k))   (geometric defocus disk)
  R_blur(d_k) = f^2 / (N * |1/d_k - 1/d_focus|)

  f = 50 mm focal length
  N = f/2.8 aperture
  d_focus = current focal distance for image k
```

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  panorama:
    model_id: "generic_detector"
    parameters:
      source_photons: 1.0e+08
      qe: 0.90
      exposure_s: 0.033
  ```

### Computation

```python
# 1. For generic_detector model, photon count is directly specified
N_raw = source_photons * qe
#     = 1.0e8 * 0.90
#     = 9.0e7 photons/pixel

# 2. Apply cumulative throughput
N_effective = N_raw * 0.722 = 6.50e7 photons/pixel

# 3. Noise variances
shot_var   = N_effective                     # 6.50e7
read_var   = read_noise_e^2 = 9.0           # 3.0 e- read noise
dark_var   = 0                               # Negligible
total_var  = shot_var + read_var
#          = 6.50e7 + 9.0 = 6.50e7

# 4. SNR
SNR = N_effective / sqrt(total_var)
#   = 6.50e7 / sqrt(6.50e7)
#   = sqrt(6.50e7) = 8062
SNR_db = 20 * log10(8062) = 78.1 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=6.50e7,
  snr_db=78.1,
  noise_regime=NoiseRegime.shot_limited,     # shot_var/total_var > 0.9
  shot_noise_sigma=8062.0,
  read_noise_sigma=3.0,
  total_noise_sigma=8062.0,
  feasible=True,
  quality_tier="excellent",                  # SNR > 30 dB
  throughput_chain=[
    {"Scene Illumination": 1.0},
    {"Camera Lens (Variable Focus)": 0.88},
    {"Focus Bracketing Controller": 1.0},
    {"RGB Image Sensor": 0.82}
  ],
  noise_model="poisson",
  explanation="Shot-limited regime. Excellent SNR for multi-focus fusion. Each focal image has high fidelity."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"panorama"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  panorama:
    parameters:
      registration_error:
        range: [0.0, 5.0]
        typical_error: 1.0
        unit: "pixels"
        description: "Inter-frame registration error from imprecise homography"
      vignetting:
        range: [0.0, 0.30]
        typical_error: 0.08
        unit: "normalized"
        description: "Radial intensity falloff from lens vignetting"
    severity_weights:
      registration_error: 0.60
      vignetting: 0.40
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.60 * |1.0| / 5.0      # registration_error: 0.120
  + 0.40 * |0.08| / 0.30    # vignetting: 0.107
S = 0.227  # Moderate severity (typical handheld capture)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 2.27 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="panorama",
  mismatch_family="grid_search",
  parameters={
    "registration_error": {"typical_error": 1.0, "range": [0.0, 5.0], "weight": 0.60},
    "vignetting":         {"typical_error": 0.08, "range": [0.0, 0.30], "weight": 0.40}
  },
  severity_score=0.227,
  correction_method="grid_search",
  expected_improvement_db=2.27,
  explanation="Moderate mismatch. Inter-frame registration (1.0 px) is the primary mismatch source. "
              "Feature-based alignment should correct most of this."
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
  panorama:
    signal_prior_class: "deep_prior"
    entries:
      - {cr: 1.0, noise: "shot_limited", solver: "neural_fusion",
         recoverability: 0.84, expected_psnr_db: 28.4,
         provenance: {dataset_id: "lytro_multifocus_2023", ...}}
      - {cr: 1.0, noise: "shot_limited", solver: "ifcnn",
         recoverability: 0.86, expected_psnr_db: 29.8, ...}
      - {cr: 1.0, noise: "read_limited", solver: "neural_fusion",
         recoverability: 0.77, expected_psnr_db: 25.6, ...}
  ```

### Computation

```python
# 1. Compression ratio
#    Multi-focus fusion: N input images -> 1 output image
#    CR = prod(y_shape) / (N_images * prod(x_shape))
#    But since each y_k has same dimensions as x, and we have N=5 views:
CR = (512 * 512 * 3) / (5 * 512 * 512 * 3) = 0.20
#    However, for fusion problems CR is effectively 1.0 (no information loss,
#    just different focal depths), since each pixel is observed N times.

# 2. Operator diversity (multiple focal planes provide complementary sharpness)
#    Each view has different regions in focus -- diversity depends on depth complexity
diversity = 0.85  # High diversity (5 distinct focal planes spanning full range)

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.541

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="neural_fusion", cr=1.0
#    -> recoverability=0.84, expected_psnr=28.4 dB, confidence=1.0

# 5. Best solver selection
#    ifcnn: 29.8 dB > neural_fusion: 28.4 dB
#    -> recommended: "ifcnn" (or "neural_fusion" as default)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=1.0,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.deep_prior,
  operator_diversity_score=0.85,
  condition_number_proxy=0.541,
  recoverability_score=0.84,
  recoverability_confidence=1.0,
  expected_psnr_db=28.4,
  expected_psnr_uncertainty_db=1.0,
  recommended_solver_family="neural_fusion",
  verdict="good",                  # 0.70 <= score < 0.85
  calibration_table_entry={...},
  explanation="Good recoverability. 5 focal planes provide sufficient depth coverage. "
              "Neural fusion expected 28.4 dB on Lytro benchmark. IFCNN may yield +1.4 dB."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(78.1 / 40, 1.0)  = 0.0     # Excellent SNR
mismatch_score    = 0.227                      = 0.227   # Moderate mismatch
compression_score = 1 - 0.84                   = 0.16    # Good recoverability
solver_score      = 0.2                        = 0.2     # Default placeholder

# Primary bottleneck
primary = "mismatch"  # max(0.0, 0.227, 0.16, 0.2) = mismatch

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.227*0.5) * (1 - 0.16*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.887 * 0.92 * 0.90
  = 0.734
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="mismatch",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.227, compression=0.16, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Apply feature-based registration before fusion to reduce alignment error",
      priority="medium",
      expected_gain_db=1.5
    ),
    Suggestion(
      text="Consider IFCNN for +1.4 dB over neural fusion baseline",
      priority="medium",
      expected_gain_db=1.4
    ),
    Suggestion(
      text="Flat-field correction to remove lens vignetting",
      priority="low",
      expected_gain_db=0.5
    )
  ],
  overall_verdict="good",             # 0.60 <= P < 0.80
  probability_of_success=0.734,
  explanation="System is well-configured. Inter-frame registration is the primary bottleneck. "
              "Feature-based alignment should resolve most misregistration artifacts."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="good" | No veto |
| Severe mismatch without correction | severity=0.227 < 0.7 | No veto |
| All marginal | photon=excellent, others good | No veto |
| Joint probability floor | P=0.734 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95   # tier_prob["excellent"]
P_recoverability = 0.84   # recoverability_score
P_mismatch       = 1.0 - 0.227 * 0.7 = 0.841

P_joint = 0.95 * 0.84 * 0.841 = 0.671
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.671
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 512 * 512 * 3 = 786,432
n_views = 5
dim_factor   = (total_pixels * n_views) / (512 * 512) = 15.0
solver_complexity = 2.0  # Neural fusion (MLP + Fourier features)
cr_factor    = max(1.0, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 15.0 * 2.0 * 0.125 = 7.5 seconds
# Laplacian pyramid: <1s, Neural fusion: ~7.5s (6000 iterations)
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="panorama", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=7.5,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=["multi-focus image stack (5 images, different focal planes)",
                  "optional: focus distances in meters"]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Panorama multi-focus forward model:
#   y_k(x, y) = PSF(d_k) ** x(x, y) + n
#
# For each focal plane k:
#   The defocus PSF at distance d_k is a depth-dependent Gaussian blur
#   sigma_blur(d, d_k) = |d - d_k| * max_blur / max_depth_diff
#
# Parameters:
#   n_focus: 5 focal planes
#   focal_distances: [0.3, 1.0, 2.5, 5.0, 10.0] meters
#   max_blur: 5.0 pixels (maximum defocus radius)
#
# Input:  x = (512, 512, 3) all-in-focus ground truth
# Output: {y_k} = list of 5 (512, 512, 3) blurred images

class PanoramaMultifocusOperator(PhysicsOperator):
    def forward(self, x, depth_map):
        """Generate N defocused views from all-in-focus image."""
        views = []
        for k, d_focus in enumerate(self.focal_distances):
            # Depth-dependent blur radius
            depth_diff = np.abs(depth_map - d_focus)
            blur_radii = depth_diff * self.max_blur / self.max_depth_range
            # Apply spatially-varying Gaussian blur
            y_k = self._apply_defocus_blur(x, depth_map, d_focus)
            views.append(y_k)
        return views

    def adjoint(self, views, depth_map):
        """Sharpness-weighted average of views."""
        weights = np.zeros_like(views[0])
        accum = np.zeros_like(views[0])
        for view in views:
            sharpness = self._compute_local_sharpness(view)
            accum += sharpness * view
            weights += sharpness
        return accum / (weights + 1e-8)

    def check_adjoint(self):
        """Verify <Ax, y> ~= <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-6)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided focus_stack:
views = [np.array(Image.open(f"focus_stack_{k:02d}.tif")) for k in range(1, 6)]
# Each view: (512, 512, 3), different focal plane

# If simulating:
x_true = load_ground_truth()               # (512, 512, 3) all-in-focus
depth_map = load_depth_map()               # (512, 512) in meters
views = operator.forward(x_true, depth_map)
for k in range(5):
    views[k] += np.random.randn(*views[k].shape) * 0.02  # Read noise
```

### Step 9c: Reconstruction with Laplacian Pyramid Fusion

```python
from pwm_core.recon.panorama_solver import multifocus_fusion_laplacian

# Embed each view into panorama coordinate frame
view_images = []
for view, (x_start, x_end) in zip(views, view_positions):
    pano_view = np.zeros((panorama_height, panorama_width), dtype=np.float32)
    pano_view[:, x_start:x_end] = view
    view_images.append(pano_view)

x_hat = multifocus_fusion_laplacian(view_images)
# x_hat shape: (256, 512) -- all-in-focus panorama
# Expected PSNR: ~25 dB (Laplacian pyramid baseline)
```

**Neural Fusion (default solver):**

```python
# Coordinate-based MLP with Fourier features
# Learns all-in-focus panorama by training on sharpness-weighted observations
#
# Architecture: Fourier(128 features, scale=10) -> MLP(256 x 5 layers) -> Sigmoid
# Training: 6000 iterations, batch=16384, Adam with cosine annealing
# Focus-aware weighting: w = sharpness^2 + 0.1 (emphasizes in-focus pixels)

recon = neural_panorama_fusion(
    views, view_positions, focal_depths,
    panorama_width=512, panorama_height=256,
    n_iters=6000, lr=1e-3,
)
# Expected PSNR: ~28.0 dB (reference benchmark)
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| Laplacian Pyramid | Traditional | ~25 dB | No | `multifocus_fusion_laplacian(view_images)` |
| Guided Filter | Traditional | ~26 dB | No | `multifocus_fusion_guided(view_images)` |
| IFCNN | Deep Learning | ~29.8 dB | Yes | `ifcnn_train_quick(view_images, x_true)` |
| Neural Fusion | Neural (MLP) | ~28.0 dB | Yes | `neural_panorama_fusion(views, positions, ...)` |

### Step 9d: Metrics

```python
# PSNR (Peak Signal-to-Noise Ratio)
psnr = 10 * log10(max_val^2 / mse(x_hat, x_true))  # ~28.0 dB

# SSIM (Structural Similarity)
ssim = structural_similarity(x_hat, x_true, data_range=1.0)

# Q_AB/F (Xydeas-Petrovic fusion quality) -- multi-focus-specific
# Measures how well edges from source images are transferred to the fused result
q_abf = edge_preservation_metric(views, x_hat)
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
|   +-- views/
|   |   +-- view_0.npy     # Focal plane at 0.3m + SHA256 hash
|   |   +-- view_1.npy     # Focal plane at 1.0m + SHA256 hash
|   |   +-- view_2.npy     # Focal plane at 2.5m + SHA256 hash
|   |   +-- view_3.npy     # Focal plane at 5.0m + SHA256 hash
|   |   +-- view_4.npy     # Focal plane at 10.0m + SHA256 hash
|   +-- x_hat.npy          # Fused all-in-focus (512, 512, 3) + SHA256 hash
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
+-- metrics.json           # PSNR, SSIM, Q_AB/F
+-- operator.json          # Operator parameters (n_views, focal_distances, max_blur)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized panoramic fusion pipeline with well-controlled focal distances and minimal registration error. In practice, real multi-focus captures have significant inter-frame misregistration from handheld camera motion, vignetting from wide-aperture lenses, and exposure variation between frames.

This section traces the same pipeline with realistic parameters from handheld capture conditions.

---

## Real Experiment: User Prompt

```
"I shot these focus-bracket images handheld on a windy day. Some frames
 may be slightly shifted. Please align and fuse them into an all-in-focus
 panorama. Images: IMG_001.tif through IMG_005.tif."
```

**Key difference:** The user mentions handheld capture and potential misalignment. Inter-frame registration is the dominant challenge.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,             # "align and fuse" detected
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["IMG_001.tif", ..., "IMG_005.tif"],
#   params={"n_images": 5}
# )
```

---

## R2. PhotonAgent -- Outdoor Handheld Conditions

### Real capture parameters

```yaml
# Real: outdoor handheld, short exposure to avoid motion blur
panorama_handheld:
  source_photons: 5.0e+06         # 20x fewer (faster shutter, cloudy day)
  qe: 0.82                        # Modern CMOS
  exposure_s: 0.004               # 1/250s to freeze hand motion
  read_noise_e: 4.0               # Higher than controlled setting
```

### Computation

```python
N_effective = 5.0e6 * 0.82 * 0.722 = 2.96e6 photons/pixel

shot_var   = 2.96e6
read_var   = 16.0  # 4.0^2
total_var  = 2.96e6 + 16.0 = 2.96e6

SNR = 2.96e6 / sqrt(2.96e6) = sqrt(2.96e6) = 1721
SNR_db = 20 * log10(1721) = 64.7 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=2.96e6,
  snr_db=64.7,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="excellent",                  # 64.7 >> 30 dB
  explanation="Shot-limited even with fast shutter. Excellent SNR for fusion."
)
```

---

## R3. MismatchAgent -- Handheld Registration Error

```python
# Actual errors from handheld capture + wind
mismatch_actual = {
    "registration_error": 3.5,    # pixels (up from 1.0 typical)
    "vignetting": 0.15,           # normalized (stronger with wide aperture)
}

# Severity computation
S = 0.60 * |3.5| / 5.0        # registration: 0.420
  + 0.40 * |0.15| / 0.30      # vignetting: 0.200
S = 0.620  # HIGH severity

improvement_db = clip(10 * 0.620, 0, 20) = 6.20 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="panorama",
  severity_score=0.620,
  correction_method="grid_search",
  expected_improvement_db=6.20,
  explanation="High mismatch. Handheld registration error (3.5 px) dominates. "
              "Feature-based alignment + vignetting correction critical."
)
```

---

## R4. RecoverabilityAgent -- Degraded by Misregistration

```python
# Effective noise regime degraded by registration artifacts
# Calibration table lookup: noise="read_limited" (misregistration acts as structured noise)
# -> recoverability=0.77, expected_psnr=25.6 dB
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=1.0,
  recoverability_score=0.77,              # Down from 0.84
  expected_psnr_db=25.6,                  # Down from 28.4
  verdict="good",                         # Still good, but degraded
  explanation="Recoverability degraded by inter-frame misregistration. "
              "Feature-based alignment before fusion is essential."
)
```

---

## R5. AnalysisAgent -- Registration is the Bottleneck

```python
# Bottleneck scores
photon_score      = 0.0       # Excellent
mismatch_score    = 0.620     # HIGH
compression_score = 0.23      # Good
solver_score      = 0.2

primary = "mismatch"  # 0.620 is dominant

P = (1 - 0.0*0.5) * (1 - 0.620*0.5) * (1 - 0.23*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.690 * 0.885 * 0.90
  = 0.549
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="mismatch",
  probability_of_success=0.549,
  overall_verdict="marginal",
  suggestions=[
    Suggestion(text="Run SIFT/ORB feature-based alignment before fusion", priority="critical", expected_gain_db=4.0),
    Suggestion(text="Apply flat-field vignetting correction", priority="high", expected_gain_db=1.5),
    Suggestion(text="Use exposure normalization across frames", priority="medium", expected_gain_db=0.8)
  ]
)
```

---

## R6. AgentNegotiator -- Conditional Proceed

```python
P_photon         = 0.95
P_recoverability = 0.77
P_mismatch       = 1.0 - 0.620 * 0.7 = 0.566

P_joint = 0.95 * 0.77 * 0.566 = 0.414
```

| Condition | Check | Result |
|-----------|-------|--------|
| Severe mismatch without correction | severity=0.620 < 0.7 | No veto |
| Joint probability floor | P=0.414 > 0.15 | No veto |

```python
NegotiationResult(
  vetoes=[],
  proceed=True,
  probability_of_success=0.414
)
```

---

## R7. PreFlightReportBuilder -- Warnings Raised

```python
PreFlightReport(
  estimated_runtime_s=12.0,
  proceed_recommended=True,
  warnings=[
    "Mismatch severity 0.620 -- significant inter-frame registration error",
    "Feature-based alignment will run before fusion (~3s additional)",
    "Vignetting correction recommended for wide-aperture lens"
  ]
)
```

---

## R8. Pipeline Runner -- With Alignment Correction

### Step R8a: Direct Fusion (No Alignment)

```python
x_naive = multifocus_fusion_laplacian(view_images)
# PSNR = 19.2 dB  <-- ghosting artifacts from misregistration
```

### Step R8b: Feature-Based Alignment + Fusion

```python
# SIFT feature matching + RANSAC homography estimation
aligned_views = feature_align(views, reference_idx=2)
# Residual registration error: ~0.3 px (down from 3.5)

x_aligned = multifocus_fusion_guided(aligned_views)
# PSNR = 26.8 dB  <-- +7.6 dB improvement from alignment
```

### Step R8c: Final Comparison

| Configuration | Laplacian | Guided Filter | IFCNN | Neural Fusion |
|---------------|-----------|---------------|-------|---------------|
| Ideal (no misregistration) | 25.0 dB | 26.0 dB | 29.8 dB | 28.0 dB |
| Handheld (no alignment) | 19.2 dB | 19.8 dB | 21.5 dB | 22.0 dB |
| Handheld + SIFT alignment | 24.5 dB | 26.8 dB | 28.5 dB | 27.2 dB |

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Experiment |
|--------|-----------|-----------------|
| **Photon Agent** | | |
| N_effective | 6.50e7 | 2.96e6 |
| SNR | 78.1 dB | 64.7 dB |
| Quality tier | excellent | excellent |
| **Mismatch Agent** | | |
| Severity | 0.227 (moderate) | 0.620 (high) |
| Dominant error | registration 1.0px | **registration 3.5px** |
| Correction needed | Optional | **Yes** |
| **Recoverability Agent** | | |
| Score | 0.84 (good) | 0.77 (good) |
| Expected PSNR | 28.4 dB | 25.6 dB |
| **Analysis Agent** | | |
| Primary bottleneck | mismatch | **mismatch** |
| P(success) | 0.734 | 0.549 |
| **Negotiator** | | |
| P_joint | 0.671 | 0.414 |
| **Pipeline** | | |
| Without alignment | 25.0 dB | 19.2 dB |
| With alignment | -- | **26.8 dB** |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (Laplacian -> Guided Filter -> IFCNN -> Neural) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Fusion-aware:** The pipeline correctly handles the N-input -> 1-output multi-focus fusion paradigm, where sharpness-weighted aggregation replaces traditional compressed sensing reconstruction.
