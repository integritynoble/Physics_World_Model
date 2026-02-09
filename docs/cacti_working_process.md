# CACTI Working Process

## End-to-End Pipeline for Coded Aperture Compressive Temporal Imaging

This document traces a complete CACTI experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct a high-speed video sequence from this CACTI snapshot.
 Measurement: cacti_snapshot.npy, Masks: cacti_masks.npy,
 8 frames compressed into 1, 256x256 spatial resolution."
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "cacti_snapshot.npy" detected
#   operator_type=OperatorType.linear_operator,
#   files=["cacti_snapshot.npy", "cacti_masks.npy"],
#   params={"n_frames": 8, "spatial_resolution": [256, 256]}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> cacti entry
cacti:
  keywords: [CACTI, video_compressive, snapshot, temporal_coding, SCI]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="cacti",
#   confidence=0.93,
#   reasoning="Matched keywords: CACTI, snapshot, video_compressive"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the CACTI registry entry:

```python
system = plan_agent.build_imaging_system("cacti")
# ImagingSystem(
#   modality_key="cacti",
#   display_name="Coded Aperture Compressive Temporal Imaging (CACTI)",
#   signal_dims={"x": [256, 256, 8], "y": [256, 256]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...5 elements...],
#   default_solver="gap_tv"
# )
```

**CACTI Element Chain (5 elements):**

```
Scene Illumination ──> Objective Lens ──> Coded Aperture (Shifting Mask) ──> Relay Optics ──> CMOS Detector
  throughput=1.0       throughput=0.88   throughput=0.50                     throughput=0.90   throughput=0.78
  noise: none          noise: aberration noise: fixed_pattern                noise: none       noise: shot+read+quant
                                               + alignment
```

**Cumulative throughput:** `0.88 x 0.50 x 0.90 x 0.78 = 0.309`

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  cacti:
    model_id: "generic_detector"
    parameters:
      source_photons: 1.0e+05
      qe: 0.70
      exposure_s: 0.033
  ```

### Computation

```python
# 1. Source photon count (generic detector model)
N_source = source_photons = 1.0e5

# 2. Quantum efficiency
N_detected = N_source * qe = 1.0e5 * 0.70 = 7.0e4

# 3. Apply cumulative throughput
N_effective = N_detected * 0.309 = 2.16e4 photons/pixel

# 4. Noise variances
shot_var   = N_effective = 2.16e4              # Poisson
read_var   = read_noise^2 = 3.0^2 = 9.0       # CMOS read noise
dark_var   = 0                                 # Negligible at 33 ms
total_var  = 2.16e4 + 9.0 = 2.16e4

# 5. SNR
SNR = N_effective / sqrt(total_var) = 2.16e4 / sqrt(2.16e4) = 147.0
SNR_db = 20 * log10(147.0) = 43.3 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=2.16e4,
  snr_db=43.3,
  noise_regime=NoiseRegime.shot_limited,      # shot_var/total_var > 0.99
  shot_noise_sigma=147.0,
  read_noise_sigma=3.0,
  total_noise_sigma=147.0,
  feasible=True,
  quality_tier="excellent",                   # SNR > 30 dB
  throughput_chain=[
    {"Scene Illumination": 1.0},
    {"Objective Lens": 0.88},
    {"Coded Aperture (Shifting Mask)": 0.50},
    {"Relay Optics": 0.90},
    {"CMOS Detector": 0.78}
  ],
  noise_model="poisson",
  explanation="Shot-noise-limited regime. CMOS sensor provides good SNR at 33 ms exposure."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"cacti"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  cacti:
    parameters:
      mask_shift:
        range: [-4, 4]
        typical_error: 1.0
        weight: 0.45
      temporal_jitter:
        range: [-2, 2]
        typical_error: 0.5
        weight: 0.35
      psf_sigma:
        range: [0.3, 2.0]
        typical_error: 0.3
        weight: 0.20
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.45 * |1.0| / 8.0       # mask_shift: 0.056
  + 0.35 * |0.5| / 4.0       # temporal_jitter: 0.044
  + 0.20 * |0.3| / 1.7       # psf_sigma: 0.035
S = 0.135  # Low severity

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.35 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="cacti",
  mismatch_family="grid_search",
  parameters={
    "mask_shift":       {"typical_error": 1.0, "range": [-4, 4], "weight": 0.45},
    "temporal_jitter":  {"typical_error": 0.5, "range": [-2, 2], "weight": 0.35},
    "psf_sigma":        {"typical_error": 0.3, "range": [0.3, 2.0], "weight": 0.20}
  },
  severity_score=0.135,
  correction_method="grid_search",
  expected_improvement_db=1.35,
  explanation="Low mismatch severity under typical lab conditions."
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
  cacti:
    signal_prior_class: "temporal_smooth"
    entries:
      - {cr: 0.125, noise: "shot_limited", solver: "gap_tv",
         recoverability: 0.80, expected_psnr_db: 32.83,
         provenance: {dataset_id: "cacti_simulated_8frame_2023", ...}}
      - {cr: 0.125, noise: "shot_limited", solver: "efficientsci",
         recoverability: 0.92, expected_psnr_db: 36.78, ...}
      - {cr: 0.125, noise: "detector_limited", solver: "gap_tv",
         recoverability: 0.68, expected_psnr_db: 28.45, ...}
  ```

### Computation

```python
# 1. Compression ratio
CR = prod(y_shape) / prod(x_shape) = (256 * 256) / (256 * 256 * 8) = 0.125

# 2. Operator diversity (shifting binary mask)
density = 0.5  # binary random mask
n_frames = 8   # different shifted versions
diversity = 4 * density * (1 - density) * sqrt(n_frames) / n_frames
         = 1.0 * 2.83 / 8 = 0.354
# With circular shift patterns, each frame sees a different mask
# Effective diversity is higher than random: ~0.85

# 3. Condition number proxy
kappa = 1 / (1 + 0.85) = 0.541

# 4. Calibration table lookup
#    Exact match: noise="shot_limited", solver="gap_tv", cr=0.125
#    -> recoverability=0.80, expected_psnr=32.83 dB, confidence=1.0

# 5. Best solver selection
#    efficientsci: 36.78 dB > gap_tv: 32.83 dB
#    -> recommended: "efficientsci" (or "gap_tv" as default)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.125,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.temporal_smooth,
  operator_diversity_score=0.85,
  condition_number_proxy=0.541,
  recoverability_score=0.80,
  recoverability_confidence=1.0,
  expected_psnr_db=32.83,
  expected_psnr_uncertainty_db=1.0,
  recommended_solver_family="gap_tv",
  verdict="sufficient",              # 0.60 <= score < 0.85
  calibration_table_entry={...},
  explanation="Sufficient recoverability. GAP-TV expected 32.83 dB; EfficientSCI can reach 36.78 dB."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(43.3 / 40, 1.0)   = 0.0    # Excellent SNR
mismatch_score    = 0.135                        = 0.135  # Low mismatch
compression_score = 1 - 0.80                     = 0.20   # Good recoverability
solver_score      = 0.2                          = 0.2    # Default placeholder

# Primary bottleneck (tie between compression and solver)
primary = "compression"  # max(0.0, 0.135, 0.20, 0.2) = compression/solver tie

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.135*0.5) * (1 - 0.20*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.933 * 0.90 * 0.90
  = 0.755
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="compression",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.135, compression=0.20, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Use EfficientSCI for +3.95 dB over GAP-TV",
      priority="high",
      expected_gain_db=3.95
    ),
    Suggestion(
      text="System is well-balanced. Solver choice is the main lever.",
      priority="medium",
      expected_gain_db=0.0
    )
  ],
  overall_verdict="sufficient",       # 0.60 <= P < 0.80
  probability_of_success=0.755,
  explanation="Well-balanced system. Solver upgrade to EfficientSCI offers largest single gain."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="sufficient" | No veto |
| Severe mismatch without correction | severity=0.135 < 0.7 | No veto |
| All marginal | photon=excellent, others=sufficient | No veto |
| Joint probability floor | P=0.755 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.80    # recoverability_score
P_mismatch       = 1.0 - 0.135 * 0.7 = 0.906

P_joint = 0.95 * 0.80 * 0.906 = 0.689
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.689
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 256 * 256 * 8 = 524,288
dim_factor   = total_pixels / (256 * 256) = 8.0
solver_complexity = 2.0  # GAP-TV (iterative, CPU)
cr_factor    = max(0.125, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 8.0 * 2.0 * 0.125 = 4.0 seconds
# GAP-TV with 100 iterations on 256x256x8 volume
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="cacti", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=4.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=["measurement (2D CACTI snapshot)", "masks (3D mask array, H x W x T)"]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# CACTI forward model: y(x,y) = sum_t M_t(x,y) * X(x,y,t)
#
# Parameters:
#   Phi:     (H, W, T) time-varying mask  [loaded from cacti_masks.npy]
#   n_frames: 8                           [temporal compression factor]
#
# Input:  x = (256, 256, 8) video cube
# Output: y = (256, 256) compressed snapshot
#         where 8 frames integrate onto a single 2D measurement

class CACTIOperator(PhysicsOperator):
    def forward(self, x):
        """y(r,c) = sum_t Phi(r,c,t) * x(r,c,t)"""
        return np.sum(self.Phi * x, axis=2)

    def adjoint(self, y):
        """x_hat(r,c,t) = y(r,c) * Phi(r,c,t) / Phi_sum(r,c)"""
        Phi_sum = np.sum(self.Phi, axis=2)
        Phi_sum[Phi_sum == 0] = 1
        return y[:, :, np.newaxis] * self.Phi / Phi_sum[:, :, np.newaxis]

    def check_adjoint(self):
        """Verify <Ax, y> ~ <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided cacti_snapshot.npy:
y = np.load("cacti_snapshot.npy")      # (256, 256)

# If simulating:
x_true = load_ground_truth_video()      # (256, 256, 8) from CACTI benchmark
mask_base = (np.random.rand(256, 256) > 0.5).astype(np.float32)

# Create shifting masks (circular shift pattern)
Phi = np.zeros((256, 256, 8), dtype=np.float32)
for f in range(8):
    Phi[:, :, f] = np.roll(mask_base, shift=f, axis=1)

y = np.sum(Phi * x_true, axis=2)       # (256, 256) compressed snapshot
y += np.random.randn(256, 256) * 0.01  # Gaussian measurement noise
```

### Step 9c: Reconstruction with GAP-TV

```python
from pwm_core.recon.cacti_gap import gap_denoise_cacti

x_hat = gap_denoise_cacti(
    y=y,                     # (256, 256) compressed measurement
    Phi=Phi,                 # (256, 256, 8) time-varying masks
    max_iter=100,            # GAP outer iterations
    lam=1.0,                 # Relaxation parameter
    accelerate=True,         # Accelerated GAP
    tv_weight=0.15,          # TV denoiser strength
    tv_iter=5                # TV denoiser iterations per step
)
# x_hat shape: (256, 256, 8) -- reconstructed video sequence
# Expected PSNR: ~32.83 dB (CACTI benchmark)
```

**GAP-TV Algorithm:**
```python
# Generalized Alternating Projection with TV denoising
# Forward: A(x, Phi) = sum(x * Phi, axis=2)
# Adjoint: At(y, Phi) = y[:,:,None] * Phi

Phi_sum = sum(Phi, axis=2)             # (H, W) normalization
x = y[:,:,None] * Phi / Phi_sum[:,:,None]  # Backprojection init
y1 = y.copy()

for k in range(max_iter):
    yb = sum(x * Phi, axis=2)          # Forward projection
    y1 = y1 + (y - yb)                 # Accelerated residual
    residual = y1 - yb
    x = x + lam * (residual / Phi_sum)[:,:,None] * Phi   # SART-like update
    for f in range(n_frames):
        x[:,:,f] = denoise_tv_chambolle(x[:,:,f], weight=tv_weight)
    x = clip(x, 0, 1)
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| GAP-TV | Traditional | 32.83 dB | No | `gap_denoise_cacti(y, Phi, max_iter=100)` |
| GAP-denoise | Traditional | 33.50 dB | No | `gap_denoise_cacti(y, Phi, tv_weight=0.15)` |
| EfficientSCI | Deep Learning | 36.78 dB | Yes | `efficientsci_recon(y, Phi, variant="tiny")` |

### Step 9d: Metrics

```python
# Per-frame PSNR
for t in range(8):
    psnr_t = 10 * log10(max_val^2 / mse(x_hat[:,:,t], x_true[:,:,t]))

# Average PSNR across all frames
avg_psnr = mean(psnr_per_frame)  # ~32.83 dB

# SSIM (structural similarity per frame)
avg_ssim = mean([ssim(x_hat[:,:,t], x_true[:,:,t]) for t in range(8)])

# Temporal consistency (CACTI-specific)
# Measures frame-to-frame smoothness of reconstruction
temp_consistency = mean([
    1 - mse(x_hat[:,:,t+1] - x_hat[:,:,t],
            x_true[:,:,t+1] - x_true[:,:,t]) / mse_norm
    for t in range(7)
])
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
|   +-- y.npy              # Measurement (256, 256) + SHA256 hash
|   +-- x_hat.npy          # Reconstruction (256, 256, 8) + SHA256 hash
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
|   +-- masks.npy          # Mask array (256, 256, 8) + SHA256 hash
+-- metrics.json           # PSNR, SSIM per frame + average, temporal consistency
+-- operator.json          # Operator parameters (mask hash, n_frames, shift_type)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized CACTI pipeline with excellent SNR (43.3 dB) and low mismatch (0.135). In practice, real CACTI systems have mask-detector synchronization issues, spatial mask shifts from mechanical vibration, and detector noise from fast readout.

This section traces the same pipeline with realistic degraded parameters.

---

## Real Experiment: User Prompt

```
"We have CACTI data from our video compressive camera. The timing
 synchronization may be slightly off -- the mask controller was
 running at 240 Hz but the camera exposure was 33 ms. Please
 reconstruct the 8-frame video.
 Measurement: lab_cacti.npy, Masks: lab_masks.npy."
```

**Key difference:** Potential temporal jitter between mask and detector.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["lab_cacti.npy", "lab_masks.npy"],
#   params={"n_frames": 8}
# )
```

---

## R2. PhotonAgent -- Realistic Lab Conditions

### Real detector parameters

```yaml
# Real lab: faster readout, reduced photon budget
cacti_lab:
  source_photons: 3.0e+04        # Dimmer scene (indoor, fast motion)
  qe: 0.60                       # Detector degradation
  exposure_s: 0.033
  read_noise_e: 6.0              # Higher at fast readout rate
```

### Computation

```python
N_detected = 3.0e4 * 0.60 = 1.8e4
N_effective = 1.8e4 * 0.309 = 5.56e3 photons/pixel

shot_var   = 5.56e3
read_var   = 6.0^2 = 36.0
total_var  = 5.56e3 + 36.0 = 5.60e3

SNR = 5.56e3 / sqrt(5.60e3) = 74.3
SNR_db = 20 * log10(74.3) = 37.4 dB
```

### Output

```python
PhotonReport(
  n_photons_per_pixel=5.56e3,
  snr_db=37.4,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="good",              # 30 dB < SNR < 40 dB
  explanation="Shot-limited but with reduced photon budget. Feasible but SNR is only moderate."
)
```

---

## R3. MismatchAgent -- Real Synchronization Errors

```python
# Actual errors from lab prototype
S = 0.45 * |2.0| / 8.0        # mask_shift: 0.113   (2 px shift from vibration!)
  + 0.35 * |1.0| / 4.0        # temporal_jitter: 0.088  (1 frame timing offset)
  + 0.20 * |0.5| / 1.7        # psf_sigma: 0.059  (relay lens defocus)
S = 0.260  # MODERATE severity

improvement_db = clip(10 * 0.260, 0, 20) = 2.60 dB
```

### Output

```python
MismatchReport(
  severity_score=0.260,
  correction_method="grid_search",
  expected_improvement_db=2.60,
  explanation="Moderate mismatch. Mask shift (2 px) from vibration is the dominant error."
)
```

---

## R4. RecoverabilityAgent -- Degraded

```python
# CR = 0.125 (unchanged)
# noise="shot_limited", solver="gap_tv"
# Calibration match: recoverability=0.80, expected_psnr=32.83 dB

# With mismatch degradation:
# recoverability_adjusted = 0.80 * (1 - 0.260*0.5) = 0.696
# expected_psnr_adjusted ~ 28.5 dB
```

### Output

```python
RecoverabilityReport(
  compression_ratio=0.125,
  recoverability_score=0.696,
  expected_psnr_db=28.5,
  verdict="sufficient",           # score >= 0.60
  explanation="Sufficient recoverability despite mismatch. Consider EfficientSCI for robustness."
)
```

---

## R5. AnalysisAgent -- Mismatch is Growing

```python
photon_score      = 1 - min(37.4 / 40, 1.0) = 0.065    # Good, not excellent
mismatch_score    = 0.260                     = 0.260
compression_score = 1 - 0.696                 = 0.304
solver_score      = 0.2

primary = "compression"  # max(0.065, 0.260, 0.304, 0.2)

P = (1 - 0.065*0.5) * (1 - 0.260*0.5) * (1 - 0.304*0.5) * (1 - 0.2*0.5)
  = 0.968 * 0.870 * 0.848 * 0.90
  = 0.643
```

---

## R6. AgentNegotiator -- Proceed

```python
P_photon         = 0.90     # tier_prob["good"]
P_recoverability = 0.696
P_mismatch       = 1.0 - 0.260 * 0.7 = 0.818

P_joint = 0.90 * 0.696 * 0.818 = 0.512
```

No veto (P_joint > 0.15). Proceed.

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=4.0,
  proceed_recommended=True,
  warnings=[
    "Moderate mask shift (2 px) detected -- temporal artifacts possible",
    "SNR (37.4 dB) is adequate but not excellent -- consider longer exposure"
  ],
  what_to_upload=["measurement (2D snapshot)", "masks (3D mask array)"]
)
```

---

## R8. Pipeline Results

| Configuration | GAP-TV | GAP-denoise | EfficientSCI |
|---------------|--------|-------------|--------------|
| Ideal operator | 32.83 dB | 33.50 dB | 36.78 dB |
| With mismatch (real) | 28.45 dB | 29.10 dB | 33.20 dB |
| Detector-limited | 28.45 dB | 28.95 dB | 32.50 dB |

**Key findings:**
- CACTI at 8x compression achieves >32 dB with correct operator (usable video)
- EfficientSCI outperforms GAP-TV by +3.95 dB (deep learned temporal prior)
- Mask spatial shift is the most destructive mismatch (temporal jitter is second)
- Even with mismatch, EfficientSCI maintains >33 dB (robust learned features)

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Experiment |
|--------|-----------|-----------------|
| **Photon Agent** | | |
| N_effective | 2.16e4 | 5.56e3 |
| SNR | 43.3 dB | 37.4 dB |
| Quality tier | excellent | good |
| **Mismatch Agent** | | |
| Severity | 0.135 (low) | 0.260 (moderate) |
| Dominant error | mask_shift (typical) | mask_shift (2 px vibration) |
| Expected gain | +1.35 dB | +2.60 dB |
| **Recoverability Agent** | | |
| Score | 0.80 (sufficient) | 0.696 (sufficient) |
| Expected PSNR | 32.83 dB | 28.5 dB |
| Verdict | sufficient | sufficient |
| **Analysis Agent** | | |
| Primary bottleneck | compression | compression |
| P(success) | 0.755 | 0.643 |
| **Negotiator** | | |
| P_joint | 0.689 | 0.512 |
| **PreFlight** | | |
| Runtime | 4.0s | 4.0s |
| Warnings | 0 | 2 |
| **Pipeline** | | |
| Final PSNR (GAP-TV) | 32.83 dB | 28.45 dB |
| Final PSNR (EfficientSCI) | 36.78 dB | 33.20 dB |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (GAP-TV -> GAP-denoise -> EfficientSCI) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Temporal-aware:** CACTI's temporal compression (8 frames -> 1 snapshot) is the defining constraint, not photon budget.
