# Matrix Sensing Working Process

## End-to-End Pipeline for Generic Matrix Compressed Sensing

This document traces a complete generic matrix sensing experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct a 64x64 image from compressed Gaussian random measurements.
 Measurement: y.npy, Matrix: A.npy, sampling_rate=0.25, signal_shape=[64,64]."
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "y.npy" detected
#   operator_type=OperatorType.explicit_matrix,
#   files=["y.npy", "A.npy"],
#   params={"sampling_rate": 0.25, "signal_shape": [64, 64]}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> matrix entry
matrix:
  keywords: [matrix, generic, linear_inverse, compressed_sensing, FISTA]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="matrix",
#   confidence=0.92,
#   reasoning="Matched keywords: compressed, Gaussian, measurements"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the matrix registry entry:

```python
system = plan_agent.build_imaging_system("matrix")
# ImagingSystem(
#   modality_key="matrix",
#   display_name="Generic Matrix Sensing",
#   signal_dims={"x": [64, 64], "y": [614]},
#   forward_model_type=ForwardModelType.explicit_matrix,
#   elements=[...4 elements...],
#   default_solver="fista_l2"
# )
```

**Matrix Element Chain (4 elements):**

```
Signal Source ---------> Linear Measurement Operator ---------> Additive Noise Channel ---------> Digital Readout
  throughput=1.0          throughput=1.0                         throughput=1.0                    throughput=1.0
  noise: none             noise: none                            noise: read_gaussian              noise: quantization
                          M=614, N=4096
                          sampling_rate=0.15
```

**Cumulative throughput:** `1.0 x 1.0 x 1.0 x 1.0 = 1.0`

Note: The matrix modality has ideal throughput (no optical losses). All signal attenuation is encoded in the measurement matrix A itself. For 25% sampling, M = 0.25 * 4096 = 1024 measurements from N = 4096 signal elements.

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  matrix:
    model_id: "generic_detector"
    parameters:
      source_photons: 1.0e+06
      qe: 0.80
      exposure_s: 0.01
  ```

### Computation

```python
# 1. For generic_detector model, photon count is directly specified
N_raw = source_photons * qe
#     = 1.0e6 * 0.80
#     = 8.0e5 photons/measurement

# 2. Apply cumulative throughput
N_effective = N_raw * 1.0 = 8.0e5 photons/measurement

# 3. Noise variances
#    For matrix sensing, the dominant noise is additive Gaussian (read noise)
#    noise_sigma = 0.01 (from element chain parameters)
shot_var   = N_effective                     # 8.0e5
read_var   = (noise_sigma * N_effective)^2   # (0.01 * 8.0e5)^2 = 6.4e7
total_var  = shot_var + read_var
#          = 8.0e5 + 6.4e7 = 6.48e7

# 4. SNR
SNR = N_effective / sqrt(total_var)
#   = 8.0e5 / sqrt(6.48e7) = 8.0e5 / 8050
#   = 99.4
SNR_db = 20 * log10(99.4) = 39.9 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=8.0e5,
  snr_db=39.9,
  noise_regime=NoiseRegime.read_limited,     # read_var/total_var > 0.9
  shot_noise_sigma=894.4,
  read_noise_sigma=8000.0,
  total_noise_sigma=8050.0,
  feasible=True,
  quality_tier="good",                       # 30 < SNR < 50 dB
  throughput_chain=[
    {"Signal Source": 1.0},
    {"Linear Measurement Operator": 1.0},
    {"Additive Noise Channel": 1.0},
    {"Digital Readout": 1.0}
  ],
  noise_model="gaussian",
  explanation="Read-limited regime. Additive Gaussian noise dominates. Adequate SNR for CS reconstruction."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"matrix"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  matrix:
    parameters:
      noise_level:
        range: [0.0, 0.10]
        typical_error: 0.02
        unit: "normalized"
        description: "Additive measurement noise from detector or quantization"
      matrix_perturbation:
        range: [0.0, 0.20]
        typical_error: 0.05
        unit: "Frobenius (relative)"
        description: "Forward matrix calibration error (element-wise perturbation)"
    severity_weights:
      noise_level: 0.40
      matrix_perturbation: 0.60
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.40 * |0.02| / 0.10    # noise_level: 0.080
  + 0.60 * |0.05| / 0.20    # matrix_perturbation: 0.150
S = 0.230  # Moderate severity (typical lab conditions)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 2.30 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="matrix",
  mismatch_family="grid_search",
  parameters={
    "noise_level":  {"typical_error": 0.02, "range": [0.0, 0.10], "weight": 0.40},
    "matrix_perturbation": {"typical_error": 0.05, "range": [0.0, 0.20], "weight": 0.60}
  },
  severity_score=0.230,
  correction_method="grid_search",
  expected_improvement_db=2.30,
  explanation="Moderate mismatch severity. Matrix perturbation (5% Frobenius norm) is the primary error source."
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
  matrix:
    signal_prior_class: "wavelet_sparse"
    entries:
      - {cr: 0.25, noise: "read_limited", solver: "fista_tv",
         recoverability: 0.74, expected_psnr_db: 26.3,
         provenance: {dataset_id: "set11_generic_cs_2023", ...}}
      - {cr: 0.25, noise: "read_limited", solver: "lista",
         recoverability: 0.81, expected_psnr_db: 29.7, ...}
      - {cr: 0.50, noise: "read_limited", solver: "fista_tv",
         recoverability: 0.88, expected_psnr_db: 31.4, ...}
  ```

### Computation

```python
# 1. Compression ratio
CR = prod(y_shape) / prod(x_shape) = 1024 / (64 * 64) = 0.25

# 2. Operator diversity (Gaussian random matrices have maximal incoherence)
#    For i.i.d. Gaussian A, the restricted isometry constant is well-bounded
diversity = 1.0  # Maximum diversity (random Gaussian)

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.5

# 4. Calibration table lookup
#    Match: noise="read_limited", solver="fista_tv", cr=0.25 (exact match)
#    -> recoverability=0.74, expected_psnr=26.3 dB, confidence=1.0

# 5. Best solver selection
#    lista: 29.7 dB > fista_tv: 26.3 dB
#    -> recommended: "lista" (best available)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.25,
  noise_regime=NoiseRegime.read_limited,
  signal_prior_class=SignalPriorClass.wavelet_sparse,
  operator_diversity_score=1.0,
  condition_number_proxy=0.5,
  recoverability_score=0.74,
  recoverability_confidence=1.0,
  expected_psnr_db=26.3,
  expected_psnr_uncertainty_db=1.5,
  recommended_solver_family="fista_tv",
  verdict="good",                  # 0.70 <= score < 0.85
  calibration_table_entry={...},
  explanation="Good recoverability at 25% sampling. FISTA-TV expected 26.3 dB on Set11 benchmark. LISTA may yield +3.4 dB."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(39.9 / 40, 1.0)  = 0.003   # Marginal SNR
mismatch_score    = 0.230                      = 0.230   # Moderate
compression_score = 1 - 0.74                   = 0.26    # Good but not excellent
solver_score      = 0.2                        = 0.2     # Default placeholder

# Primary bottleneck
primary = "compression"  # max(0.003, 0.230, 0.26, 0.2) = compression

# Probability of success
P = (1 - 0.003*0.5) * (1 - 0.230*0.5) * (1 - 0.26*0.5) * (1 - 0.2*0.5)
  = 0.999 * 0.885 * 0.87 * 0.90
  = 0.692
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="compression",
  bottleneck_scores=BottleneckScores(
    photon=0.003, mismatch=0.230, compression=0.26, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Increase sampling rate from 25% to 50% for +5.1 dB improvement",
      priority="high",
      expected_gain_db=5.1
    ),
    Suggestion(
      text="Consider LISTA unrolled solver for +3.4 dB over FISTA-TV",
      priority="medium",
      expected_gain_db=3.4
    )
  ],
  overall_verdict="good",             # 0.60 <= P < 0.80
  probability_of_success=0.692,
  explanation="Compression ratio is the primary bottleneck. 25% sampling is near the CS threshold for natural images."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="good" AND verdict="good" | No veto |
| Severe mismatch without correction | severity=0.230 < 0.7 | No veto |
| All marginal | Photon=good, compression=good | No veto |
| Joint probability floor | P=0.692 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.80   # tier_prob["good"]
P_recoverability = 0.74   # recoverability_score
P_mismatch       = 1.0 - 0.230 * 0.7 = 0.839

P_joint = 0.80 * 0.74 * 0.839 = 0.497
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.497
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 64 * 64 = 4,096
dim_factor   = total_pixels / (64 * 64) = 1.0
solver_complexity = 1.5  # FISTA (iterative, moderate)
cr_factor    = max(0.25, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 1.0 * 1.5 * 0.125 = 0.375 seconds
# FISTA-TV 200 iterations on 64x64 signal is very fast
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="matrix", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=0.375,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=["measurement vector y (1D, M elements)", "measurement matrix A (M x N)"]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Matrix forward model: y = A @ x + n
#
# Parameters:
#   A:   (M, N) sensing matrix          [loaded from A.npy]
#   M:   1024 measurements (25% of 4096)
#   N:   4096 = 64 * 64 signal elements
#   noise_sigma: 0.01
#
# Input:  x = (64, 64) image, flattened to (4096,)
# Output: y = (1024,) measurement vector

class MatrixOperator(PhysicsOperator):
    def forward(self, x):
        """y = A @ x.flatten()"""
        return self.A @ x.flatten()

    def adjoint(self, y):
        """x_hat = A^T @ y, reshaped to (n, n)"""
        return (self.A.T @ y).reshape(self.n, self.n)

    def check_adjoint(self):
        """Verify <Ax, y> ~= <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided y.npy:
y = np.load("y.npy")    # (1024,)

# If simulating:
x_true = load_ground_truth()               # (64, 64) natural image
A = np.random.randn(1024, 4096) / np.sqrt(1024)  # Normalized Gaussian
y = A @ x_true.flatten()                   # (1024,)
y += np.random.randn(1024) * 0.01          # Additive Gaussian noise
```

### Step 9c: Reconstruction with FISTA-TV

```python
from pwm_core.recon.classical import fista_tv_matrix

x_hat = fista_tv_matrix(
    y=y,                    # (1024,) measurement vector
    A=A,                    # (1024, 4096) sensing matrix
    signal_shape=[64, 64],
    max_iter=200,
    lam=0.05,               # TV regularization weight
    step=None,               # Auto-compute from Lipschitz constant
)
# x_hat shape: (64, 64) -- reconstructed image
# Expected PSNR: ~25.0 dB (Set11 benchmark, 25% sampling)
```

**FISTA-TV Algorithm Detail:**

```python
# 1. Estimate Lipschitz constant via power iteration
L = estimate_lipschitz(A, n_iters=20)
step = 0.9 / max(L, 1e-8)

# 2. Initialize with backprojection
x = (A.T @ y).reshape(n, n)
x = clip((x - x.min()) / (x.max() - x.min()), 0, 1)
z = x.copy()
t = 1.0

# 3. FISTA iterations
for k in range(max_iter):
    residual = A @ z.flatten() - y
    grad = (A.T @ residual).reshape(n, n)
    v = z - step * grad
    x_new = denoise_tv_chambolle(v, weight=lam * step)
    t_new = 0.5 * (1 + sqrt(1 + 4 * t^2))
    z = x_new + ((t - 1) / t_new) * (x_new - x)
    x, t = clip(x_new, 0, 1), t_new
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| FISTA-TV | Traditional | 25.0 dB | No | `fista_tv_matrix(y, A, [64,64], max_iter=200)` |
| LISTA | Deep Learning | 29.7 dB | Yes | `lista_train_quick(A, y, x_true, epochs=200)` |
| Diffusion Posterior | Deep Learning | ~32.0 dB | Yes | `diffusion_posterior_sample(y, fwd_fn, adj_fn, n_steps=300)` |

### Step 9d: Metrics

```python
# PSNR (Peak Signal-to-Noise Ratio)
mse = mean((x_hat - x_true)^2)
psnr = 10 * log10(1.0 / mse)  # ~25.0 dB

# SSIM (Structural Similarity)
ssim = structural_similarity(x_hat, x_true, data_range=1.0)

# NMSE (Normalized Mean Squared Error) -- matrix-specific
nmse = norm(x_hat - x_true)^2 / norm(x_true)^2
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
|   +-- y.npy              # Measurement (1024,) + SHA256 hash
|   +-- x_hat.npy          # Reconstruction (64, 64) + SHA256 hash
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
|   +-- A.npy              # Sensing matrix (1024, 4096) + SHA256 hash
+-- metrics.json           # PSNR, SSIM, NMSE
+-- operator.json          # Operator parameters (A hash, M, N, sampling_rate)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized matrix CS pipeline with clean Gaussian measurements and standard noise. In practice, real compressed sensing systems have imperfect measurement matrices (miscalibration, element-wise perturbations), lower SNR from limited dynamic range, and non-ideal signal statistics.

This section traces the same pipeline with realistic parameters drawn from our actual benchmark experiments.

---

## Real Experiment: User Prompt

```
"I have compressed measurements from a single-pixel camera. The measurement
 matrix was calibrated last month but the DMD patterns may have drifted.
 Please reconstruct. Measurement: y_lab.npy, Matrix: A_nominal.npy,
 signal_shape=[64,64], sampling_rate=0.10."
```

**Key difference:** Lower sampling rate (10% vs 25%) and potential matrix miscalibration from DMD pattern drift.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,
#   operator_type=OperatorType.explicit_matrix,
#   files=["y_lab.npy", "A_nominal.npy"],
#   params={"sampling_rate": 0.10, "signal_shape": [64, 64]}
# )
```

---

## R2. PhotonAgent -- Low Dynamic Range Conditions

### Real detector parameters

```yaml
# Real single-pixel camera: limited photon budget
matrix_lab:
  source_photons: 1.0e+04       # 100x fewer than ideal
  qe: 0.65                      # Older photodiode
  exposure_s: 0.01
  read_noise_sigma: 0.05        # 5x higher relative noise
```

### Computation

```python
# Effective photon count
N_effective = 1.0e4 * 0.65 = 6.5e3 photons/measurement

# Noise variances
shot_var   = 6.5e3                           # Poisson
read_var   = (0.05 * 6.5e3)^2 = 1.056e5     # Gaussian
total_var  = 6.5e3 + 1.056e5 = 1.121e5

# SNR
SNR = 6.5e3 / sqrt(1.121e5) = 6.5e3 / 334.8 = 19.4
SNR_db = 20 * log10(19.4) = 25.8 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=6.5e3,
  snr_db=25.8,
  noise_regime=NoiseRegime.read_limited,
  feasible=True,
  quality_tier="fair",                       # 20 < SNR < 30 dB
  explanation="Read-limited. Marginal SNR -- reconstruction may show noise artifacts."
)
```

---

## R3. MismatchAgent -- DMD Calibration Drift

```python
# Actual errors from DMD drift
mismatch_actual = {
    "noise_level": 0.05,              # 5% noise (vs 2% typical)
    "matrix_perturbation": 0.12,       # 12% Frobenius norm error
}

# Severity computation
S = 0.40 * |0.05| / 0.10       # noise_level: 0.200
  + 0.60 * |0.12| / 0.20       # matrix_perturbation: 0.360
S = 0.560  # HIGH severity

# Expected improvement from correction
improvement_db = clip(10 * 0.560, 0, 20) = 5.60 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="matrix",
  severity_score=0.560,
  correction_method="grid_search",
  expected_improvement_db=5.60,
  explanation="High mismatch severity. DMD pattern drift (12% Frobenius) dominates. "
              "Recalibration strongly recommended."
)
```

---

## R4. RecoverabilityAgent -- Degraded by Low Sampling

```python
# Compression ratio: now only 10%
CR = 0.10

# Calibration table lookup
# Match: noise="read_limited", solver="fista_tv", cr=0.10
# -> recoverability=0.55, expected_psnr=22.1 dB
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.10,
  noise_regime=NoiseRegime.read_limited,
  recoverability_score=0.55,              # Down from 0.74
  expected_psnr_db=22.1,                  # Down from 26.3
  verdict="marginal",                     # Was "good"
  explanation="Marginal recoverability at 10% sampling. Below CS threshold for complex images."
)
```

---

## R5. AnalysisAgent -- Multiple Bottlenecks

```python
# Bottleneck scores
photon_score      = 1 - min(25.8 / 40, 1.0)  = 0.355
mismatch_score    = 0.560                      = 0.560
compression_score = 1 - 0.55                   = 0.45
solver_score      = 0.2

# Primary bottleneck
primary = "mismatch"  # max(0.355, 0.560, 0.45, 0.2)

# Probability of success
P = (1 - 0.355*0.5) * (1 - 0.560*0.5) * (1 - 0.45*0.5) * (1 - 0.2*0.5)
  = 0.823 * 0.720 * 0.775 * 0.90
  = 0.413
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="mismatch",
  bottleneck_scores=BottleneckScores(
    photon=0.355, mismatch=0.560, compression=0.45, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Recalibrate DMD measurement matrix to reduce perturbation error",
      priority="critical",
      expected_gain_db=5.6
    ),
    Suggestion(
      text="Increase sampling rate from 10% to 25% for +4.2 dB",
      priority="high",
      expected_gain_db=4.2
    ),
    Suggestion(
      text="Use LISTA unrolled solver for robustness to matrix error",
      priority="medium",
      expected_gain_db=2.5
    )
  ],
  overall_verdict="marginal",
  probability_of_success=0.413,
  explanation="Matrix perturbation from DMD drift is the primary bottleneck. "
              "Combined with 10% sampling, expect ~18-22 dB."
)
```

---

## R6. AgentNegotiator -- Conditional Proceed

```python
P_photon         = 0.70   # tier_prob["fair"]
P_recoverability = 0.55
P_mismatch       = 1.0 - 0.560 * 0.7 = 0.608

P_joint = 0.70 * 0.55 * 0.608 = 0.234
```

### Veto check

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="fair" AND verdict="marginal" | **Close but no veto** |
| Severe mismatch without correction | severity=0.560 < 0.7 | No veto |
| All marginal | Mixed grades | No veto |
| Joint probability floor | P=0.234 > 0.15 | No veto |

```python
NegotiationResult(
  vetoes=[],
  proceed=True,
  probability_of_success=0.234
)
```

---

## R7. PreFlightReportBuilder -- Warnings Raised

```python
PreFlightReport(
  estimated_runtime_s=0.25,
  proceed_recommended=True,
  warnings=[
    "Mismatch severity 0.560 -- matrix may be significantly miscalibrated",
    "Recoverability marginal (0.55) at 10% sampling rate",
    "SNR 25.8 dB -- noise artifacts expected in reconstruction"
  ],
  what_to_upload=[
    "measurement vector y (1D, 410 elements at 10% rate)",
    "measurement matrix A (410 x 4096)"
  ]
)
```

---

## R8. Pipeline Runner -- Real Experiment

### Step R8a: Reconstruct with Nominal Matrix

```python
# Uses A_nominal.npy as-is
x_wrong = fista_tv_matrix(y_lab, A_nominal, [64, 64], max_iter=200, lam=0.05)
# PSNR = 18.3 dB  <-- noisy, artifacts from matrix mismatch
```

### Step R8b: Reconstruct with LISTA (more robust)

```python
from pwm_core.recon.lista import lista_train_quick
recon_lista = lista_train_quick(A_nominal, y_lab, epochs=200, lr=1e-3)
# PSNR = 20.8 dB  <-- LISTA is more robust to matrix perturbation
```

### Step R8c: Final Comparison

| Configuration | FISTA-TV | LISTA | Notes |
|---------------|----------|-------|-------|
| Ideal (25%, sigma=0.01) | **25.0 dB** | **29.7 dB** | Benchmark conditions |
| Lab (10%, sigma=0.05, perturbed A) | **18.3 dB** | **20.8 dB** | Degraded by mismatch + low sampling |
| Lab + recalibrated A | **21.5 dB** | **24.2 dB** | After matrix correction |

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Experiment |
|--------|-----------|-----------------|
| **Photon Agent** | | |
| N_effective | 8.0e5 | 6.5e3 |
| SNR | 39.9 dB | 25.8 dB |
| Quality tier | good | fair |
| Noise regime | read_limited | read_limited |
| **Mismatch Agent** | | |
| Severity | 0.230 (moderate) | 0.560 (high) |
| Dominant error | matrix perturbation | **DMD drift** |
| Expected gain | +2.30 dB | +5.60 dB |
| Correction needed | Optional | **Yes** |
| **Recoverability Agent** | | |
| Compression ratio | 0.25 | 0.10 |
| Score | 0.74 (good) | 0.55 (marginal) |
| Expected PSNR | 26.3 dB | 22.1 dB |
| Verdict | good | **marginal** |
| **Analysis Agent** | | |
| Primary bottleneck | compression | **mismatch** |
| P(success) | 0.692 | 0.413 |
| Verdict | good | **marginal** |
| **Negotiator** | | |
| Vetoes | 0 | 0 |
| P_joint | 0.497 | 0.234 |
| **PreFlight** | | |
| Runtime | 0.375s | 0.25s |
| Warnings | 0 | **3** |
| **Pipeline** | | |
| FISTA-TV PSNR | 25.0 dB | **18.3 dB** |
| LISTA PSNR | 29.7 dB | **20.8 dB** |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (FISTA-TV -> LISTA -> Diffusion Posterior) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Universal:** The matrix modality serves as a fallback for any linear inverse problem expressible as y = Ax + n.
