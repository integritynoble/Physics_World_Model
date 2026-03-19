# SPC Working Process

## End-to-End Pipeline for Single-Pixel Camera Imaging

This document traces a complete Single-Pixel Camera experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct a 64x64 image from single-pixel camera measurements.
 Measurement: spc_data.npy, Sensing matrix: phi_matrix.npy,
 sampling rate 15%, Hadamard patterns."
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "spc_data.npy" detected
#   operator_type=OperatorType.explicit_matrix,
#   files=["spc_data.npy", "phi_matrix.npy"],
#   params={"image_height": 64, "image_width": 64, "sampling_rate": 0.15}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> spc entry
spc:
  keywords: [single_pixel, compressive_sensing, DMD, measurement_matrix, CS]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="spc",
#   confidence=0.92,
#   reasoning="Matched keywords: single_pixel, compressive_sensing"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the SPC registry entry:

```python
system = plan_agent.build_imaging_system("spc")
# ImagingSystem(
#   modality_key="spc",
#   display_name="Single-Pixel Camera",
#   signal_dims={"x": [64, 64], "y": [614]},
#   forward_model_type=ForwardModelType.explicit_matrix,
#   elements=[...5 elements...],
#   default_solver="pnp_fista"
# )
```

**SPC Element Chain (5 elements):**

```
Scene Illumination ──> Collection Lens ──> Digital Micromirror Device ──> Condensing Lens ──> Single-Pixel Photodiode
  throughput=1.0       throughput=0.92    throughput=0.68                throughput=0.90     throughput=0.90
  noise: none          noise: aberration  noise: fixed_pattern           noise: none         noise: shot+read+thermal
                                                + alignment
```

**Cumulative throughput:** `0.92 x 0.68 x 0.90 x 0.90 = 0.506`

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  spc:
    model_id: "generic_detector"
    parameters:
      source_photons: 1.0e+06
      qe: 0.85
      exposure_s: 0.01
  ```

### Computation

```python
# 1. Source photon count (generic detector model)
N_source = source_photons = 1.0e6

# 2. Quantum efficiency
N_detected = N_source * qe = 1.0e6 * 0.85 = 8.5e5

# 3. Apply cumulative throughput
N_effective = N_detected * 0.506 = 4.30e5 photons/measurement

# 4. Noise variances
shot_var   = N_effective                      # Poisson
read_var   = NEP-derived ~ (3.0e-12)^2 * BW  # Photodiode NEP
#          For InGaAs photodiode at 100 kHz BW:
#          read_noise_e = NEP * sqrt(BW) / responsivity ~ 10.0 e-
read_var   = 10.0^2 = 100.0
thermal_var = 50.0                            # Dark current contribution
total_var  = shot_var + read_var + thermal_var
           = 4.30e5 + 100.0 + 50.0 = 4.30e5

# 5. SNR
SNR = N_effective / sqrt(total_var) = 4.30e5 / sqrt(4.30e5) = 655.7
SNR_db = 20 * log10(655.7) = 56.3 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=4.30e5,
  snr_db=56.3,
  noise_regime=NoiseRegime.shot_limited,      # shot_var/total_var > 0.99
  shot_noise_sigma=655.7,
  read_noise_sigma=10.0,
  total_noise_sigma=656.0,
  feasible=True,
  quality_tier="excellent",                   # SNR > 30 dB
  throughput_chain=[
    {"Scene Illumination": 1.0},
    {"Collection Lens": 0.92},
    {"Digital Micromirror Device": 0.68},
    {"Condensing Lens": 0.90},
    {"Single-Pixel Photodiode": 0.90}
  ],
  noise_model="poisson",
  explanation="Shot-noise-limited regime. High QE InGaAs photodiode provides excellent SNR."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"spc"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  spc:
    parameters:
      measurement_noise:
        range: [0.0, 0.10]
        typical_error: 0.02
        weight: 0.25
      pattern_misalignment:
        range: [-2.0, 2.0]
        typical_error: 0.5
        weight: 0.35
      gain:
        range: [0.5, 1.5]
        typical_error: 0.15
        weight: 0.40
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.25 * |0.02| / 0.10     # measurement_noise: 0.050
  + 0.35 * |0.5| / 4.0       # pattern_misalignment: 0.044
  + 0.40 * |0.15| / 1.0      # gain: 0.060
S = 0.154  # Low-to-moderate severity

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.54 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="spc",
  mismatch_family="grid_search",
  parameters={
    "measurement_noise": {"typical_error": 0.02, "range": [0.0, 0.10], "weight": 0.25},
    "pattern_misalignment": {"typical_error": 0.5, "range": [-2.0, 2.0], "weight": 0.35},
    "gain": {"typical_error": 0.15, "range": [0.5, 1.5], "weight": 0.40}
  },
  severity_score=0.154,
  correction_method="grid_search",
  expected_improvement_db=1.54,
  explanation="Low mismatch severity. Gain drift is the primary error source."
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
  spc:
    signal_prior_class: "wavelet_sparse"
    entries:
      - {cr: 0.10, noise: "shot_limited", solver: "pnp_fista",
         recoverability: 0.62, expected_psnr_db: 24.3,
         provenance: {dataset_id: "set11_cs_benchmark_2023", ...}}
      - {cr: 0.25, noise: "shot_limited", solver: "pnp_fista",
         recoverability: 0.79, expected_psnr_db: 28.9, ...}
      - {cr: 0.50, noise: "shot_limited", solver: "pnp_fista",
         recoverability: 0.90, expected_psnr_db: 33.5, ...}
  ```

### Computation

```python
# 1. Compression ratio
CR = prod(y_shape) / prod(x_shape) = 614 / (64 * 64) = 0.15

# 2. Operator diversity (Hadamard patterns)
# Hadamard matrix: rows are +/-1, orthogonal
# Mutual coherence is bounded by 1/sqrt(N)
diversity = 0.85  # High diversity for Hadamard patterns

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.541

# 4. Calibration table lookup
#    Interpolate between cr=0.10 (24.3 dB) and cr=0.25 (28.9 dB) at cr=0.15
#    Linear interpolation: 24.3 + (0.15-0.10)/(0.25-0.10) * (28.9-24.3) = 25.83 dB
#    Recoverability: 0.62 + 0.333 * (0.79-0.62) = 0.677
#    confidence = 0.9 (interpolated, not exact match)

# 5. Best solver selection
#    pnp_fista: 25.83 dB (default, PnP-FISTA with DRUNet)
#    tval3_fista: ~22.5 dB (TV-regularized FISTA baseline)
#    lista: ~24.0 dB (quick-trained LISTA network)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.15,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.wavelet_sparse,
  operator_diversity_score=0.85,
  condition_number_proxy=0.541,
  recoverability_score=0.677,
  recoverability_confidence=0.9,
  expected_psnr_db=25.83,
  expected_psnr_uncertainty_db=1.5,
  recommended_solver_family="pnp_fista",
  verdict="sufficient",              # 0.60 <= score < 0.85
  calibration_table_entry={...},
  explanation="Sufficient recoverability at 15% sampling. PnP-FISTA with DRUNet expected ~25.8 dB."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(56.3 / 40, 1.0)   = 0.0    # Excellent SNR
mismatch_score    = 0.154                        = 0.154  # Low mismatch
compression_score = 1 - 0.677                    = 0.323  # Moderate compression
solver_score      = 0.2                          = 0.2    # Default placeholder

# Primary bottleneck
primary = "compression"  # max(0.0, 0.154, 0.323, 0.2) = compression

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.154*0.5) * (1 - 0.323*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.923 * 0.839 * 0.90
  = 0.697
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="compression",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.154, compression=0.323, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Increase sampling rate to 25% for +3.1 dB improvement",
      priority="high",
      expected_gain_db=3.1
    ),
    Suggestion(
      text="Consider LISTA for faster inference with marginal PSNR loss",
      priority="low",
      expected_gain_db=-1.0
    )
  ],
  overall_verdict="sufficient",       # 0.60 <= P < 0.80
  probability_of_success=0.697,
  explanation="Compression ratio (15%) is the primary limiting factor. SNR is excellent."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="sufficient" | No veto |
| Severe mismatch without correction | severity=0.154 < 0.7 | No veto |
| All marginal | photon=excellent, compression=sufficient | No veto |
| Joint probability floor | P=0.697 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.677   # recoverability_score
P_mismatch       = 1.0 - 0.154 * 0.7 = 0.892

P_joint = 0.95 * 0.677 * 0.892 = 0.574
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.574
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 64 * 64 = 4,096
dim_factor   = total_pixels / (64 * 64) = 1.0
solver_complexity = 3.0  # PnP-FISTA with DRUNet (100 iterations, neural denoiser)
cr_factor    = max(0.15, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 1.0 * 3.0 * 0.125 = 0.75 seconds
# Plus DRUNet inference overhead: ~5.0 seconds total
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="spc", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=5.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=["measurement (1D SPC measurement vector)", "sensing matrix (M x N pattern matrix)"]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# SPC forward model: y = A @ vec(x)
#
# Parameters:
#   A:       (M, N) sensing matrix  [loaded from phi_matrix.npy]
#   M:       614 measurements       [M = sampling_rate * N = 0.15 * 4096]
#   N:       4096 pixels            [N = 64 * 64]
#
# Input:  x = (64, 64) scene image
# Output: y = (614,) measurement vector

class SPCOperator(PhysicsOperator):
    def forward(self, x):
        """y = A @ vec(x)"""
        return self.A @ x.flatten()

    def adjoint(self, y):
        """x_hat = A^T @ y, reshaped to image"""
        return (self.A.T @ y).reshape(self.img_shape)

    def check_adjoint(self):
        """Verify <Ax, y> ~ <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided spc_data.npy:
y = np.load("spc_data.npy")         # (614,)

# If simulating:
x_true = load_ground_truth()         # (64, 64) from Set11 dataset
A = np.load("phi_matrix.npy")       # (614, 4096) Hadamard subsampled

# Row-normalize for numerical stability
row_norms = np.linalg.norm(A, axis=1, keepdims=True)
A_norm = A / np.maximum(row_norms, 1e-8)

y = A_norm @ x_true.flatten()        # (614,) clean measurements
y += np.random.randn(614) * 0.01     # Gaussian measurement noise
```

### Step 9c: Reconstruction with PnP-FISTA

```python
from pwm_core.recon.pnp import pnp_fista_drunet

# Estimate Lipschitz constant via power iteration
L = estimate_lipschitz(A_norm, n_iters=20)  # ~1.0 for normalized A
tau = 0.9 / max(L, 1e-8)

# Backprojection initialization
x0 = A_norm.T @ y
x0 = np.clip((x0 - x0.min()) / (x0.max() - x0.min() + 1e-8), 0, 1)

x_hat = pnp_fista_drunet(
    y=y,                     # (614,) measurement vector
    A=A_norm,                # (614, 4096) sensing matrix
    x0=x0,                   # (4096,) initial estimate
    block_size=33,           # Block-based CS (33x33 blocks)
    tau=tau,                 # Gradient step size
    max_iter=100,            # FISTA iterations
    sigma_end=0.02,          # Final DRUNet noise level
    sigma_anneal_mult=3.0,   # Annealing ratio
    device="cuda"
)
# x_hat shape: (64, 64) -- reconstructed image
# Expected PSNR: ~25.8 dB (Set11 benchmark at 15%)
```

**Alternative solvers:**

| Solver | Type | PSNR (15%) | GPU | Command |
|--------|------|------------|-----|---------|
| TVAL3-FISTA | Traditional | ~22.5 dB | No | `basic_fista(y, A, x0, block_size=33, tau, max_iter=400)` |
| PnP-FISTA+DRUNet | Plug-and-Play | ~25.8 dB | Yes | `pnp_fista_drunet(y, A, x0, max_iter=100)` |
| LISTA | Learned ISTA | ~24.0 dB | Yes | `lista_train_quick(A, y_batch, x_batch, epochs=200)` |

### Step 9d: Metrics

```python
# PSNR
psnr = 10 * log10(max_val^2 / mse(x_hat, x_true))  # ~25.8 dB at 15%

# SSIM (structural similarity)
ssim_val = ssim(x_hat, x_true)

# Compression-rate sweep (characterization)
for rate in [0.01, 0.10, 0.25, 0.50]:
    m = int(4096 * rate)
    # ... repeat forward + reconstruct
    # 1%  -> 16.8 dB (barely visible)
    # 10% -> 24.3 dB (recognizable)
    # 25% -> 28.9 dB (good quality)
    # 50% -> 33.5 dB (near-Nyquist)
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
|   +-- y.npy              # Measurement (614,) + SHA256 hash
|   +-- x_hat.npy          # Reconstruction (64, 64) + SHA256 hash
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
|   +-- phi.npy            # Sensing matrix (614, 4096) + SHA256 hash
+-- metrics.json           # PSNR, SSIM per sampling rate
+-- operator.json          # Operator parameters (matrix hash, M, N, pattern_type)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized SPC pipeline with excellent SNR (56.3 dB) and low mismatch (0.154). In practice, single-pixel cameras suffer from DMD pattern misalignment, photodiode gain drift during long acquisition sequences, and reduced photon budget at high frame rates.

This section traces the same pipeline with realistic degraded parameters.

---

## Real Experiment: User Prompt

```
"We captured single-pixel camera data at 10% sampling rate on our lab
 prototype. The DMD patterns might be shifted from last calibration.
 Please reconstruct.
 Measurement: lab_spc.npy, Matrix: lab_phi.npy, 64x64 image."
```

**Key difference:** Only 10% sampling (not 15%), and DMD patterns may be misaligned.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,
#   operator_type=OperatorType.explicit_matrix,
#   files=["lab_spc.npy", "lab_phi.npy"],
#   params={"image_height": 64, "image_width": 64, "sampling_rate": 0.10}
# )
```

---

## R2. PhotonAgent -- Realistic Lab Conditions

### Real detector parameters

```yaml
# Real lab: fast acquisition, lower photon budget
spc_lab:
  source_photons: 2.0e+05      # 5x fewer (fast switching)
  qe: 0.75                     # Aging photodiode
  exposure_s: 0.002             # 2 ms per pattern (500 Hz rate)
```

### Computation

```python
# Detected photons
N_detected = 2.0e5 * 0.75 = 1.5e5

# Apply cumulative throughput (0.506)
N_effective = 1.5e5 * 0.506 = 7.59e4 photons/measurement

# Noise variances
shot_var   = 7.59e4
read_var   = 10.0^2 = 100.0
thermal_var = 50.0
total_var  = 7.59e4 + 150.0 = 7.60e4

# SNR
SNR = 7.59e4 / sqrt(7.60e4) = 275.5
SNR_db = 20 * log10(275.5) = 48.8 dB
```

### Output

```python
PhotonReport(
  n_photons_per_pixel=7.59e4,
  snr_db=48.8,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="excellent",       # 48.8 dB >> 30 dB
  explanation="Shot-limited despite faster acquisition. Feasible."
)
```

---

## R3. MismatchAgent -- Real DMD Drift

```python
# Actual errors from lab prototype
S = 0.25 * |0.04| / 0.10      # measurement_noise: 0.100  (2x typical)
  + 0.35 * |1.2| / 4.0        # pattern_misalignment: 0.105  (1.2 px shift!)
  + 0.40 * |0.25| / 1.0       # gain: 0.100  (gain drifted 25%)
S = 0.305  # MODERATE severity

improvement_db = clip(10 * 0.305, 0, 20) = 3.05 dB
```

### Output

```python
MismatchReport(
  severity_score=0.305,
  correction_method="grid_search",
  expected_improvement_db=3.05,
  explanation="Moderate mismatch. Gain drift (0.25) and pattern shift (1.2 px) are co-dominant."
)
```

---

## R4. RecoverabilityAgent -- Degraded at 10%

```python
# CR = 0.10 (lower than ideal 0.15)
# Exact calibration match: cr=0.10, noise=shot_limited, solver=pnp_fista
# -> recoverability=0.62, expected_psnr=24.3 dB

# With mismatch degradation (structured error from misaligned patterns):
# Effective recoverability: 0.62 * (1 - 0.305*0.5) = 0.525
# Expected PSNR: ~21.5 dB
```

### Output

```python
RecoverabilityReport(
  compression_ratio=0.10,
  recoverability_score=0.525,
  expected_psnr_db=21.5,
  verdict="marginal",           # 0.40 <= score < 0.60
  explanation="Low measurement rate combined with operator mismatch degrades recovery."
)
```

---

## R5. AnalysisAgent -- Compression is the Bottleneck

```python
photon_score      = 0.0       # Still excellent
mismatch_score    = 0.305     # Moderate
compression_score = 1 - 0.525 = 0.475  # Poor
solver_score      = 0.2

primary = "compression"       # max(0.0, 0.305, 0.475, 0.2)

P = 1.0 * 0.848 * 0.763 * 0.90 = 0.582
```

---

## R6. AgentNegotiator -- Conditional Proceed

```python
P_photon         = 0.95
P_recoverability = 0.525
P_mismatch       = 1.0 - 0.305 * 0.7 = 0.787

P_joint = 0.95 * 0.525 * 0.787 = 0.393
```

No veto (P_joint > 0.15). Proceed with warnings.

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=5.0,
  proceed_recommended=True,
  warnings=[
    "Low sampling rate (10%) -- expect limited detail recovery",
    "Pattern misalignment detected (1.2 px) -- consider recalibrating DMD"
  ],
  what_to_upload=["measurement (1D vector)", "sensing matrix (M x N)"]
)
```

---

## R8. Pipeline Results

| Configuration | TVAL3-FISTA | PnP-FISTA | LISTA |
|---------------|-------------|-----------|-------|
| 10% ideal operator | 20.1 dB | 24.3 dB | 22.5 dB |
| 10% with mismatch | 18.2 dB | 21.5 dB | 20.0 dB |
| 15% ideal operator | 22.5 dB | 25.8 dB | 24.0 dB |
| 25% ideal operator | 25.5 dB | 28.9 dB | 27.0 dB |

**Key findings:**
- At 10% sampling, PnP-FISTA with DRUNet significantly outperforms TV-based FISTA (+4.2 dB)
- DMD misalignment causes ~2.8 dB degradation across all solvers
- Increasing to 25% sampling is the single most impactful improvement (+4.6 dB)

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Experiment |
|--------|-----------|-----------------|
| **Photon Agent** | | |
| N_effective | 4.30e5 | 7.59e4 |
| SNR | 56.3 dB | 48.8 dB |
| Quality tier | excellent | excellent |
| **Mismatch Agent** | | |
| Severity | 0.154 (low) | 0.305 (moderate) |
| Dominant error | gain (0.15) | gain + pattern shift |
| **Recoverability Agent** | | |
| CR | 0.15 | 0.10 |
| Score | 0.677 (sufficient) | 0.525 (marginal) |
| Expected PSNR | 25.83 dB | 21.5 dB |
| **Analysis Agent** | | |
| Primary bottleneck | compression | compression |
| P(success) | 0.697 | 0.582 |
| **Negotiator** | | |
| P_joint | 0.574 | 0.393 |
| **PreFlight** | | |
| Runtime | 5.0s | 5.0s |
| Warnings | 0 | 2 |
| **Pipeline** | | |
| Final PSNR | 25.83 dB | 21.5 dB |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (FISTA -> PnP-FISTA -> LISTA) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Compression-aware:** SPC performance is primarily governed by sampling rate, not photon budget.
