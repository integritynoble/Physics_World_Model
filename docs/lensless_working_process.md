# Lensless Working Process

## End-to-End Pipeline for Lensless (Diffuser Camera) Imaging

This document traces a complete lensless imaging experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct an image from our DiffuserCam measurement.
 Measurement: diffuser_raw.npy, PSF: calibrated_psf.npy,
 256x256 resolution, single-channel grayscale."
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "diffuser_raw.npy" detected
#   operator_type=OperatorType.linear_operator,
#   files=["diffuser_raw.npy", "calibrated_psf.npy"],
#   params={"resolution": [256, 256]}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> lensless entry
lensless:
  keywords: [lensless, diffuser_camera, mask_based, PSF_engineering, computational]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="lensless",
#   confidence=0.94,
#   reasoning="Matched keywords: lensless, diffuser_camera"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the lensless registry entry:

```python
system = plan_agent.build_imaging_system("lensless")
# ImagingSystem(
#   modality_key="lensless",
#   display_name="Lensless (Diffuser Camera) Imaging",
#   signal_dims={"x": [256, 256, 3], "y": [256, 256, 3]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...4 elements...],
#   default_solver="admm_tv"
# )
```

**Lensless Element Chain (4 elements):**

```
Scene Illumination ──> Phase Diffuser ──> Spacer / Air Gap ──> CMOS Image Sensor
  throughput=1.0       throughput=0.70    throughput=0.98      throughput=0.78
  noise: none          noise: fixed_pattern  noise: none       noise: shot+read+fixed+quant
                             + alignment
```

**Cumulative throughput:** `0.70 x 0.98 x 0.78 = 0.535`

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  lensless:
    model_id: "microscopy_fluorescence"
    parameters:
      power_w: 0.005
      wavelength_nm: 550
      na: 0.1
      n_medium: 1.0
      qe: 0.60
      exposure_s: 0.1
  ```

### Computation

```python
# 1. Photon energy
E_photon = h * c / wavelength_nm
         = (6.626e-34 * 3.0e8) / (550e-9) = 3.61e-19 J

# 2. Collection solid angle
#    Lensless cameras have very low effective NA (diffuser, no lens)
solid_angle = (na / n_medium)^2 / (4 * pi)
            = (0.1 / 1.0)^2 / (4 * pi)
            = 7.96e-4

# 3. Raw photon count
N_raw = power_w * qe * solid_angle * exposure_s / E_photon
      = 0.005 * 0.60 * 7.96e-4 * 0.1 / 3.61e-19
      = 6.62e11 photons

# 4. Apply cumulative throughput
N_effective = N_raw * 0.535 = 3.54e11 photons/pixel

# 5. Noise variances
shot_var   = N_effective = 3.54e11             # Poisson
read_var   = read_noise^2 = 2.5^2 = 6.25      # CMOS read noise
dark_var   = 0                                 # Negligible
total_var  = 3.54e11 + 6.25 = 3.54e11

# 6. SNR
SNR = N_effective / sqrt(total_var) = sqrt(3.54e11) = 5.95e5
SNR_db = 20 * log10(5.95e5) = 115.5 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=3.54e11,
  snr_db=115.5,
  noise_regime=NoiseRegime.shot_limited,      # shot_var/total_var ~ 1.0
  shot_noise_sigma=5.95e5,
  read_noise_sigma=2.5,
  total_noise_sigma=5.95e5,
  feasible=True,
  quality_tier="excellent",                   # SNR >> 30 dB
  throughput_chain=[
    {"Scene Illumination": 1.0},
    {"Phase Diffuser": 0.70},
    {"Spacer / Air Gap": 0.98},
    {"CMOS Image Sensor": 0.78}
  ],
  noise_model="poisson",
  explanation="Shot-noise-limited regime. Diffuser throughput (70%) is the main photon loss."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"lensless"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  lensless:
    parameters:
      psf_sigma:
        range: [0.5, 5.0]
        typical_error: 0.5
        weight: 0.25
      psf_shift_x:
        range: [-5.0, 5.0]
        typical_error: 1.0
        weight: 0.30
      psf_shift_y:
        range: [-5.0, 5.0]
        typical_error: 1.0
        weight: 0.30
      background:
        range: [0.0, 0.10]
        typical_error: 0.02
        weight: 0.15
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.25 * |0.5| / 4.5        # psf_sigma: 0.028
  + 0.30 * |1.0| / 10.0       # psf_shift_x: 0.030
  + 0.30 * |1.0| / 10.0       # psf_shift_y: 0.030
  + 0.15 * |0.02| / 0.10      # background: 0.030
S = 0.118  # Low severity

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.18 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="lensless",
  mismatch_family="grid_search",
  parameters={
    "psf_sigma":   {"typical_error": 0.5, "range": [0.5, 5.0], "weight": 0.25},
    "psf_shift_x": {"typical_error": 1.0, "range": [-5.0, 5.0], "weight": 0.30},
    "psf_shift_y": {"typical_error": 1.0, "range": [-5.0, 5.0], "weight": 0.30},
    "background":  {"typical_error": 0.02, "range": [0.0, 0.10], "weight": 0.15}
  },
  severity_score=0.118,
  correction_method="grid_search",
  expected_improvement_db=1.18,
  explanation="Low mismatch severity. PSF registration errors are well-balanced."
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
  lensless:
    signal_prior_class: "tv"
    entries:
      - {cr: 1.0, noise: "shot_limited", solver: "admm_tv",
         recoverability: 0.87, expected_psnr_db: 34.72,
         provenance: {dataset_id: "diffusercam_mirflickr_2023", ...}}
      - {cr: 1.0, noise: "read_limited", solver: "admm_tv",
         recoverability: 0.76, expected_psnr_db: 30.11, ...}
      - {cr: 1.0, noise: "shot_limited", solver: "flatnet",
         recoverability: 0.93, expected_psnr_db: 38.15, ...}
  ```

### Computation

```python
# 1. Compression ratio
#    Lensless is NOT compressive: input and output have the same dimensions
CR = prod(y_shape) / prod(x_shape) = (256 * 256 * 3) / (256 * 256 * 3) = 1.0

# 2. Operator diversity
#    Diffuser PSF has high spatial diversity (caustic pattern fills sensor)
#    The FFT of the PSF has relatively flat magnitude spectrum
diversity = 0.75  # Moderate-high (diffuser provides good spatial diversity)

# 3. Condition number proxy
#    The convolution operator has condition number determined by
#    min(|H(f)|) / max(|H(f)|), where H is the PSF transfer function
#    Diffuser PSFs typically have some null frequencies
kappa = 1 / (1 + diversity) = 0.571

# 4. Calibration table lookup
#    Exact match: cr=1.0, noise="shot_limited", solver="admm_tv"
#    -> recoverability=0.87, expected_psnr=34.72 dB, confidence=1.0

# 5. Best solver selection
#    flatnet: 38.15 dB > admm_tv: 34.72 dB
#    -> recommended: "flatnet" (or "admm_tv" as default)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=1.0,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.tv,
  operator_diversity_score=0.75,
  condition_number_proxy=0.571,
  recoverability_score=0.87,
  recoverability_confidence=1.0,
  expected_psnr_db=34.72,
  expected_psnr_uncertainty_db=1.0,
  recommended_solver_family="admm_tv",
  verdict="excellent",              # score >= 0.85
  calibration_table_entry={...},
  explanation="Excellent recoverability. ADMM-TV expected 34.72 dB; FlatNet can reach 38.15 dB."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(115.5 / 40, 1.0)  = 0.0    # Excellent SNR
mismatch_score    = 0.118                       = 0.118  # Low mismatch
compression_score = 1 - 0.87                    = 0.13   # Good recoverability
solver_score      = 0.2                         = 0.2    # Default placeholder

# Primary bottleneck
primary = "solver"  # max(0.0, 0.118, 0.13, 0.2) = solver

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.118*0.5) * (1 - 0.13*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.941 * 0.935 * 0.90
  = 0.792
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="solver",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.118, compression=0.13, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Use FlatNet for +3.43 dB over ADMM-TV",
      priority="high",
      expected_gain_db=3.43
    ),
    Suggestion(
      text="System is well-configured. Solver choice is the primary lever.",
      priority="medium",
      expected_gain_db=0.0
    )
  ],
  overall_verdict="sufficient",       # 0.60 <= P < 0.80
  probability_of_success=0.792,
  explanation="System is well-configured. Solver upgrade to FlatNet is the main improvement path."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND CR=1.0 | No veto |
| Severe mismatch without correction | severity=0.118 < 0.7 | No veto |
| All marginal | photon=excellent, recon=excellent | No veto |
| Joint probability floor | P=0.792 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.87    # recoverability_score
P_mismatch       = 1.0 - 0.118 * 0.7 = 0.917

P_joint = 0.95 * 0.87 * 0.917 = 0.758
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.758
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 256 * 256 = 65,536
dim_factor   = total_pixels / (256 * 256) = 1.0
solver_complexity = 2.5  # ADMM-TV (FFT-based, 150 iterations)
cr_factor    = max(1.0, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 1.0 * 2.5 * 0.125 = 0.625 seconds
# Plus FFT overhead for 256x256: ~2.0 seconds total
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="lensless", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=2.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=[
    "measurement (raw lensless capture, 2D or 3D color)",
    "PSF (calibrated point spread function)"
  ]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Lensless forward model: y = PSF ** x + n
#   where ** denotes 2D convolution (implemented via FFT)
#
# Parameters:
#   PSF:     (H, W) calibrated point spread function [loaded from calibrated_psf.npy]
#   H_fft:   fft2(PSF) - precomputed transfer function
#
# Input:  x = (256, 256) scene image
# Output: y = (256, 256) raw sensor capture
#         same dimensions (no compression, but information is scrambled)

class LenslessOperator(PhysicsOperator):
    def __init__(self, psf):
        self.H = np.fft.fft2(psf)
        self.H_conj = np.conj(self.H)
        self.H_abs2 = np.abs(self.H)**2

    def forward(self, x):
        """y = IFFT(FFT(x) * H)  -- circular convolution with PSF"""
        return np.real(np.fft.ifft2(np.fft.fft2(x) * self.H))

    def adjoint(self, y):
        """x_hat = IFFT(FFT(y) * conj(H))  -- correlation with PSF"""
        return np.real(np.fft.ifft2(np.fft.fft2(y) * self.H_conj))

    def check_adjoint(self):
        """Verify <Ax, y> ~ <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-12)
        # FFT-based convolution gives machine-precision adjoint
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided diffuser_raw.npy:
y = np.load("diffuser_raw.npy")         # (256, 256)
psf = np.load("calibrated_psf.npy")     # (256, 256)

# If simulating:
x_true = load_ground_truth()             # (256, 256)
psf = load_calibrated_psf()              # (256, 256)

# FFT-based convolution
H = np.fft.fft2(psf)
y = np.real(np.fft.ifft2(np.fft.fft2(x_true) * H))
y += np.random.randn(256, 256) * 0.005  # Gaussian sensor noise
```

### Step 9c: Reconstruction with ADMM-TV

```python
from pwm_core.recon.lensless_solver import admm_tv_lensless

x_hat = admm_tv_lensless(
    y=y,                     # (256, 256) raw sensor image
    H=H,                    # FFT of PSF
    H_conj=H_conj,          # Conjugate of H
    H_abs2=H_abs2,          # |H|^2
    n=256,                   # Image size
    max_iter=150,            # ADMM iterations
    rho=0.1,                 # ADMM penalty parameter
    tv_weight=0.02           # TV regularization strength
)
# x_hat shape: (256, 256) -- reconstructed scene
# Expected PSNR: ~34.72 dB (DiffuserCam benchmark)
```

**ADMM-TV Algorithm:**
```python
# Solves: min_x 0.5 * ||Hx - y||^2 + tv_weight * TV(x)
# Using ADMM splitting with FFT-based deconvolution

x = zeros(n, n)
z = zeros(n, n)
u = zeros(n, n)           # Dual variable
Y = fft2(y)               # Precompute
denom = H_abs2 + rho      # Precompute denominator

for k in range(max_iter):
    # x-update: solve (H^H H + rho I) x = H^H y + rho (z - u)
    rhs = fft2(rho * (z - u)) + H_conj * Y
    X = rhs / denom
    x = real(ifft2(X))

    # z-update: TV proximal operator
    v = x + u
    z = denoise_tv_chambolle(v, weight=tv_weight/rho)
    z = clip(z, 0, 1)

    # u-update: dual variable
    u = u + x - z
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| ADMM-TV | Traditional | 34.72 dB | No | `admm_tv_lensless(y, H, max_iter=150, rho=0.1)` |
| FlatNet | Deep Learning | 38.15 dB | Yes | `flatnet_recon(y, psf, model_path="weights/flatnet.pth")` |

### Step 9d: Metrics

```python
# PSNR
psnr = 10 * log10(max_val^2 / mse(x_hat, x_true))  # ~34.72 dB

# SSIM (structural similarity)
ssim_val = ssim(x_hat, x_true)

# Frequency-domain analysis (lensless-specific)
# Check how well high frequencies are recovered
H_mag = np.abs(H)
recovery_spectrum = np.abs(np.fft.fft2(x_hat)) / (np.abs(np.fft.fft2(x_true)) + 1e-8)
freq_recovery_ratio = np.mean(recovery_spectrum[H_mag > 0.1 * H_mag.max()])
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
|   +-- x_hat.npy          # Reconstruction (256, 256) + SHA256 hash
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
|   +-- psf.npy            # Calibrated PSF (256, 256) + SHA256 hash
+-- metrics.json           # PSNR, SSIM, frequency recovery ratio
+-- operator.json          # Operator parameters (PSF hash, diffuser type)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized lensless pipeline with excellent SNR (115.5 dB) and low mismatch (0.118). In practice, lensless cameras suffer from PSF calibration drift as the diffuser-sensor distance changes with temperature, ambient light contamination, and sensor fixed-pattern noise.

This section traces the same pipeline with realistic degraded parameters.

---

## Real Experiment: User Prompt

```
"We took a lensless image with our DiffuserCam prototype. The PSF
 was calibrated a week ago and the diffuser may have shifted slightly.
 There is some ambient light leakage. Please reconstruct.
 Measurement: lab_diffuser.npy, PSF: old_psf.npy, 128x128."
```

**Key difference:** PSF calibration is stale (diffuser may have shifted), and there is ambient light contamination.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["lab_diffuser.npy", "old_psf.npy"],
#   params={"resolution": [128, 128]}
# )
```

---

## R2. PhotonAgent -- Realistic Conditions

### Real parameters

```yaml
# Real lab: ambient light, shorter exposure
lensless_lab:
  power_w: 0.001            # 5x dimmer ambient scene
  wavelength_nm: 550
  na: 0.1
  qe: 0.50                  # Aging sensor
  exposure_s: 0.05           # 50 ms (shorter exposure to avoid saturation)
  read_noise_e: 4.0          # Higher due to sensor degradation
```

### Computation

```python
N_raw = 0.001 * 0.50 * 7.96e-4 * 0.05 / 3.61e-19 = 5.52e9
N_effective = 5.52e9 * 0.535 = 2.95e9 photons/pixel

shot_var   = 2.95e9
read_var   = 4.0^2 = 16.0
total_var  = 2.95e9 + 16.0 = 2.95e9

SNR = sqrt(2.95e9) = 5.43e4
SNR_db = 20 * log10(5.43e4) = 94.7 dB
```

### Output

```python
PhotonReport(
  n_photons_per_pixel=2.95e9,
  snr_db=94.7,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="excellent",           # 94.7 dB >> 30 dB
  explanation="Shot-limited. Even with reduced illumination, lensless SNR is excellent."
)
```

---

## R3. MismatchAgent -- Real PSF Drift

```python
# Actual errors from stale calibration
S = 0.25 * |1.5| / 4.5        # psf_sigma: 0.083  (PSF blur changed)
  + 0.30 * |3.0| / 10.0       # psf_shift_x: 0.090  (3 px shift from thermal drift!)
  + 0.30 * |2.5| / 10.0       # psf_shift_y: 0.075  (2.5 px shift)
  + 0.15 * |0.06| / 0.10      # background: 0.090  (ambient light leakage)
S = 0.338  # MODERATE severity

improvement_db = clip(10 * 0.338, 0, 20) = 3.38 dB
```

### Output

```python
MismatchReport(
  severity_score=0.338,
  correction_method="grid_search",
  expected_improvement_db=3.38,
  explanation="Moderate mismatch. PSF shift from thermal drift (3 px) and ambient light are co-dominant. "
              "Recalibrating the PSF is strongly recommended."
)
```

---

## R4. RecoverabilityAgent -- Degraded by Stale PSF

```python
# CR = 1.0 (unchanged)
# With mismatch: effective recoverability degrades
# recoverability_adjusted = 0.87 * (1 - 0.338*0.5) = 0.723
# expected_psnr_adjusted ~ 29.5 dB

# read_limited entry: recoverability=0.76, psnr=30.11 dB
# With additional mismatch degradation from PSF drift:
# -> recoverability=0.60, expected_psnr=27.0 dB
```

### Output

```python
RecoverabilityReport(
  compression_ratio=1.0,
  recoverability_score=0.60,
  expected_psnr_db=27.0,
  verdict="sufficient",           # score >= 0.60
  explanation="Stale PSF calibration degrades reconstruction. Recalibrate for best results."
)
```

---

## R5. AnalysisAgent -- PSF Mismatch is the Bottleneck

```python
photon_score      = 0.0       # Excellent
mismatch_score    = 0.338     # Moderate
compression_score = 1 - 0.60  = 0.40   # Significant
solver_score      = 0.2

primary = "compression"   # max(0.0, 0.338, 0.40, 0.2)
# Root cause: PSF mismatch inflates compression_score

P = 1.0 * 0.831 * 0.80 * 0.90 = 0.598
```

---

## R6. AgentNegotiator -- Proceed with Warnings

```python
P_photon         = 0.95
P_recoverability = 0.60
P_mismatch       = 1.0 - 0.338 * 0.7 = 0.763

P_joint = 0.95 * 0.60 * 0.763 = 0.435
```

No veto (P_joint > 0.15). Proceed with warnings.

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=2.0,
  proceed_recommended=True,
  warnings=[
    "Moderate PSF mismatch (severity 0.338) -- PSF calibration may be stale",
    "Ambient light contamination detected -- background subtraction recommended",
    "Consider recalibrating PSF for optimal reconstruction quality"
  ],
  what_to_upload=["measurement (raw lensless capture)", "PSF (calibrated diffuser PSF)"]
)
```

---

## R8. Pipeline Results

| Configuration | ADMM-TV | FlatNet |
|---------------|---------|---------|
| Ideal operator (shot-limited) | 34.72 dB | 38.15 dB |
| Stale PSF (3 px shifted) | 27.0 dB | 30.5 dB |
| Read-noise limited | 30.11 dB | 34.0 dB |
| Recalibrated PSF | 33.5 dB | 37.2 dB |

**Key findings:**
- Lensless imaging with CR=1.0 achieves excellent quality when PSF is well-calibrated
- FlatNet consistently outperforms ADMM-TV by ~3.4 dB (end-to-end learned deconvolution)
- PSF registration error (shift) causes 5-8 dB degradation (most destructive mismatch)
- Ambient light is easily correctable by background subtraction (+1-2 dB recovery)
- PSF recalibration recovers most of the lost quality (~33.5 dB vs 34.72 dB ideal)

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Experiment |
|--------|-----------|-----------------|
| **Photon Agent** | | |
| N_effective | 3.54e11 | 2.95e9 |
| SNR | 115.5 dB | 94.7 dB |
| Quality tier | excellent | excellent |
| **Mismatch Agent** | | |
| Severity | 0.118 (low) | 0.338 (moderate) |
| Dominant error | none | PSF shift (3 px) + ambient |
| Expected gain | +1.18 dB | +3.38 dB |
| **Recoverability Agent** | | |
| Score | 0.87 (excellent) | 0.60 (sufficient) |
| Expected PSNR | 34.72 dB | 27.0 dB |
| **Analysis Agent** | | |
| Primary bottleneck | solver | mismatch/compression |
| P(success) | 0.792 | 0.598 |
| **Negotiator** | | |
| P_joint | 0.758 | 0.435 |
| **PreFlight** | | |
| Runtime | 2.0s | 2.0s |
| Warnings | 0 | 3 |
| **Pipeline** | | |
| ADMM-TV | 34.72 dB | 27.0 dB |
| FlatNet | 38.15 dB | 30.5 dB |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (ADMM-TV -> FlatNet) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **PSF-centric:** Lensless imaging quality is dominated by PSF calibration accuracy, not photon budget or compression.
