# Widefield Fluorescence Microscopy Working Process

## End-to-End Pipeline for Widefield Fluorescence Deconvolution

This document traces a complete widefield fluorescence microscopy experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Deconvolve this widefield fluorescence image of HeLa cells.
 Measurement: hela_widefield.tif, NA=0.75, emission=525nm, pixel_size=0.325um."
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "hela_widefield.tif" detected
#   operator_type=OperatorType.linear_operator,
#   files=["hela_widefield.tif"],
#   params={"emission_wavelength_nm": 525, "numerical_aperture": 0.75,
#           "pixel_size_um": 0.325}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> widefield entry
widefield:
  keywords: [fluorescence, widefield, deconvolution, PSF, epifluorescence]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="widefield",
#   confidence=0.92,
#   reasoning="Matched keywords: widefield, fluorescence, deconvolution"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the widefield registry entry:

```python
system = plan_agent.build_imaging_system("widefield")
# ImagingSystem(
#   modality_key="widefield",
#   display_name="Widefield Fluorescence Microscopy",
#   signal_dims={"x": [512, 512], "y": [512, 512]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...5 elements...],
#   default_solver="richardson_lucy"
# )
```

**Widefield Element Chain (5 elements):**

```
Mercury/LED Source --> Excitation Filter --> Objective Lens (20x/0.75 NA) --> Emission Filter --> sCMOS Detector
  throughput=1.0       throughput=0.85       throughput=0.80                  throughput=0.90     throughput=0.82
  noise: none          noise: none           noise: aberration                noise: none         noise: shot+read+fixed_pattern
```

**Cumulative throughput:** `0.85 x 0.80 x 0.90 x 0.82 = 0.502`

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  widefield:
    model_id: "microscopy_fluorescence"
    parameters:
      power_w: 0.001
      wavelength_nm: 525
      na: 0.75
      n_medium: 1.0
      qe: 0.72
      exposure_s: 0.05
  ```

### Computation

```python
# 1. Photon energy
E_photon = h * c / wavelength_nm
         = (6.626e-34 * 3e8) / (525e-9)
         = 3.786e-19 J

# 2. Collection solid angle
solid_angle = (na / n_medium)^2 / (4 * pi)
            = (0.75 / 1.0)^2 / (4 * pi)
            = 0.5625 / 12.566
            = 0.04476

# 3. Raw photon count
N_raw = power_w * qe * solid_angle * exposure_s / E_photon
      = 0.001 * 0.72 * 0.04476 * 0.05 / 3.786e-19
      = 4.25e12 photons

# 4. Apply cumulative throughput
N_effective = N_raw * 0.502 = 2.13e12 photons/pixel

# 5. Noise variances
shot_var   = N_effective                       # Poisson
read_var   = read_noise^2 = 1.6^2 = 2.56      # Gaussian (1.6 e-)
dark_var   = 0                                 # Negligible for short exposure
total_var  = shot_var + read_var + dark_var

# 6. SNR
SNR = N_effective / sqrt(total_var) ~ sqrt(N_effective) = 1.46e6
SNR_db = 20 * log10(SNR) = 123.3 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=2.13e12,
  snr_db=123.3,
  noise_regime=NoiseRegime.shot_limited,    # shot_var/total_var > 0.9
  shot_noise_sigma=1.46e6,
  read_noise_sigma=1.6,
  total_noise_sigma=1.46e6,
  feasible=True,
  quality_tier="excellent",                 # SNR > 30 dB
  throughput_chain=[
    {"Mercury/LED Source": 1.0},
    {"Excitation Filter": 0.85},
    {"Objective Lens": 0.80},
    {"Emission Filter": 0.90},
    {"sCMOS Detector": 0.82}
  ],
  noise_model="poisson",
  explanation="Shot-noise-limited regime. Excellent SNR for deconvolution."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"widefield"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  widefield:
    parameters:
      psf_sigma:   {range: [0.5, 3.0], typical_error: 0.3, weight: 0.40}
      defocus:     {range: [-2.0, 2.0], typical_error: 0.5, weight: 0.30}
      background:  {range: [0.0, 0.15], typical_error: 0.03, weight: 0.15}
      gain:        {range: [0.5, 1.5], typical_error: 0.1, weight: 0.15}
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.40 * |0.3| / 2.5     # psf_sigma: 0.048
  + 0.30 * |0.5| / 4.0     # defocus:   0.038
  + 0.15 * |0.03| / 0.15   # background: 0.030
  + 0.15 * |0.1| / 1.0     # gain:       0.015
S = 0.131  # Low severity (typical lab conditions)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.31 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="widefield",
  mismatch_family="grid_search",
  parameters={
    "psf_sigma":  {"typical_error": 0.3, "range": [0.5, 3.0], "weight": 0.40},
    "defocus":    {"typical_error": 0.5, "range": [-2.0, 2.0], "weight": 0.30},
    "background": {"typical_error": 0.03, "range": [0.0, 0.15], "weight": 0.15},
    "gain":       {"typical_error": 0.1, "range": [0.5, 1.5], "weight": 0.15}
  },
  severity_score=0.131,
  correction_method="grid_search",
  expected_improvement_db=1.31,
  explanation="Low mismatch severity under typical conditions. PSF width is the dominant uncertainty."
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
  widefield:
    signal_prior_class: "tv"
    entries:
      - {cr: 1.0, noise: "shot_limited", solver: "richardson_lucy",
         recoverability: 0.82, expected_psnr_db: 27.3,
         provenance: {dataset_id: "bii_widefield_hela_2023", ...}}
      - {cr: 1.0, noise: "shot_limited", solver: "care_restore_2d",
         recoverability: 0.89, expected_psnr_db: 31.4, ...}
  ```

### Computation

```python
# 1. Compression ratio
CR = prod(y_shape) / prod(x_shape) = (512 * 512) / (512 * 512) = 1.0
# Widefield is NOT compressed: y and x have the same dimensionality.
# The problem is deconvolution (inversion), not compressed sensing.

# 2. Operator diversity
# For convolution, diversity is determined by the PSF bandwidth.
# Gaussian PSF with sigma=2.0 px covers ~6x6 pixel support.
# OTF bandwidth = 1/(2*pi*sigma) ~ 0.08 cycles/px
diversity = 0.6  # Moderate (PSF acts as low-pass filter, not random mask)

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.625

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="richardson_lucy", cr=1.0 (exact match)
#    -> recoverability=0.82, expected_psnr=27.3 dB, confidence=1.0

# 5. Best solver selection
#    care_restore_2d: 31.4 dB > richardson_lucy: 27.3 dB
#    -> recommended: "care_restore_2d" (or "richardson_lucy" as default)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=1.0,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.tv,
  operator_diversity_score=0.6,
  condition_number_proxy=0.625,
  recoverability_score=0.82,
  recoverability_confidence=1.0,
  expected_psnr_db=27.3,
  expected_psnr_uncertainty_db=0.8,
  recommended_solver_family="richardson_lucy",
  verdict="good",                # score >= 0.75
  calibration_table_entry={...},
  explanation="Good recoverability. RL expected 27.3 dB; CARE can reach 31.4 dB on BII benchmark."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(123.3 / 40, 1.0)  = 0.0    # Excellent SNR
mismatch_score    = 0.131                       = 0.131  # Low mismatch
compression_score = 1 - 0.82                    = 0.18   # Good recoverability
solver_score      = 0.2                         = 0.2    # Default placeholder

# Primary bottleneck
primary = "solver"  # max(0.0, 0.131, 0.18, 0.2) = solver

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.131*0.5) * (1 - 0.18*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.935 * 0.91 * 0.90
  = 0.766
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="solver",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.131, compression=0.18, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Consider CARE U-Net for +4.1 dB over Richardson-Lucy",
      priority="medium",
      expected_gain_db=4.1
    ),
    Suggestion(
      text="Measure the PSF with fluorescent beads for better operator accuracy",
      priority="low",
      expected_gain_db=1.0
    )
  ],
  overall_verdict="good",           # P >= 0.70
  probability_of_success=0.766,
  explanation="System is well-configured. Solver choice is the primary bottleneck. "
              "CARE can provide significant improvement over RL."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="good" | No veto |
| Severe mismatch without correction | severity=0.131 < 0.7 | No veto |
| All marginal | All excellent/good | No veto |
| Joint probability floor | P=0.766 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95   # tier_prob["excellent"]
P_recoverability = 0.82   # recoverability_score
P_mismatch       = 1.0 - 0.131 * 0.7 = 0.908

P_joint = 0.95 * 0.82 * 0.908 = 0.707
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.707
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 512 * 512 = 262,144
dim_factor   = total_pixels / (256 * 256) = 4.0
solver_complexity = 1.0  # RL (iterative, CPU)
cr_factor    = max(1.0, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 4.0 * 1.0 * 0.125 = 1.0 seconds
# RL 30 iterations on 512x512 ~ 1-3 seconds on CPU
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="widefield", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=1.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=["measurement (2D widefield fluorescence image, TIFF/PNG)"]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Widefield forward model: y = PSF ** x + n
#
# The PSF is a 2D Gaussian approximating the Airy pattern:
#   PSF(r) = exp(-r^2 / (2 * sigma^2))
#   sigma = 0.21 * lambda / NA = 0.21 * 525 / 0.75 = 147 nm
#   In pixels: sigma_px = 147 / (pixel_size * 1000) = 147 / 325 ~ 0.45 px
#   (or use registry value psf_sigma_px = 2.0 for degraded widefield)
#
# Input:  x = (512, 512) fluorescence distribution
# Output: y = (512, 512) blurred image (same size)

class WidefieldOperator(PhysicsOperator):
    def forward(self, x):
        """y = PSF ** x (2D convolution)"""
        return fftconvolve(x, self.psf, mode='same')

    def adjoint(self, y):
        """x_hat = PSF_flipped ** y (correlation)"""
        return fftconvolve(y, self.psf[::-1, ::-1], mode='same')

    def check_adjoint(self):
        """Verify <Ax, y> ~ <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided measurement:
y = load_tiff("hela_widefield.tif")    # (512, 512)

# If simulating:
x_true = load_ground_truth()            # (512, 512) from BII dataset
y = operator.forward(x_true)            # (512, 512) blurred
y = np.random.poisson(y * gain) / gain  # Shot noise (Poisson)
y += np.random.randn(*y.shape) * 0.01   # Read noise (Gaussian, sigma=0.01)
```

### Step 9c: Reconstruction with Richardson-Lucy

```python
from pwm_core.recon import run_richardson_lucy

x_hat, info = run_richardson_lucy(
    y=y,                      # (512, 512) blurred measurement
    physics=physics,          # Contains PSF
    cfg={"iters": 30}         # 30 RL iterations
)
# x_hat shape: (512, 512) -- deconvolved image
# Expected PSNR: ~27.3 dB (BII HeLa benchmark)
```

**Richardson-Lucy algorithm:**

```python
# RL update rule (multiplicative):
# x_{k+1} = x_k * (PSF_flipped ** (y / (PSF ** x_k)))
#
# Iteration 0: x_0 = y (initialize with measurement)
# For k = 0..29:
#   prediction = convolve(x_k, PSF)
#   ratio = y / (prediction + eps)
#   correction = correlate(ratio, PSF)
#   x_{k+1} = x_k * correction
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| Richardson-Lucy | Traditional | 27.3 dB | No | `run_richardson_lucy(y, physics, {"iters": 30})` |
| CARE U-Net | Deep Learning | 31.4 dB | Yes | `care_restore_2d(y, model_path="weights/care_widefield.pth")` |
| Wiener Filter | Traditional | 25.5 dB | No | `wiener_deconv(y, psf, snr=100)` |

### Step 9d: Metrics

```python
# PSNR
psnr = 10 * log10(max_val^2 / mse(x_hat, x_true))
# ~ 27.3 dB (Richardson-Lucy, 30 iterations)

# SSIM (structural similarity)
ssim_val = ssim(x_hat, x_true)

# Resolution metric: FWHM of reconstructed point source
# Deconvolved FWHM should approach diffraction limit: 0.61 * lambda / NA
# = 0.61 * 525 / 0.75 = 427 nm
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
|   +-- y.npy              # Measurement (512, 512) + SHA256 hash
|   +-- x_hat.npy          # Reconstruction (512, 512) + SHA256 hash
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
|   +-- psf.npy            # PSF used (15, 15) + SHA256 hash
+-- metrics.json           # PSNR, SSIM, resolution metrics
+-- operator.json          # Operator parameters (PSF sigma, NA, wavelength)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an **idealized** widefield pipeline with optimal lab conditions (SNR 123.3 dB, mismatch severity 0.131). In practice, real widefield systems suffer from unknown PSF shape, autofluorescence background, stage drift causing defocus, and photobleaching.

This section traces the **same pipeline** with realistic parameters drawn from actual benchmark experiments.

---

## Real Experiment: User Prompt

```
"I have a widefield fluorescence image of fixed HeLa cells. The focus might
 be slightly off and there is significant autofluorescence background.
 Please deconvolve. Measurement: hela_raw.tif, NA=0.75, emission=525nm."
```

**Key difference:** The user mentions potential defocus and autofluorescence, indicating operator mismatch.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,                    # No explicit "calibrate" request
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["hela_raw.tif"],
#   params={"numerical_aperture": 0.75, "emission_wavelength_nm": 525}
# )
```

---

## R2. PhotonAgent -- Moderate Lab Conditions

### Real detector parameters

```yaml
# Real lab: moderate exposure, autofluorescence adds background
widefield_lab:
  power_w: 0.0005         # 0.5 mW (reduced to limit bleaching)
  wavelength_nm: 525
  na: 0.75
  n_medium: 1.0
  qe: 0.72
  exposure_s: 0.03        # 30 ms exposure
  read_noise_e: 2.5       # Older sCMOS
```

### Computation

```python
# Raw photon count
N_raw = 0.0005 * 0.72 * 0.04476 * 0.03 / 3.786e-19 = 1.28e12

# Apply cumulative throughput (0.502)
N_effective = 1.28e12 * 0.502 = 6.41e11 photons/pixel

# Noise variances
shot_var   = 6.41e11
read_var   = 2.5^2 = 6.25
total_var  = 6.41e11 + 6.25

# SNR
SNR = 6.41e11 / sqrt(6.41e11) = 8.01e5
SNR_db = 20 * log10(8.01e5) = 118.1 dB
```

### Output

```python
PhotonReport(
  n_photons_per_pixel=6.41e11,
  snr_db=118.1,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="excellent",
  explanation="Shot-limited. Widefield collects many photons due to full-field illumination."
)
```

---

## R3. MismatchAgent -- Defocus + Background

```python
# Actual errors (not typical)
psi_true = {
    "psf_sigma": +0.6,    # PSF 30% wider than assumed (coverslip thickness)
    "defocus":   +1.2,     # 1.2 um defocus from stage drift
    "background": 0.08,    # 8% autofluorescence background
    "gain":       0.05,    # Minor gain offset
}

# Severity computation
S = 0.40 * |0.6| / 2.5     # psf_sigma: 0.096
  + 0.30 * |1.2| / 4.0     # defocus:   0.090
  + 0.15 * |0.08| / 0.15   # background: 0.080
  + 0.15 * |0.05| / 1.0    # gain:       0.008
S = 0.274  # Moderate severity

# Expected improvement from correction
improvement_db = clip(10 * 0.274, 0, 20) = 2.74 dB
```

### Output

```python
MismatchReport(
  severity_score=0.274,
  correction_method="grid_search",
  expected_improvement_db=2.74,
  explanation="Moderate mismatch. Defocus (1.2 um) and PSF width error are primary sources."
)
```

---

## R4. RecoverabilityAgent -- Degraded by Mismatch

```python
# CR unchanged at 1.0
# With mismatch, RL performance degrades:
# Calibration lookup -> recoverability=0.61, expected_psnr=22.1 dB

RecoverabilityReport(
  compression_ratio=1.0,
  recoverability_score=0.61,
  expected_psnr_db=22.1,
  verdict="marginal",
  explanation="PSF mismatch degrades deconvolution. Consider measuring the PSF."
)
```

---

## R5. AnalysisAgent -- Mismatch is the Bottleneck

```python
photon_score      = 0.0
mismatch_score    = 0.274
compression_score = 1 - 0.61 = 0.39
solver_score      = 0.2

primary = "compression"  # 0.39 (inflated by mismatch)

P = 1.0 * 0.863 * 0.805 * 0.90 = 0.625

SystemAnalysis(
  primary_bottleneck="mismatch",
  probability_of_success=0.625,
  suggestions=[
    Suggestion(text="Measure PSF with sub-diffraction beads", priority="high",
               expected_gain_db=2.5),
    Suggestion(text="Subtract autofluorescence background before deconvolution",
               priority="high", expected_gain_db=1.5),
    Suggestion(text="Use CARE (trained on matched pairs) for robustness to PSF error",
               priority="medium", expected_gain_db=4.0)
  ],
  overall_verdict="marginal"
)
```

---

## R6. AgentNegotiator -- Proceed with Warnings

```python
P_photon         = 0.95
P_recoverability = 0.61
P_mismatch       = 1.0 - 0.274 * 0.7 = 0.808

P_joint = 0.95 * 0.61 * 0.808 = 0.468

NegotiationResult(
  vetoes=[],
  proceed=True,
  probability_of_success=0.468
)
```

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=1.5,
  proceed_recommended=True,
  warnings=[
    "Mismatch severity 0.274 -- PSF and defocus errors may limit deconvolution",
    "Recoverability marginal (0.61) -- consider CARE for better results"
  ],
  what_to_upload=["measurement (2D widefield fluorescence image, TIFF)"]
)
```

---

## R8. Pipeline Runner -- With Background Subtraction

### Step R8a: Deconvolve with Wrong PSF

```python
# Uses theoretical Gaussian PSF (sigma=2.0 px, no defocus)
operator_wrong = WidefieldOperator(psf=gaussian_psf(sigma=2.0))
x_wrong, _ = run_richardson_lucy(y_lab, operator_wrong, {"iters": 30})
# PSNR = 22.1 dB  <-- degraded by PSF mismatch
```

### Step R8b: Background Subtraction + Corrected PSF

```python
# Estimate and subtract background
background = estimate_background(y_lab, method="rolling_ball", radius=50)
y_corrected = y_lab - background

# Use wider PSF to account for defocus
psf_corrected = gaussian_psf(sigma=2.6)  # sigma + 0.6 correction
operator_corrected = WidefieldOperator(psf=psf_corrected)

x_corrected, _ = run_richardson_lucy(y_corrected, operator_corrected, {"iters": 30})
# PSNR = 24.8 dB  <-- +2.7 dB improvement
```

### Step R8c: CARE Deep Learning

```python
from pwm_core.recon.care_unet import care_restore_2d

x_care = care_restore_2d(
    y=y_corrected,
    model_path="weights/care_widefield.pth",
    device="cuda"
)
# PSNR = 31.4 dB  <-- best result, robust to PSF mismatch
```

### Step R8d: Final Comparison

| Configuration | RL (30 iter) | CARE | Notes |
|---------------|-------------|------|-------|
| Wrong PSF, no background sub | **22.1 dB** | -- | Baseline |
| Corrected PSF + background sub | **24.8 dB** | -- | +2.7 dB |
| CARE (deep learning) | -- | **31.4 dB** | Best quality |
| Reference (ideal conditions) | **27.3 dB** | **31.4 dB** | Upper bound |

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Experiment |
|--------|-----------|-----------------|
| **Photon Agent** | | |
| N_effective | 2.13e12 | 6.41e11 |
| SNR | 123.3 dB | 118.1 dB |
| Quality tier | excellent | excellent |
| **Mismatch Agent** | | |
| Severity | 0.131 (low) | 0.274 (moderate) |
| Dominant error | psf_sigma | defocus + psf_sigma |
| Expected gain | +1.31 dB | +2.74 dB |
| **Recoverability Agent** | | |
| Score | 0.82 (good) | 0.61 (marginal) |
| Expected PSNR | 27.3 dB | 22.1 dB |
| Verdict | good | **marginal** |
| **Analysis Agent** | | |
| Primary bottleneck | solver | **mismatch** |
| P(success) | 0.766 | 0.625 |
| **Negotiator** | | |
| P_joint | 0.707 | 0.468 |
| **PreFlight** | | |
| Runtime | 1.0s | 1.5s |
| Warnings | 0 | 2 |
| **Pipeline** | | |
| RL PSNR | 27.3 dB | 22.1 -> 24.8 dB |
| CARE PSNR | 31.4 dB | **31.4 dB** |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (RL -> CARE -> Wiener) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Adaptive:** The same pipeline automatically recommends PSF measurement and background subtraction when mismatch severity warrants it.
