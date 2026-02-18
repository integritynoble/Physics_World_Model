# FPM Working Process

## End-to-End Pipeline for Fourier Ptychographic Microscopy

This document traces a complete FPM experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct a high-resolution complex image from FPM LED array data.
 Images: led_images.npy, LED positions: led_pos.npy,
 15x15 LED grid (225 images), 4x/0.1 NA objective, wavelength 530 nm."
```

---

## 2. PlanAgent — Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "led_images.npy" detected
#   operator_type=OperatorType.nonlinear_operator,
#   files=["led_images.npy", "led_pos.npy"],
#   params={"n_leds": 225, "led_grid": [15, 15],
#           "objective_na": 0.1, "wavelength_nm": 530,
#           "magnification": 4}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> fpm entry
fpm:
  keywords: [FPM, fourier_ptychography, synthetic_aperture, LED_array, high_resolution]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="fpm",
#   confidence=0.95,
#   reasoning="Matched keywords: FPM, fourier_ptychography, LED_array"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the FPM registry entry:

```python
system = plan_agent.build_imaging_system("fpm")
# ImagingSystem(
#   modality_key="fpm",
#   display_name="Fourier Ptychographic Microscopy",
#   signal_dims={"x": [1024, 1024], "y": [256, 256, 225]},
#   forward_model_type=ForwardModelType.nonlinear_operator,
#   elements=[...4 elements...],
#   default_solver="sequential_phase_retrieval"
# )
```

**FPM Element Chain (4 elements):**

```
LED Array (15x15) --> Sample (Thin Specimen) --> Objective Lens (4x/0.1 NA) --> CMOS Camera
  throughput=0.90      throughput=0.80            throughput=0.85               throughput=0.78
  noise: alignment     noise: none               noise: aberration             noise: shot+read+quant
  225 LEDs, 530 nm     max_phase=2 rad           synthetic NA=0.5             6.5 um pixels
  4mm pitch, 80mm      max_absorption=0.5         5x resolution gain           2 e- read noise
  distance                                                                      12-bit ADC
```

**Cumulative throughput:** `0.90 x 0.80 x 0.85 x 0.78 = 0.478`

**Forward model equation:**
```
y_j = |F^{-1}{ P(k - k_j) * O(k) }|^2,   j = 1..225

O(k): high-resolution object spectrum (1024 x 1024, complex)
P(k): pupil function (circular aperture, NA = 0.1)
k_j:  illumination wave vector for LED j
y_j:  low-resolution intensity image from LED j (256 x 256)
```

The nonlinearity arises from the squared magnitude `|.|^2` which discards phase information. Each LED captures a different Fourier sub-aperture of the object, and the collection of 225 images tiles the Fourier plane to achieve a synthetic NA of 0.5 (5x the native objective NA).

---

## 3. PhotonAgent — SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  fpm:
    model_id: "led_array"
    parameters:
      power_w: 0.05
      wavelength_nm: 632
      na: 0.1
      qe: 0.70
      exposure_s: 0.05
  ```

### Computation

```python
# 1. Photon energy at 530 nm (center wavelength from modalities.yaml)
E_photon = h * c / wavelength_nm
#        = 6.626e-34 * 3e8 / (530e-9)
#        = 3.75e-19 J

# 2. Collection solid angle (low NA = 0.1 in air)
solid_angle = (na / n_medium)^2 / (4 * pi)
#           = (0.1 / 1.0)^2 / (4 * pi)
#           = 0.01 / 12.566
#           = 7.96e-4

# 3. Raw photon count per LED exposure
# Power per LED: 0.05 W (total array) / 225 (one LED at a time) = 2.22e-4 W
P_per_led = 0.05 / 225 = 2.22e-4 W

N_raw = P_per_led * qe * solid_angle * exposure_s / E_photon
#     = 2.22e-4 * 0.70 * 7.96e-4 * 0.05 / 3.75e-19
#     = 6.19e-9 / 3.75e-19
#     = 1.65e10 photons

# 4. Per-pixel (256 x 256 = 65536 low-res pixels)
N_per_pixel = 1.65e10 / 65536 = 2.52e5 photons/pixel

# 5. Apply cumulative throughput
N_effective = 2.52e5 * 0.478 = 1.20e5 photons/pixel

# 6. Noise variances
shot_var   = N_effective = 1.20e5         # Poisson
read_var   = read_noise^2 = 4.0           # 2 e- read noise
dark_var   = 0                             # Negligible
total_var  = 1.20e5 + 4.0

# 7. SNR
SNR = N_effective / sqrt(total_var) = sqrt(1.20e5) = 346
SNR_db = 20 * log10(346) = 50.8 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=1.20e5,
  snr_db=50.8,
  noise_regime=NoiseRegime.shot_limited,     # shot_var >> read_var
  shot_noise_sigma=346,
  read_noise_sigma=2.0,
  total_noise_sigma=346,
  feasible=True,
  quality_tier="excellent",                  # SNR > 30 dB
  throughput_chain=[
    {"LED Array (Variable Angle Illumination)": 0.90},
    {"Sample (Thin Specimen)": 0.80},
    {"Objective Lens (4x / 0.1 NA)": 0.85},
    {"CMOS Camera": 0.78}
  ],
  noise_model="poisson",
  explanation="Shot-noise-limited regime. 120k photons/pixel per LED exposure "
              "provides excellent SNR for iterative phase retrieval."
)
```

---

## 4. MismatchAgent — Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"fpm"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  fpm:
    parameters:
      led_position_xy:
        range: [-2.0, 2.0]
        typical_error: 0.3
        unit: "mm"
        description: "LED position error from PCB manufacturing tolerance"
      objective_na_error:
        range: [-0.05, 0.05]
        typical_error: 0.01
        unit: "unitless"
        description: "Objective NA calibration error affecting pupil function"
      defocus:
        range: [-5.0, 5.0]
        typical_error: 1.0
        unit: "um"
        description: "Sample defocus from stage drift"
    severity_weights:
      led_position_xy: 0.45
      objective_na_error: 0.25
      defocus: 0.30
    correction_method: "gradient_descent"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.45 * |0.3| / 4.0       # led_position:    0.0338
  + 0.25 * |0.01| / 0.10     # na_error:        0.025
  + 0.30 * |1.0| / 10.0      # defocus:         0.030
S = 0.089  # Low severity (well-calibrated system)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 0.89 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="fpm",
  mismatch_family="gradient_descent",
  parameters={
    "led_position_xy":   {"typical_error": 0.3, "range": [-2, 2], "weight": 0.45},
    "objective_na_error": {"typical_error": 0.01, "range": [-0.05, 0.05], "weight": 0.25},
    "defocus":           {"typical_error": 1.0, "range": [-5, 5], "weight": 0.30}
  },
  severity_score=0.089,
  correction_method="gradient_descent",
  expected_improvement_db=0.89,
  explanation="Low mismatch severity. LED positions well-characterized by PCB design. "
              "Gradient descent can jointly optimize positions with object recovery."
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
  fpm:
    signal_prior_class: "deep_prior"
    entries:
      - {cr: 0.055, noise: "shot_limited", solver: "fpm_iterative",
         recoverability: 0.85, expected_psnr_db: 34.2,
         provenance: {dataset_id: "fpm_hela_led_array_2023", ...}}
      - {cr: 0.055, noise: "detector_limited", solver: "fpm_iterative",
         recoverability: 0.72, expected_psnr_db: 30.5, ...}
      - {cr: 0.055, noise: "shot_limited", solver: "deep_fpm",
         recoverability: 0.91, expected_psnr_db: 38.1, ...}
  ```

### Computation

```python
# 1. Compression ratio
#    x = (1024, 1024) = 1,048,576 complex pixels (HR object)
#    y = (256, 256, 225) = 14,745,600 intensity measurements
CR = prod(x_shape) / prod(y_shape) = 1048576 / 14745600 = 0.071
# NOTE: CR < 1 means more measurements than unknowns (oversampled)
# But each measurement is intensity-only (phase lost), so
# the effective CR accounts for the phase retrieval challenge
# Calibration table uses CR = 0.055 (accounts for complex output: 2x)
# Effective: 2 * 1048576 / 14745600 = 0.142 (amplitude + phase)

# 2. Operator diversity
#    225 LED angles tile the Fourier plane with overlap
#    Each low-NA image captures a different sub-aperture
#    High angular diversity -> well-conditioned phase retrieval
diversity = 0.8  # High — 225 illumination angles with Fourier overlap

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.556

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="fpm_iterative", cr=0.055
#    -> recoverability=0.85, expected_psnr=34.2 dB

# 5. Best solver selection
#    deep_fpm: 38.1 dB > fpm_iterative: 34.2 dB
#    -> recommended: "fpm_iterative" (default, established)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.055,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.deep_prior,
  operator_diversity_score=0.8,
  condition_number_proxy=0.556,
  recoverability_score=0.85,
  recoverability_confidence=1.0,
  expected_psnr_db=34.2,
  expected_psnr_uncertainty_db=1.0,
  recommended_solver_family="fpm_iterative",
  verdict="excellent",                    # score >= 0.85
  calibration_table_entry={...},
  explanation="Excellent recoverability. 225 LED images provide sufficient "
              "Fourier coverage for 5x resolution enhancement. "
              "Sequential phase retrieval expected 34.2 dB on HeLa benchmark."
)
```

---

## 6. AnalysisAgent — Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(50.8 / 40, 1.0)   = 0.0     # Excellent SNR
mismatch_score    = 0.089                        = 0.089   # Low mismatch
compression_score = 1 - 0.85                     = 0.15    # Excellent recoverability
solver_score      = 0.15                         = 0.15    # Phase retrieval convergence

# Primary bottleneck
primary = "compression"  # max(0.0, 0.089, 0.15, 0.15) = tied; compression chosen

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.089*0.5) * (1 - 0.15*0.5) * (1 - 0.15*0.5)
  = 1.0 * 0.956 * 0.925 * 0.925
  = 0.818
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="compression",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.089, compression=0.15, solver=0.15
  ),
  suggestions=[
    Suggestion(
      text="Deep FPM network can improve +3.9 dB over iterative retrieval",
      priority="medium",
      expected_gain_db=3.9
    ),
    Suggestion(
      text="Gradient descent with joint pupil recovery for aberration correction",
      priority="medium",
      expected_gain_db=1.5
    )
  ],
  overall_verdict="excellent",           # P >= 0.80
  probability_of_success=0.818,
  explanation="Well-configured FPM system. Phase retrieval from 225 angles is "
              "well-conditioned. Solver convergence is the practical limitation."
)
```

---

## 7. AgentNegotiator — Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="excellent" | No veto |
| Severe mismatch without correction | severity=0.089 < 0.7 | No veto |
| All marginal | All excellent | No veto |
| Joint probability floor | P=0.818 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.85    # recoverability_score
P_mismatch       = 1.0 - 0.089 * 0.7 = 0.938

P_joint = 0.95 * 0.85 * 0.938 = 0.757
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.757
)
```

---

## 8. PreFlightReportBuilder — Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 1024 * 1024 = 1048576          # HR output
n_leds = 225
dim_factor   = total_pixels / (256 * 256) = 16.0
solver_complexity = 3.0     # Iterative phase retrieval (many FFTs)
n_iters = 30
cr_factor    = max(0.055, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 16.0 * 3.0 * 0.125 * n_iters / 10 = 36.0 seconds
# Note: Each iteration loops over 225 LEDs, performing 2 FFTs each
# Actual: 30 iters * 225 LEDs * 2 FFTs * (256^2 FFT) ~ 45s on CPU
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="fpm", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=45.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=[
    "Stack of low-resolution images for each LED (N_LEDs x H_low x W_low)",
    "LED array positions or illumination angles",
    "Objective NA and wavelength"
  ]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# FPM forward model: y_j = |IFFT{ P(k-k_j) * O(k) }|^2
#
# O(k): object spectrum in Fourier space (hr_size x hr_size, complex)
# P(k): pupil function (circular, radius ~ NA * k_max)
# k_j:  wave vector shift from LED j
#
# Each LED at angle theta_j shifts the object spectrum by k_j = (2*pi/lambda)*sin(theta_j)
# The low-NA objective acts as a low-pass filter (pupil)
# The camera measures intensity only: |.|^2
#
# Parameters:
#   led_positions:  (225, 2) k-space offsets from LED angles
#   pupil:          (lr_size, lr_size) binary circular mask
#   hr_size:        1024 (high-resolution output)
#   lr_size:        256 (low-resolution input)
#
# Input:  x = (1024, 1024) complex high-resolution object
# Output: y = (225, 256, 256) low-resolution intensity images

class FPMOperator(PhysicsOperator):
    def forward(self, x):
        """y_j = |IFFT(P(k-k_j) * O(k))|^2"""
        O = fftshift(fft2(x))              # Object spectrum
        y = np.zeros((n_leds, lr_size, lr_size))
        for j in range(n_leds):
            ky, kx = led_positions[j]
            center = (hr_size//2 + ky, hr_size//2 + kx)
            O_crop = crop_spectrum(O, center, lr_size)
            psi = O_crop * pupil            # Apply pupil
            img = ifft2(ifftshift(psi))     # Inverse FFT
            y[j] = np.abs(img)**2           # Intensity (phase lost)
        return y

    def adjoint(self, y):
        """Approximate adjoint: sum phase-corrected sub-apertures"""
        O_hat = np.zeros((hr_size, hr_size), dtype=complex)
        for j in range(n_leds):
            amp = np.sqrt(np.maximum(y[j], 0))
            psi = fftshift(fft2(amp))
            center = (hr_size//2 + led_positions[j,0], hr_size//2 + led_positions[j,1])
            place_spectrum(O_hat, psi * pupil, center)
        return ifft2(ifftshift(O_hat))
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided led_images.npy:
y = np.load("led_images.npy")     # (225, 256, 256)
led_pos = np.load("led_pos.npy")  # (225, 2) k-space positions

# If simulating:
# Ground truth: complex object (amplitude + phase)
amp_true = np.zeros((256, 256))       # Circular features
phase_true = np.zeros((256, 256))     # Gaussian phase bumps

x_true = amp_true * np.exp(1j * phase_true)

# LED grid in k-space (5x5 for benchmark, 15x15 for full)
led_positions = []
for iy in range(5):
    for ix in range(5):
        ky = (iy - 2) * led_spacing
        kx = (ix - 2) * led_spacing
        led_positions.append([ky, kx])

# Forward model
O_true = fftshift(fft2(x_true))
for j in range(25):
    O_crop = crop_spectrum(O_true, center_j, lr_size)
    psi = O_crop * pupil
    img = ifft2(ifftshift(psi))
    lr_images[j] = np.abs(img)**2 + noise
```

### Step 9c: Reconstruction with Sequential Phase Retrieval

```python
from pwm_core.recon.fpm_solver import sequential_phase_retrieval, gradient_descent_fpm

# Algorithm 1: Sequential Phase Retrieval (Zheng et al. 2013)
recon_seq = sequential_phase_retrieval(
    lr_images=y,                    # (225, 256, 256) intensity images
    led_positions=led_pos,          # (225, 2) k-space offsets
    hr_size=1024,                   # High-resolution output size
    lr_size=256,                    # Low-resolution input size
    pupil=pupil,                    # (256, 256) circular aperture
    n_iters=30                      # Iterations over all LEDs
)
# Iterative Fourier stitching:
#   1. Crop current estimate of sub-aperture spectrum
#   2. Apply pupil, IFFT to image domain
#   3. Replace magnitude with measured sqrt(y_j), keep estimated phase
#   4. FFT back, update object spectrum within pupil region
# Output: (1024, 1024) complex object
# Expected PSNR: ~34.2 dB (amplitude) on HeLa LED benchmark

# Algorithm 2: Gradient Descent with Joint Pupil Recovery
recon_gd = gradient_descent_fpm(
    lr_images=y,
    led_positions=led_pos,
    hr_size=1024,
    lr_size=256,
    pupil=pupil,
    n_iters=50,                     # More iterations
    step_size=1.0,
    pupil_update=True               # Jointly optimize pupil function
)
# Wirtinger gradient descent on amplitude-based loss
# Joint object + pupil optimization recovers aberrations
# Expected PSNR: ~38.1 dB with deep FPM approach
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| Sequential PR | Traditional | 34.2 dB | No | `sequential_phase_retrieval(y, led_pos, hr, lr, pupil)` |
| Gradient Descent | Traditional | 36.0 dB | No | `gradient_descent_fpm(y, led_pos, hr, lr, pupil, n_iters=50)` |
| Deep FPM | Deep Learning | 38.1 dB | Yes | `deep_fpm(y, led_pos, model_path=...)` |

### Step 9d: Metrics

```python
# Amplitude PSNR (using max_val=1.0)
amp_recon = np.abs(recon)
psnr_amp = 10 * log10(1.0 / mse(amp_recon, amp_true))
# ~34.2 dB (sequential), ~38.1 dB (deep FPM)

# Phase accuracy (mean absolute phase error)
phase_recon = np.angle(recon)
mae_phase = mean(|phase_recon - phase_true|)

# SSIM on amplitude
ssim_amp = compute_ssim(amp_recon, amp_true)

# FPM-specific metrics:
# Resolution gain: achieved resolution / diffraction limit at native NA
# Native resolution = lambda / (2 * NA) = 530 / (2 * 0.1) = 2650 nm
# Synthetic resolution = lambda / (2 * NA_synth) = 530 / (2 * 0.5) = 530 nm
# Resolution gain = 2650 / 530 = 5x

# Fourier ring correlation (FRC) for resolution assessment
frc = compute_frc(recon, reference)
resolution_px = frc_cutoff(frc, threshold=1/7)

# Phase quantitativeness (for quantitative phase imaging)
# Mean phase bias and phase noise level
phase_std = std(phase_recon[background_mask])
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
|   +-- y.npy              # LED images (225, 256, 256) + SHA256 hash
|   +-- x_hat_amp.npy      # Recovered amplitude (1024, 1024) + SHA256 hash
|   +-- x_hat_phase.npy    # Recovered phase (1024, 1024) + SHA256 hash
|   +-- x_true.npy         # Ground truth complex (if available) + SHA256 hash
|   +-- pupil_hat.npy      # Recovered pupil (if joint estimation) + SHA256 hash
+-- metrics.json           # PSNR(amp), MAE(phase), SSIM, FRC, resolution
+-- operator.json          # Operator params (LED positions, pupil, NA, wavelength)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized FPM pipeline with well-calibrated LED positions, correct pupil model, and negligible noise. In practice, real FPM systems face LED position calibration errors (PCB manufacturing tolerances), pupil aberrations (objective lens imperfections), sample defocus (axial stage drift), and LED intensity variations.

---

## Real Experiment: User Prompt

```
"FPM imaging of unstained tissue section. The LED array positions might
 not be perfectly calibrated — we used a standard PCB with ~0.3 mm
 tolerance. Some edge LEDs may be dimmer. Also the sample may have
 drifted slightly during acquisition.
 Images: tissue_fpm.npy, LED positions: led_nominal.npy, 15x15 array."
```

**Key difference:** LED position errors, sample drift, and LED intensity non-uniformity.

---

## R1. PlanAgent — Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.operator_correction,   # "positions not perfectly calibrated"
#   has_measured_y=True,
#   operator_type=OperatorType.nonlinear_operator,
#   files=["tissue_fpm.npy", "led_nominal.npy"],
#   params={"n_leds": 225, "led_grid": [15, 15]}
# )
```

---

## R2. PhotonAgent — Non-Uniform LED Illumination

```python
# Edge LEDs are dimmer due to oblique angle and distance
# Center LED: 100% power, edge LEDs: ~40% power
# Effective photon count varies by LED position

N_center = 1.20e5 photons/pixel   # Center LED
N_edge   = 4.80e4 photons/pixel   # Edge LED (40%)
N_average = 8.4e4 photons/pixel   # Weighted average

SNR_avg = sqrt(N_average) = 290
SNR_avg_db = 20 * log10(290) = 49.2 dB

PhotonReport(
  n_photons_per_pixel=8.4e4,
  snr_db=49.2,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="excellent",
  explanation="Shot-limited. Edge LEDs ~40% dimmer than center; weighted "
              "average SNR 49.2 dB. LED intensity calibration recommended."
)
```

---

## R3. MismatchAgent — LED Position Errors + Defocus

```python
# Actual errors
psi_true = {
    "led_position_xy": 0.8,      # 0.8 mm (2.7x typical)
    "objective_na_error": 0.02,   # 2x typical
    "defocus": 3.0,               # 3 um stage drift during 225 exposures
}

# Severity
S = 0.45 * |0.8| / 4.0      # led_position:    0.090
  + 0.25 * |0.02| / 0.10    # na_error:        0.050
  + 0.30 * |3.0| / 10.0     # defocus:         0.090
S = 0.230  # Moderate severity

improvement_db = clip(10 * 0.230, 0, 20) = 2.30 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  severity_score=0.230,
  correction_method="gradient_descent",
  expected_improvement_db=2.30,
  explanation="Moderate mismatch. LED position error (0.8 mm) and sample defocus "
              "(3 um) compound. Gradient descent can jointly optimize positions, "
              "pupil, and object."
)
```

---

## R4. RecoverabilityAgent — Detector-Limited

```python
# With LED position errors, effective overlap is reduced
# Calibration table: noise="detector_limited", solver="fpm_iterative"
# -> recoverability=0.72, expected_psnr=30.5 dB

RecoverabilityReport(
  recoverability_score=0.72,
  expected_psnr_db=30.5,
  verdict="good",
  explanation="LED position errors reduce Fourier coverage overlap, degrading "
              "phase retrieval convergence. Joint calibration recommended."
)
```

---

## R5. AnalysisAgent — Mismatch is Bottleneck

```python
photon_score      = 1 - min(49.2 / 40, 1.0)   = 0.0
mismatch_score    = 0.230
compression_score = 1 - 0.72                     = 0.28
solver_score      = 0.15

primary = "compression"  # max(0.0, 0.230, 0.28, 0.15)
# Root cause: mismatch inflates compression score

P = (1-0.0*0.5) * (1-0.230*0.5) * (1-0.28*0.5) * (1-0.15*0.5)
  = 1.0 * 0.885 * 0.86 * 0.925
  = 0.704
```

```python
SystemAnalysis(
  primary_bottleneck="mismatch",
  probability_of_success=0.704,
  overall_verdict="good",
  suggestions=[
    Suggestion(text="Apply gradient descent with LED position refinement", priority="high"),
    Suggestion(text="Enable joint pupil recovery for aberration correction", priority="high"),
    Suggestion(text="Apply LED intensity calibration from background images", priority="medium"),
  ]
)
```

---

## R6. AgentNegotiator — Proceed

```python
P_joint = 0.95 * 0.72 * (1 - 0.230*0.7) = 0.95 * 0.72 * 0.839 = 0.574

NegotiationResult(
  vetoes=[],
  proceed=True,
  probability_of_success=0.574
)
```

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=320.0,       # Gradient descent with 50 iters + joint pupil
  proceed_recommended=True,
  warnings=[
    "LED position error ~0.8 mm — joint position calibration via gradient descent",
    "Sample defocus 3 um — defocused pupil model will be estimated",
    "Edge LEDs 60% dimmer — intensity normalization applied automatically"
  ]
)
```

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Tissue |
|--------|-----------|-------------|
| **Photon Agent** | | |
| N_effective | 1.20e5 | 8.4e4 |
| SNR | 50.8 dB | 49.2 dB |
| Quality tier | excellent | excellent |
| **Mismatch Agent** | | |
| Severity | 0.089 (low) | 0.230 (moderate) |
| Dominant error | none | LED position + defocus |
| Expected gain | +0.89 dB | +2.30 dB |
| **Recoverability Agent** | | |
| Score | 0.85 (excellent) | 0.72 (good) |
| Expected PSNR | 34.2 dB | 30.5 dB |
| Verdict | excellent | good |
| **Analysis Agent** | | |
| Primary bottleneck | compression | **mismatch** |
| P(success) | 0.818 | 0.704 |
| **Negotiator** | | |
| P_joint | 0.757 | 0.574 |
| **PreFlight** | | |
| Runtime | 45s | 320s |
| Warnings | 0 | 3 |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (sequential -> gradient descent -> deep FPM) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Adaptive:** FPM-specific joint calibration of LED positions, pupil aberrations, and object recovery in a single optimization loop.
