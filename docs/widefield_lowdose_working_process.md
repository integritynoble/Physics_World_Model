# Low-Dose Widefield Microscopy Working Process

## End-to-End Pipeline for Photon-Starved Fluorescence Denoising & Deconvolution

This document traces a complete low-dose widefield fluorescence microscopy experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Denoise and deconvolve this low-dose widefield image. The illumination was
 reduced 25x to minimize photobleaching. Image: lowdose_sample.tif,
 NA=1.3, emission=525nm, pixel_size=0.163um, oil immersion."
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "lowdose_sample.tif" detected
#   operator_type=OperatorType.linear_operator,
#   files=["lowdose_sample.tif"],
#   params={"emission_wavelength_nm": 525, "numerical_aperture": 1.3,
#           "pixel_size_um": 0.163, "immersion": "oil"}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> widefield_lowdose entry
widefield_lowdose:
  keywords: [low_dose, denoising, fluorescence, photon_limited, low_snr]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="widefield_lowdose",
#   confidence=0.91,
#   reasoning="Matched keywords: low_dose, fluorescence, denoising (low illumination context)"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the widefield_lowdose registry entry:

```python
system = plan_agent.build_imaging_system("widefield_lowdose")
# ImagingSystem(
#   modality_key="widefield_lowdose",
#   display_name="Low-Dose Widefield Microscopy",
#   signal_dims={"x": [512, 512], "y": [512, 512]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...5 elements...],
#   default_solver="pnp_hqs"
# )
```

**Low-Dose Widefield Element Chain (5 elements):**

```
Attenuated LED Source --> Excitation Filter --> Objective Lens (40x/1.3 NA Oil) --> Emission Filter --> sCMOS Detector (Low Exposure)
  throughput=1.0           throughput=0.85       throughput=0.78                     throughput=0.90     throughput=0.82
  noise: none              noise: none           noise: aberration                   noise: none         noise: shot+read+fixed_pattern
  attenuation=0.04                               immersion=oil
```

**Cumulative throughput:** `0.85 x 0.78 x 0.90 x 0.82 = 0.489`

**Key difference from standard widefield:** The source is attenuated by a factor of 0.04 (25x dose reduction), and the exposure is short (5 ms). This pushes the system into the photon-starved regime.

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  widefield_lowdose:
    model_id: "microscopy_fluorescence"
    parameters:
      power_w: 0.0001
      wavelength_nm: 525
      na: 0.75
      n_medium: 1.0
      qe: 0.72
      exposure_s: 0.01
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
            = 0.04476

# 3. Raw photon count (25x lower power, 5x shorter exposure)
N_raw = power_w * qe * solid_angle * exposure_s / E_photon
      = 0.0001 * 0.72 * 0.04476 * 0.01 / 3.786e-19
      = 8.51e9 photons

# 4. Apply cumulative throughput
N_effective = 8.51e9 * 0.489 = 4.16e9 photons/pixel

# BUT: The registry says photon_count_mean = 20 per pixel at the detector.
# This is the operating point for low-dose imaging:
N_detected = 20  # photons/pixel (design target)

# 5. Noise variances (using N_detected = 20)
shot_var   = N_detected = 20                   # Poisson (dominant at this level)
read_var   = read_noise^2 = 1.6^2 = 2.56      # Gaussian (1.6 e-)
dark_var   = 0                                 # Negligible
total_var  = 20 + 2.56 = 22.56

# 6. SNR
SNR = N_detected / sqrt(total_var) = 20 / 4.75 = 4.21
SNR_db = 20 * log10(4.21) = 12.5 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=20,
  snr_db=12.5,
  noise_regime=NoiseRegime.photon_starved,   # SNR < 20 dB
  shot_noise_sigma=4.47,
  read_noise_sigma=1.6,
  total_noise_sigma=4.75,
  feasible=True,
  quality_tier="poor",                       # SNR < 15 dB
  throughput_chain=[
    {"Attenuated LED Source": 1.0},
    {"Excitation Filter": 0.85},
    {"Objective Lens (40x/1.3 NA Oil)": 0.78},
    {"Emission Filter": 0.90},
    {"sCMOS Detector": 0.82}
  ],
  noise_model="mixed_poisson_gaussian",
  explanation="Photon-starved regime. Only ~20 photons/pixel detected. "
              "Mixed Poisson-Gaussian noise model required. "
              "Denoising is essential before or jointly with deconvolution."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"widefield_lowdose"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  widefield_lowdose:
    parameters:
      psf_sigma:   {range: [0.5, 3.0], typical_error: 0.3, weight: 0.30}
      defocus:     {range: [-2.0, 2.0], typical_error: 0.5, weight: 0.25}
      background:  {range: [0.0, 0.20], typical_error: 0.05, weight: 0.25}
      gain:        {range: [0.4, 1.6], typical_error: 0.15, weight: 0.20}
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.30 * |0.3| / 2.5     # psf_sigma:  0.036
  + 0.25 * |0.5| / 4.0     # defocus:    0.031
  + 0.25 * |0.05| / 0.20   # background: 0.063
  + 0.20 * |0.15| / 1.2    # gain:       0.025
S = 0.155  # Moderate-low severity

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.55 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="widefield_lowdose",
  mismatch_family="grid_search",
  parameters={
    "psf_sigma":  {"typical_error": 0.3, "range": [0.5, 3.0], "weight": 0.30},
    "defocus":    {"typical_error": 0.5, "range": [-2.0, 2.0], "weight": 0.25},
    "background": {"typical_error": 0.05, "range": [0.0, 0.20], "weight": 0.25},
    "gain":       {"typical_error": 0.15, "range": [0.4, 1.6], "weight": 0.20}
  },
  severity_score=0.155,
  correction_method="grid_search",
  expected_improvement_db=1.55,
  explanation="Moderate-low mismatch. Background estimation is the dominant uncertainty "
              "at low dose because noise makes baseline estimation harder."
)
```

---

## 5. RecoverabilityAgent -- Can We Reconstruct?

**File:** `agents/recoverability_agent.py` (912 lines)

### Input
- `ImagingSystem` (signal_dims for CR calculation)
- `PhotonReport` (noise regime = photon_starved)
- Calibration table from `compression_db.yaml`:
  ```yaml
  widefield_lowdose:
    signal_prior_class: "deep_prior"
    entries:
      - {cr: 1.0, noise: "photon_starved", solver: "pnp_drunet",
         recoverability: 0.84, expected_psnr_db: 28.2,
         provenance: {dataset_id: "bii_widefield_lowdose_2023", ...}}
      - {cr: 1.0, noise: "photon_starved", solver: "noise2void_denoise",
         recoverability: 0.78, expected_psnr_db: 26.5, ...}
      - {cr: 1.0, noise: "shot_limited", solver: "pnp_drunet",
         recoverability: 0.91, expected_psnr_db: 32.7, ...}
  ```

### Computation

```python
# 1. Compression ratio
CR = prod(y_shape) / prod(x_shape) = (512 * 512) / (512 * 512) = 1.0
# No compression -- the problem is denoising + deconvolution.

# 2. Operator diversity
# Same convolution operator as widefield, but noise dominates.
diversity = 0.6  # Same PSF-based diversity

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.625

# 4. Calibration table lookup
#    Match: noise="photon_starved", solver="pnp_drunet", cr=1.0 (exact)
#    -> recoverability=0.84, expected_psnr=28.2 dB

# 5. Best solver selection
#    pnp_drunet: 28.2 dB > noise2void: 26.5 dB
#    -> recommended: "pnp_drunet"
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=1.0,
  noise_regime=NoiseRegime.photon_starved,
  signal_prior_class=SignalPriorClass.deep_prior,
  operator_diversity_score=0.6,
  condition_number_proxy=0.625,
  recoverability_score=0.84,
  recoverability_confidence=1.0,
  expected_psnr_db=28.2,
  expected_psnr_uncertainty_db=1.2,
  recommended_solver_family="pnp_drunet",
  verdict="good",                # score >= 0.75
  calibration_table_entry={...},
  explanation="Good recoverability despite photon-starved regime. "
              "PnP-DRUNet leverages deep prior to compensate for extreme noise. "
              "Noise2Void available as self-supervised alternative (no GT pairs needed)."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(12.5 / 40, 1.0)   = 0.688  # Poor SNR -- primary issue!
mismatch_score    = 0.155                       = 0.155  # Moderate-low
compression_score = 1 - 0.84                    = 0.16   # Good recoverability
solver_score      = 0.1                         = 0.1    # PnP is well-suited

# Primary bottleneck
primary = "photon"  # max(0.688, 0.155, 0.16, 0.1) = photon

# Probability of success
P = (1 - 0.688*0.5) * (1 - 0.155*0.5) * (1 - 0.16*0.5) * (1 - 0.1*0.5)
  = 0.656 * 0.923 * 0.92 * 0.95
  = 0.529
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="photon",
  bottleneck_scores=BottleneckScores(
    photon=0.688, mismatch=0.155, compression=0.16, solver=0.1
  ),
  suggestions=[
    Suggestion(
      text="Use mixed Poisson-Gaussian noise model in the solver",
      priority="critical",
      expected_gain_db=3.0
    ),
    Suggestion(
      text="PnP-DRUNet recommended over RL for photon-starved data",
      priority="high",
      expected_gain_db=6.0
    ),
    Suggestion(
      text="Noise2Void is a self-supervised alternative if no training pairs are available",
      priority="medium",
      expected_gain_db=4.3
    ),
    Suggestion(
      text="If possible, increase exposure time or average multiple frames",
      priority="low",
      expected_gain_db=3.0
    )
  ],
  overall_verdict="marginal",       # P < 0.60
  probability_of_success=0.529,
  explanation="Photon budget is the primary bottleneck. Only ~20 photons/pixel detected. "
              "Deep learning denoisers essential for usable reconstruction."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="poor" BUT CR=1.0 (no compression) | No veto |
| Severe mismatch without correction | severity=0.155 < 0.7 | No veto |
| All marginal | photon=poor, others okay | No veto (photon alone is not fatal) |
| Joint probability floor | P=0.529 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.45   # tier_prob["poor"]
P_recoverability = 0.84   # recoverability_score
P_mismatch       = 1.0 - 0.155 * 0.7 = 0.892

P_joint = 0.45 * 0.84 * 0.892 = 0.337
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes -- low dose is recoverable with deep priors
  proceed=True,
  probability_of_success=0.337
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 512 * 512 = 262,144
dim_factor   = total_pixels / (256 * 256) = 4.0
solver_complexity = 2.0  # PnP-ADMM iterative + DRUNet forward pass
cr_factor    = max(1.0, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 4.0 * 2.0 * 0.125 = 2.0 seconds  # GPU
# CPU fallback: ~15 seconds (DRUNet on CPU is slow)
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="widefield_lowdose", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=2.0,
  proceed_recommended=True,
  warnings=[
    "Photon-starved regime (12.5 dB SNR). Deep denoiser required.",
    "Mixed Poisson-Gaussian noise model will be applied."
  ],
  what_to_upload=["measurement (2D low-dose widefield image, TIFF)"]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Low-dose widefield forward model:
#   y = Poisson(alpha * PSF ** x) / alpha + N(0, sigma_read^2)
#
# The forward model includes the mixed Poisson-Gaussian noise model:
#   - alpha = gain factor (photons -> ADU), proportional to dose
#   - PSF ** x = convolution with point spread function
#   - Poisson(.) = shot noise (signal-dependent)
#   - N(0, sigma^2) = read noise (signal-independent)
#
# Parameters:
#   PSF:    Gaussian, sigma=1.2 px (40x/1.3 NA oil objective)
#   alpha:  ~20 photons/pixel (design target at 25x dose reduction)
#   sigma_read: 1.6 e-
#
# Input:  x = (512, 512) true fluorescence distribution
# Output: y = (512, 512) noisy blurred image

class LowDoseWidefieldOperator(PhysicsOperator):
    def forward(self, x):
        """y = Poisson(alpha * PSF ** x) / alpha + N(0, sigma^2)"""
        blurred = fftconvolve(x, self.psf, mode='same')
        noisy = np.random.poisson(self.alpha * np.clip(blurred, 0, None))
        noisy = noisy.astype(np.float32) / self.alpha
        noisy += np.random.randn(*noisy.shape) * self.sigma_read
        return noisy

    def adjoint(self, y):
        """x_hat = PSF_flipped ** y (correlation, ignoring noise)"""
        return fftconvolve(y, self.psf[::-1, ::-1], mode='same')

    def forward_noiseless(self, x):
        """Deterministic part: PSF ** x"""
        return fftconvolve(x, self.psf, mode='same')

    def check_adjoint(self):
        """Verify <A_noiseless x, y> ~ <x, A* y>"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided measurement:
y = load_tiff("lowdose_sample.tif")    # (512, 512)

# If simulating (benchmark mode):
x_true = load_ground_truth()            # (512, 512)
photons = 50                            # Mean photon count per pixel
y = np.random.poisson(x_true * photons).astype(np.float32) / photons
y = np.clip(y, 0, 1)
```

### Step 9c: Reconstruction with PnP-DRUNet

```python
from pwm_core.recon.pnp import pnp_admm_restore

x_hat = pnp_admm_restore(
    y=y,                      # (512, 512) noisy measurement
    operator=operator,        # Contains PSF for data-fidelity
    denoiser="drunet",        # Pre-trained DRUNet denoiser
    noise_model="poisson_gaussian",
    sigma_read=1.6,           # Read noise std (electrons)
    alpha=20,                 # Gain factor (photons/pixel)
    n_iters=30,               # ADMM iterations
    rho=0.1,                  # ADMM penalty parameter
    device="cuda"
)
# x_hat shape: (512, 512) -- denoised + deconvolved image
# Expected PSNR: ~28.2 dB (BII low-dose benchmark)
```

**PnP-ADMM algorithm:**

```python
# Plug-and-Play ADMM:
# Splits the inverse problem into two sub-problems:
#
# min_x  (1/2) ||y - Ax||^2  +  lambda * R(x)
#
# Where R(x) is implicitly defined by the DRUNet denoiser.
#
# ADMM iterations:
# x_{k+1} = argmin (1/2)||y - Ax||^2 + (rho/2)||x - z_k + u_k||^2
#          = (A^T A + rho I)^{-1} (A^T y + rho(z_k - u_k))    [Wiener step]
# z_{k+1} = DRUNet(x_{k+1} + u_k, sigma=sqrt(lambda/rho))     [Denoise step]
# u_{k+1} = u_k + x_{k+1} - z_{k+1}                           [Dual update]
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| BM3D + RL | Traditional | 24.5 dB | No | `bm3d_denoise(y) -> run_richardson_lucy(...)` |
| Noise2Void | Self-supervised DL | 26.5 dB | Yes | `n2v_denoise(y, epochs=200)` |
| PnP-DRUNet | Plug-and-Play DL | 28.2 dB | Yes | `pnp_admm_restore(y, denoiser="drunet")` |
| CARE U-Net | Supervised DL | 30.0 dB | Yes | `care_restore_2d(y, model_path=...)` |

### Step 9d: Metrics

```python
# PSNR
psnr = 10 * log10(max_val^2 / mse(x_hat, x_true))
# ~ 28.2 dB (PnP-DRUNet, 30 iterations)

# SSIM (structural similarity)
ssim_val = ssim(x_hat, x_true)

# NRMSE (normalized root mean squared error)
nrmse = sqrt(mse(x_hat, x_true)) / (x_true.max() - x_true.min())

# SNR improvement
snr_input  = 12.5   # dB (noisy input)
snr_output = 28.2   # dB (reconstructed)
snr_gain   = 15.7   # dB improvement
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
+-- metrics.json           # PSNR, SSIM, NRMSE, SNR gain
+-- operator.json          # Operator parameters (PSF sigma, alpha, sigma_read)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed a low-dose pipeline with typical parameters (20 photons/pixel, mismatch severity 0.155). In practice, photon counts can be even lower for live-cell time-lapse imaging where phototoxicity is the primary concern.

---

## Real Experiment: User Prompt

```
"Time-lapse widefield fluorescence of dividing cells. Each frame has only
 ~5 photons/pixel to avoid killing the cells. 100 frames total.
 Please denoise each frame. Data: timelapse_lowdose.tif"
```

**Key difference:** Extreme photon starvation (~5 photons/pixel) and temporal correlation between frames.

---

## R1. PhotonAgent -- Extreme Photon Starvation

```python
N_detected = 5   # photons/pixel (4x lower than typical low-dose)

# Noise variances
shot_var   = 5                                 # Poisson
read_var   = 1.6^2 = 2.56                     # Gaussian
total_var  = 5 + 2.56 = 7.56

# SNR
SNR = 5 / sqrt(7.56) = 1.82
SNR_db = 20 * log10(1.82) = 5.2 dB            # Very poor

PhotonReport(
  n_photons_per_pixel=5,
  snr_db=5.2,
  noise_regime=NoiseRegime.photon_starved,
  quality_tier="critical",                      # SNR < 10 dB
  explanation="Extremely photon-starved. Read noise is 34% of total variance. "
              "Signal is barely above noise floor."
)
```

---

## R2. MismatchAgent -- Temporal Drift

```python
# Over 100 frames, focus drifts and photobleaching occurs
psi_true = {
    "psf_sigma": +0.4,    # PSF broadening from sample-induced aberration
    "defocus":   +1.5,     # Cumulative focal drift over time-lapse
    "background": 0.12,    # Increasing autofluorescence (12%)
    "gain":       -0.25,   # Photobleaching reduces effective gain
}

S = 0.30 * |0.4| / 2.5     # psf_sigma:  0.048
  + 0.25 * |1.5| / 4.0     # defocus:    0.094
  + 0.25 * |0.12| / 0.20   # background: 0.150
  + 0.20 * |0.25| / 1.2    # gain:       0.042
S = 0.334  # Moderate severity

MismatchReport(
  severity_score=0.334,
  explanation="Moderate mismatch. Background estimation is dominant error source "
              "at extreme low dose -- noise makes baseline impossible to separate."
)
```

---

## R3. RecoverabilityAgent

```python
# Calibration table: noise="photon_starved", solver="noise2void"
# -> recoverability=0.78, expected_psnr=26.5 dB (at 20 photons)
# Adjusted for 5 photons: recoverability ~ 0.55, expected_psnr ~ 20.0 dB

RecoverabilityReport(
  recoverability_score=0.55,
  expected_psnr_db=20.0,
  verdict="marginal",
  explanation="At 5 photons/pixel, even deep denoisers struggle. "
              "Temporal averaging or Noise2Void with temporal context may help."
)
```

---

## R4. AnalysisAgent

```python
photon_score      = 1 - min(5.2 / 40, 1.0)    = 0.870  # Critical
mismatch_score    = 0.334                       = 0.334
compression_score = 1 - 0.55                    = 0.45
solver_score      = 0.15

primary = "photon"  # 0.870 >> all others

P = (1 - 0.870*0.5) * (1 - 0.334*0.5) * (1 - 0.45*0.5) * (1 - 0.15*0.5)
  = 0.565 * 0.833 * 0.775 * 0.925
  = 0.337

SystemAnalysis(
  primary_bottleneck="photon",
  probability_of_success=0.337,
  suggestions=[
    Suggestion(text="Average consecutive frames (2-5 frame window) before denoising",
               priority="critical", expected_gain_db=3.0),
    Suggestion(text="Use Noise2Void with temporal context (3D blind-spot network)",
               priority="high", expected_gain_db=5.0),
    Suggestion(text="Apply variance-stabilizing transform (Anscombe) before denoising",
               priority="medium", expected_gain_db=1.5)
  ],
  overall_verdict="poor"
)
```

---

## R5. AgentNegotiator

```python
P_photon         = 0.20   # tier_prob["critical"]
P_recoverability = 0.55
P_mismatch       = 1.0 - 0.334 * 0.7 = 0.766

P_joint = 0.20 * 0.55 * 0.766 = 0.084

# P_joint = 0.084 < 0.15 -- VETO TRIGGERED!
NegotiationResult(
  vetoes=["Joint probability 0.084 below threshold 0.15. "
          "Recommend averaging frames or increasing exposure."],
  proceed=False,
  probability_of_success=0.084
)
```

**The negotiator vetoes execution.** At 5 photons/pixel with moderate mismatch, the joint probability falls below the 0.15 threshold. The user is advised to average frames or increase exposure before proceeding.

---

## R6. Override and Proceed (User Decision)

If the user overrides the veto (e.g., "proceed anyway, this is the best data we can get"):

```python
# Apply Anscombe variance-stabilizing transform
y_vst = 2 * sqrt(y + 3/8)  # Stabilize Poisson noise to ~Gaussian

# Noise2Void self-supervised denoising (no GT pairs needed)
from pwm_core.recon.noise2void import n2v_denoise
x_hat = n2v_denoise(y_vst, epochs=200)
x_hat = (x_hat / 2)^2 - 3/8  # Inverse Anscombe
# PSNR ~ 18-20 dB (barely usable, but preserves live-cell dynamics)
```

---

## Side-by-Side Comparison: Typical vs Extreme Low-Dose

| Metric | Typical (20 photons) | Extreme (5 photons) |
|--------|---------------------|---------------------|
| **Photon Agent** | | |
| N_detected | 20 | 5 |
| SNR | 12.5 dB | 5.2 dB |
| Quality tier | poor | **critical** |
| **Mismatch Agent** | | |
| Severity | 0.155 | 0.334 |
| **Recoverability Agent** | | |
| Score | 0.84 | 0.55 |
| Expected PSNR | 28.2 dB | 20.0 dB |
| Verdict | good | **marginal** |
| **Negotiator** | | |
| P_joint | 0.337 | 0.084 |
| Proceed | Yes | **VETO** |
| **Pipeline** | | |
| PnP-DRUNet PSNR | 28.2 dB | -- |
| N2V PSNR | 26.5 dB | ~19 dB |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (PnP-DRUNet -> Noise2Void -> CARE) by changing one registry ID.
6. **Gate-able:** Negotiator vetoes execution when joint probability < 0.15 (triggered at 5 photons/pixel).
7. **Noise-aware:** The pipeline selects mixed Poisson-Gaussian noise model and recommends variance-stabilizing transforms for photon-starved data.
