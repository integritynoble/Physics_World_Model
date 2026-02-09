# Confocal Live-Cell Microscopy Working Process

## End-to-End Pipeline for Laser Scanning Confocal Live-Cell Imaging

This document traces a complete confocal live-cell microscopy experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Deconvolve this confocal image of live U2OS cells expressing GFP-tagged
 actin. Acquired with 60x/1.4 NA oil objective, 488 nm laser, 1 AU pinhole.
 Measurement: confocal_livecell.tif"
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "confocal_livecell.tif" detected
#   operator_type=OperatorType.linear_operator,
#   files=["confocal_livecell.tif"],
#   params={"excitation_wavelength_nm": 488, "numerical_aperture": 1.4,
#           "pinhole_au": 1.0, "magnification": 60}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> confocal_livecell entry
confocal_livecell:
  keywords: [confocal, live_cell, scanning, pinhole, fluorescence]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="confocal_livecell",
#   confidence=0.94,
#   reasoning="Matched keywords: confocal, live_cell, fluorescence"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the confocal_livecell registry entry:

```python
system = plan_agent.build_imaging_system("confocal_livecell")
# ImagingSystem(
#   modality_key="confocal_livecell",
#   display_name="Confocal Live-Cell Microscopy",
#   signal_dims={"x": [512, 512], "y": [512, 512]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...5 elements...],
#   default_solver="richardson_lucy"
# )
```

**Confocal Live-Cell Element Chain (5 elements):**

```
Laser Source (488 nm) --> Scanning Mirrors (Galvo) --> Objective Lens (60x/1.4 NA Oil) --> Confocal Pinhole (1 AU) --> PMT Detector
  throughput=1.0            throughput=0.92              throughput=0.75                      throughput=0.70           throughput=0.25
  noise: none               noise: alignment             noise: aberration                    noise: alignment          noise: shot+read+thermal
  power_mw=5                scan_rate=8000 Hz            immersion=oil                        pinhole_diam=60 um        QE=0.25
                            bidirectional=true           psf_sigma_lat=1.0 px
                                                         psf_sigma_ax=3.5 px
```

**Cumulative throughput:** `0.92 x 0.75 x 0.70 x 0.25 = 0.121`

**Key physics:** The confocal PSF is the product of excitation and detection PSFs, making it sharper than widefield (~1.4x better lateral resolution) but at the cost of severely reduced light throughput (only 12.1% of photons reach the detector, vs 50.2% for widefield).

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  confocal_livecell:
    model_id: "microscopy_fluorescence"
    parameters:
      power_w: 0.0005
      wavelength_nm: 488
      na: 1.4
      n_medium: 1.515
      qe: 0.45
      exposure_s: 0.001
  ```

### Computation

```python
# 1. Photon energy
E_photon = h * c / wavelength_nm
         = (6.626e-34 * 3e8) / (488e-9)
         = 4.074e-19 J

# 2. Collection solid angle
#    For oil immersion, NA can exceed 1.0 because n_medium > 1:
solid_angle = (na / n_medium)^2 / (4 * pi)
            = (1.4 / 1.515)^2 / (4 * pi)
            = 0.8531 / 12.566
            = 0.0679

# 3. Raw photon count
#    Note: confocal scans point-by-point; exposure_s is per-pixel dwell time
N_raw = power_w * qe * solid_angle * exposure_s / E_photon
      = 0.0005 * 0.45 * 0.0679 * 0.001 / 4.074e-19
      = 3.75e10 photons

# 4. Apply cumulative throughput (0.121 -- very low for confocal!)
N_effective = 3.75e10 * 0.121 = 4.54e9 photons/pixel

# 5. Noise variances
shot_var   = N_effective = 4.54e9              # Poisson
read_var   = 0                                 # PMT has negligible read noise
dark_var   = dark_current_cps * exposure_s
           = 50 * 0.001 = 0.05                 # Negligible
thermal_var = 100                              # PMT thermal noise (moderate)
total_var  = 4.54e9 + 0 + 0.05 + 100

# 6. SNR
SNR = N_effective / sqrt(total_var) ~ sqrt(N_effective) = 6.74e4
SNR_db = 20 * log10(6.74e4) = 96.6 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=4.54e9,
  snr_db=96.6,
  noise_regime=NoiseRegime.shot_limited,    # shot_var/total_var >> 0.9
  shot_noise_sigma=6.74e4,
  read_noise_sigma=0.0,                     # PMT
  total_noise_sigma=6.74e4,
  feasible=True,
  quality_tier="excellent",                 # SNR > 30 dB
  throughput_chain=[
    {"Laser Source (488 nm)": 1.0},
    {"Scanning Mirrors (Galvo)": 0.92},
    {"Objective Lens (60x/1.4 NA Oil)": 0.75},
    {"Confocal Pinhole (1 AU)": 0.70},
    {"PMT Detector": 0.25}
  ],
  noise_model="poisson",
  explanation="Shot-noise-limited despite low confocal throughput (12.1%). "
              "PMT has no read noise. Adequate SNR for deconvolution."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"confocal_livecell"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  confocal_livecell:
    parameters:
      psf_sigma:   {range: [0.3, 2.5], typical_error: 0.2, weight: 0.35}
      defocus:     {range: [-1.5, 1.5], typical_error: 0.4, weight: 0.35}
      background:  {range: [0.0, 0.10], typical_error: 0.02, weight: 0.15}
      gain:        {range: [0.6, 1.4], typical_error: 0.08, weight: 0.15}
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.35 * |0.2| / 2.2     # psf_sigma:  0.032
  + 0.35 * |0.4| / 3.0     # defocus:    0.047
  + 0.15 * |0.02| / 0.10   # background: 0.030
  + 0.15 * |0.08| / 0.8    # gain:       0.015
S = 0.124  # Low severity

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.24 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="confocal_livecell",
  mismatch_family="grid_search",
  parameters={
    "psf_sigma":  {"typical_error": 0.2, "range": [0.3, 2.5], "weight": 0.35},
    "defocus":    {"typical_error": 0.4, "range": [-1.5, 1.5], "weight": 0.35},
    "background": {"typical_error": 0.02, "range": [0.0, 0.10], "weight": 0.15},
    "gain":       {"typical_error": 0.08, "range": [0.6, 1.4], "weight": 0.15}
  },
  severity_score=0.124,
  correction_method="grid_search",
  expected_improvement_db=1.24,
  explanation="Low mismatch severity. Focal drift during live-cell imaging is the "
              "dominant concern for time-lapse, but manageable for single frames."
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
  confocal_livecell:
    signal_prior_class: "tv"
    entries:
      - {cr: 1.0, noise: "shot_limited", solver: "richardson_lucy",
         recoverability: 0.79, expected_psnr_db: 26.1,
         provenance: {dataset_id: "bii_confocal_livecell_2023", ...}}
      - {cr: 1.0, noise: "shot_limited", solver: "care_restore_2d",
         recoverability: 0.87, expected_psnr_db: 30.2, ...}
  ```

### Computation

```python
# 1. Compression ratio
CR = prod(y_shape) / prod(x_shape) = (512 * 512) / (512 * 512) = 1.0
# No compression -- confocal is a point-scanned 1:1 imaging system.

# 2. Operator diversity
# Confocal PSF is the product of excitation and detection PSFs:
#   h_confocal(r) = h_exc(r) * h_det(r)
# This is narrower than widefield PSF, giving better diversity in high frequencies.
diversity = 0.7  # Slightly better than widefield due to confocal sectioning

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.588

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="richardson_lucy", cr=1.0
#    -> recoverability=0.79, expected_psnr=26.1 dB

# 5. Best solver selection
#    care_restore_2d: 30.2 dB > richardson_lucy: 26.1 dB
#    -> recommended: "richardson_lucy" (default) or "care_restore_2d" (best)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=1.0,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.tv,
  operator_diversity_score=0.7,
  condition_number_proxy=0.588,
  recoverability_score=0.79,
  recoverability_confidence=1.0,
  expected_psnr_db=26.1,
  expected_psnr_uncertainty_db=0.9,
  recommended_solver_family="richardson_lucy",
  verdict="good",                # score >= 0.75
  calibration_table_entry={...},
  explanation="Good recoverability. RL expected 26.1 dB; CARE reaches 30.2 dB on BII confocal benchmark."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(96.6 / 40, 1.0)   = 0.0    # Excellent SNR
mismatch_score    = 0.124                       = 0.124  # Low mismatch
compression_score = 1 - 0.79                    = 0.21   # Good recoverability
solver_score      = 0.2                         = 0.2    # Default

# Primary bottleneck
primary = "compression"  # max(0.0, 0.124, 0.21, 0.2) = compression

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.124*0.5) * (1 - 0.21*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.938 * 0.895 * 0.90
  = 0.755
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="compression",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.124, compression=0.21, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Use CARE for +4.1 dB over Richardson-Lucy on confocal data",
      priority="medium",
      expected_gain_db=4.1
    ),
    Suggestion(
      text="Measure the confocal PSF with sub-diffraction beads at 488 nm",
      priority="low",
      expected_gain_db=0.8
    )
  ],
  overall_verdict="good",           # P >= 0.70
  probability_of_success=0.755,
  explanation="System is well-configured for confocal live-cell imaging. "
              "Solver choice (RL vs CARE) is the primary lever for improvement."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND CR=1.0 | No veto |
| Severe mismatch without correction | severity=0.124 < 0.7 | No veto |
| All marginal | All good/excellent | No veto |
| Joint probability floor | P=0.755 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95   # tier_prob["excellent"]
P_recoverability = 0.79   # recoverability_score
P_mismatch       = 1.0 - 0.124 * 0.7 = 0.913

P_joint = 0.95 * 0.79 * 0.913 = 0.685
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.685
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
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="confocal_livecell", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=1.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=["measurement (2D confocal single-slice TIFF)"]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Confocal forward model: y = h_confocal ** x + n
#
# The confocal PSF is the product of excitation and detection PSFs:
#   h_confocal(r) = h_exc(r) * h_det(r)
#
# For a 1 AU pinhole, the effective confocal PSF is approximately:
#   h_confocal(r) ~ exp(-r^2 / (2 * sigma_lat^2))
#   sigma_lat = 0.44 * lambda_em / NA = 0.44 * 525 / 1.4 = 165 nm
#   In pixels: sigma_lat_px ~ 1.0 px (at 60x mag, 6.5 um pixel -> 108 nm/px)
#
# The confocal PSF FWHM is ~1.4x smaller than widefield:
#   FWHM_confocal ~ 0.37 * lambda / NA  (vs 0.51 * lambda / NA for widefield)
#
# Input:  x = (512, 512) fluorescence distribution
# Output: y = (512, 512) confocal image (same size)

class ConfocalOperator(PhysicsOperator):
    def forward(self, x):
        """y = h_confocal ** x (2D convolution with confocal PSF)"""
        return fftconvolve(x, self.psf_confocal, mode='same')

    def adjoint(self, y):
        """x_hat = h_confocal_flipped ** y"""
        return fftconvolve(y, self.psf_confocal[::-1, ::-1], mode='same')

    def check_adjoint(self):
        """Verify <Ax, y> ~ <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided measurement:
y = load_tiff("confocal_livecell.tif")   # (512, 512)

# If simulating:
x_true = load_ground_truth()              # (512, 512) from BII dataset
y = operator.forward(x_true)              # (512, 512) blurred
y = np.random.poisson(y * gain) / gain    # Shot noise (Poisson)
y += np.random.randn(*y.shape) * 0.02     # Detector noise
```

### Step 9c: Reconstruction with Richardson-Lucy

```python
from pwm_core.recon import run_richardson_lucy

class ConfocalPhysics:
    def __init__(self, psf):
        self.psf = psf

physics = ConfocalPhysics(psf_confocal)

x_hat, info = run_richardson_lucy(
    y=y,                      # (512, 512) confocal image
    physics=physics,          # Contains confocal PSF
    cfg={"iters": 30}         # 30 RL iterations
)
# x_hat shape: (512, 512) -- deconvolved image
# Expected PSNR: ~26.1 dB (BII confocal live-cell benchmark)
```

**Richardson-Lucy for confocal:**

```python
# The RL algorithm is identical to widefield, but uses the confocal PSF:
#
# x_{k+1} = x_k * (h_confocal_flipped ** (y / (h_confocal ** x_k)))
#
# Because the confocal PSF is narrower, RL converges faster and
# produces sharper results than widefield RL, but is more sensitive
# to noise amplification at high iteration counts.
#
# For live-cell data with moderate noise, 30 iterations is typical.
# Early stopping at 20 iterations may be needed for noisy data.
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| Richardson-Lucy | Traditional | 26.1 dB | No | `run_richardson_lucy(y, physics, {"iters": 30})` |
| CARE U-Net | Deep Learning | 30.2 dB | Yes | `care_restore_2d(y, model_path="weights/care_confocal.pth")` |
| Wiener Filter | Traditional | 24.0 dB | No | `wiener_deconv(y, psf, snr=50)` |

### Step 9d: Metrics

```python
# PSNR
psnr = 10 * log10(max_val^2 / mse(x_hat, x_true))
# ~ 26.1 dB (Richardson-Lucy, 30 iterations)

# SSIM (structural similarity)
ssim_val = ssim(x_hat, x_true)

# Resolution metric: confocal lateral resolution
# Theoretical: 0.37 * lambda / NA = 0.37 * 525 / 1.4 = 139 nm
# After deconvolution: approach ~120 nm (sub-diffraction improvement)

# Temporal resolution metric (for live-cell):
# Frame rate = scan_rate_hz / pixels_per_line / n_lines
# = 8000 / 512 / 512 ~ 0.03 frames/s (30 seconds per frame)
# Bidirectional scanning: ~15 seconds per frame
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
|   +-- psf.npy            # Confocal PSF (11, 11) + SHA256 hash
+-- metrics.json           # PSNR, SSIM, resolution metrics
+-- operator.json          # Operator parameters (PSF sigma, NA, wavelength, pinhole)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized confocal live-cell pipeline (SNR 96.6 dB, mismatch 0.124). In practice, live-cell confocal imaging faces unique challenges: fast scanning reduces dwell time (fewer photons), cells move during acquisition, and photobleaching/phototoxicity limit laser power.

---

## Real Experiment: User Prompt

```
"Fast-scanning confocal of live U2OS cells. Had to use very low laser
 power to keep cells alive. Pixel dwell time is only 2 microseconds.
 Cells are moving so there may be motion blur. Measurement: live_fast.tif,
 60x/1.4 NA oil, 488 nm, 1 AU pinhole."
```

**Key difference:** Very short dwell time (2 us vs nominal 1 ms) reduces photon count by 500x. Cell motion introduces additional blurring.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["live_fast.tif"],
#   params={"excitation_wavelength_nm": 488, "numerical_aperture": 1.4,
#           "pinhole_au": 1.0, "pixel_dwell_time_us": 2}
# )
```

---

## R2. PhotonAgent -- Fast Scanning, Low Photon Budget

### Real parameters

```yaml
confocal_livecell_fast:
  power_w: 0.0002          # 200 uW (reduced for live cells)
  wavelength_nm: 488
  na: 1.4
  n_medium: 1.515
  qe: 0.25                 # PMT QE
  exposure_s: 0.000002      # 2 microsecond dwell time (500x shorter)
```

### Computation

```python
# Raw photon count (500x fewer than ideal)
N_raw = 0.0002 * 0.25 * 0.0679 * 0.000002 / 4.074e-19
      = 1.67e7 photons

# Apply cumulative throughput (0.121)
N_effective = 1.67e7 * 0.121 = 2.02e6 photons/pixel

# Noise variances
shot_var   = 2.02e6
dark_var   = 50 * 0.000002 = 0.0001            # Negligible
thermal_var = 100
total_var  = 2.02e6 + 100

# SNR
SNR = 2.02e6 / sqrt(2.02e6) = 1421
SNR_db = 20 * log10(1421) = 63.1 dB
```

### Output

```python
PhotonReport(
  n_photons_per_pixel=2.02e6,
  snr_db=63.1,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="excellent",                      # Still > 30 dB
  explanation="Adequate SNR even with fast scanning. Confocal laser scanning "
              "concentrates photon budget on one pixel at a time."
)
```

---

## R3. MismatchAgent -- Motion + Focal Drift

```python
# Live-cell mismatch sources
psi_true = {
    "psf_sigma": +0.5,    # Cell motion broadens effective PSF
    "defocus":   +0.8,     # Focal drift during 15-second scan
    "background": 0.05,    # Out-of-focus fluorescence
    "gain":       -0.15,   # PMT gain drift over time
}

# Severity computation
S = 0.35 * |0.5| / 2.2     # psf_sigma:  0.080
  + 0.35 * |0.8| / 3.0     # defocus:    0.093
  + 0.15 * |0.05| / 0.10   # background: 0.075
  + 0.15 * |0.15| / 0.8    # gain:       0.028
S = 0.276  # Moderate severity

improvement_db = clip(10 * 0.276, 0, 20) = 2.76 dB
```

### Output

```python
MismatchReport(
  severity_score=0.276,
  correction_method="grid_search",
  expected_improvement_db=2.76,
  explanation="Moderate mismatch. Cell motion and focal drift during scanning are "
              "the primary error sources. Motion blur makes PSF non-stationary."
)
```

---

## R4. RecoverabilityAgent -- Degraded by Motion

```python
# Calibration lookup: noise="photon_starved", solver="richardson_lucy"
# -> recoverability=0.55, expected_psnr=21.8 dB

RecoverabilityReport(
  compression_ratio=1.0,
  recoverability_score=0.55,
  expected_psnr_db=21.8,
  verdict="marginal",
  explanation="Motion blur creates a non-stationary PSF that RL cannot fully correct. "
              "CARE trained on live-cell data is more robust."
)
```

---

## R5. AnalysisAgent

```python
photon_score      = 1 - min(63.1 / 40, 1.0)   = 0.0
mismatch_score    = 0.276
compression_score = 1 - 0.55                    = 0.45
solver_score      = 0.2

primary = "compression"  # 0.45 (inflated by motion mismatch)

P = 1.0 * 0.862 * 0.775 * 0.90 = 0.601

SystemAnalysis(
  primary_bottleneck="mismatch",
  probability_of_success=0.601,
  suggestions=[
    Suggestion(text="Use CARE trained on confocal live-cell pairs for motion robustness",
               priority="high", expected_gain_db=4.0),
    Suggestion(text="Enable hardware autofocus to reduce focal drift",
               priority="medium", expected_gain_db=1.5),
    Suggestion(text="Reduce frame size to decrease scan time and motion blur",
               priority="medium", expected_gain_db=1.0)
  ],
  overall_verdict="marginal"
)
```

---

## R6. AgentNegotiator

```python
P_photon         = 0.95
P_recoverability = 0.55
P_mismatch       = 1.0 - 0.276 * 0.7 = 0.807

P_joint = 0.95 * 0.55 * 0.807 = 0.422

NegotiationResult(
  vetoes=[],
  proceed=True,
  probability_of_success=0.422
)
```

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=1.5,
  proceed_recommended=True,
  warnings=[
    "Mismatch severity 0.276 -- cell motion and focal drift may limit deconvolution",
    "Recoverability marginal (0.55) -- CARE recommended for live-cell data"
  ],
  what_to_upload=["measurement (2D confocal single-slice TIFF)"]
)
```

---

## R8. Pipeline Runner

### Step R8a: RL with Standard PSF

```python
x_rl, _ = run_richardson_lucy(y_live, ConfocalPhysics(psf_standard), {"iters": 20})
# PSNR = 21.8 dB  <-- limited by non-stationary PSF (motion blur)
# RL amplifies noise in high-frequency structures without correcting motion
```

### Step R8b: CARE Deep Learning

```python
from pwm_core.recon.care_unet import care_restore_2d

x_care = care_restore_2d(
    y=y_live,
    model_path="weights/care_confocal_livecell.pth",
    device="cuda"
)
# PSNR = 30.2 dB  <-- +8.4 dB improvement, robust to motion
```

### Step R8c: Final Comparison

| Configuration | RL (20 iter) | CARE | Notes |
|---------------|-------------|------|-------|
| Fast scan, motion blur | **21.8 dB** | **30.2 dB** | CARE handles non-stationary PSF |
| Standard scan (no motion) | **26.1 dB** | **30.2 dB** | RL recovers when PSF is correct |

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Experiment |
|--------|-----------|-----------------|
| **Photon Agent** | | |
| N_effective | 4.54e9 | 2.02e6 |
| SNR | 96.6 dB | 63.1 dB |
| Quality tier | excellent | excellent |
| **Mismatch Agent** | | |
| Severity | 0.124 (low) | 0.276 (moderate) |
| Dominant error | defocus | motion + defocus |
| **Recoverability Agent** | | |
| Score | 0.79 (good) | 0.55 (marginal) |
| Expected PSNR | 26.1 dB | 21.8 dB |
| Verdict | good | **marginal** |
| **Negotiator** | | |
| P_joint | 0.685 | 0.422 |
| **Pipeline** | | |
| RL PSNR | 26.1 dB | 21.8 dB |
| CARE PSNR | 30.2 dB | **30.2 dB** |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (RL -> CARE -> Wiener) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Live-cell aware:** The pipeline recognizes that confocal live-cell imaging trades SNR for temporal resolution and recommends deep learning solvers that are robust to motion-induced PSF mismatch.
