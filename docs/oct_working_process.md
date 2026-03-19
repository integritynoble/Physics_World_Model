# OCT Working Process

## End-to-End Pipeline for Optical Coherence Tomography

This document traces a complete OCT experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct B-scans from OCT spectral interferograms.
 Data: interferograms.npy, 512 B-scans x 512 A-scans x 1024 spectral samples,
 center wavelength 1060 nm, bandwidth 100 nm."
```

---

## 2. PlanAgent — Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "interferograms.npy" detected
#   operator_type=OperatorType.nonlinear_operator,
#   files=["interferograms.npy"],
#   params={"n_bscans": 512, "n_ascans": 512, "n_spectral": 1024,
#           "center_wavelength_nm": 1060, "bandwidth_nm": 100}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> oct entry
oct:
  keywords: [OCT, interferometry, low_coherence, retinal_imaging, cross_sectional]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="oct",
#   confidence=0.95,
#   reasoning="Matched keywords: OCT, interferometry, B-scan"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the OCT registry entry:

```python
system = plan_agent.build_imaging_system("oct")
# ImagingSystem(
#   modality_key="oct",
#   display_name="Optical Coherence Tomography",
#   signal_dims={"x": [512, 512, 512], "y": [512, 512, 1024]},
#   forward_model_type=ForwardModelType.nonlinear_operator,
#   elements=[...5 elements...],
#   default_solver="fft_recon"
# )
```

**OCT Element Chain (5 elements):**

```
SLD/Swept Source --> Beam Splitter --> Sample Arm (Galvo+Obj) --> Reference Arm --> Spectrometer/Detector
  throughput=1.0     throughput=0.50   throughput=0.75            throughput=0.90   throughput=0.80
  noise: none        noise: none       noise: aberration+align   noise: none        noise: shot+read+thermal
  1060nm center      50:50 coupler     0.05 NA, galvo XY         dispersion comp    1024 spectral samples
  100nm bandwidth    fiber coupler     15 um lateral res          path match 0.1um   105 dB dynamic range
  100 kHz sweep                        6mm scan range                                 14-bit ADC
```

**Cumulative throughput:** `0.50 x 0.75 x 0.90 x 0.80 = 0.270`

**Forward model equation:**
```
y(k) = |E_r + E_s(k)|^2
     = |E_r|^2 + |E_s|^2 + 2*Re(E_r* * E_s(k))
     = DC + autocorrelation + cross-correlation

After balanced detection:
y_balanced(k) = 2*Re(E_r* * E_s(k))  (cross-term only)

A-scan = IFFT(y_balanced) -> depth reflectivity profile
```

---

## 3. PhotonAgent — SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  oct:
    model_id: "interferometric"
    parameters:
      power_w: 0.002
      wavelength_nm: 840
      na: 0.05
      qe: 0.75
      exposure_s: 0.00003
  ```

Note: OCT SNR is fundamentally different from incoherent imaging. The reference arm amplifies the weak sample signal through heterodyne detection: SNR = (eta * P_ref * R_sample) / (h*nu*delta_f), where eta is the detector quantum efficiency, R_sample is sample reflectivity, and delta_f is the detection bandwidth.

### Computation

```python
# 1. Photon energy at 1060 nm (swept source)
E_photon = h * c / wavelength_nm
#        = 6.626e-34 * 3e8 / (1060e-9)
#        = 1.876e-19 J

# 2. Sample arm power
P_sample = power_w * cumulative_throughput_sample
#        = 0.020 * 0.50 * 0.75 = 0.0075 W  (after BS + sample optics)
# Note: modalities.yaml lists source power_mw=20, so power_w=0.020

# 3. Reference arm power
P_ref = 0.020 * 0.50 * 0.90 = 0.009 W

# 4. OCT shot-noise limited SNR (standard formula)
# SNR = (eta * P_sample * R_sample) / (h * nu * delta_f)
# For swept-source: delta_f = sweep_rate / n_spectral = 100e3 / 1024 = 97.7 Hz
# With balanced detection:
R_sample = 1e-4           # Typical tissue reflectivity (-40 dB)
eta = 0.75
P_detected = P_sample * R_sample = 0.0075 * 1e-4 = 7.5e-7 W

# Heterodyne amplification: signal current ~ sqrt(P_ref * P_sample * R)
# SNR_shot = eta * P_sample * R / (h * nu * BW)
BW_per_pixel = 100e3 / 2  # Nyquist: 50 kHz effective
SNR = eta * P_detected / (E_photon * BW_per_pixel)
    = 0.75 * 7.5e-7 / (1.876e-19 * 50e3)
    = 5.625e-7 / 9.38e-15
    = 5.99e7

SNR_db = 10 * log10(5.99e7) = 77.8 dB
# Note: OCT SNR is conventionally reported in 10*log10 (power),
# not 20*log10 (amplitude)

# 5. Sensitivity (minimum detectable reflectivity)
R_min = E_photon * BW_per_pixel / (eta * P_sample)
      = 1.876e-19 * 50e3 / (0.75 * 0.0075)
      = 9.38e-15 / 5.625e-3
      = 1.67e-12  ->  -118 dB (excellent)
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=3.99e7,                 # Photons per A-scan sample
  snr_db=77.8,
  noise_regime=NoiseRegime.shot_limited,      # Balanced detection removes RIN
  shot_noise_sigma=6316,                       # sqrt(N_eff)
  read_noise_sigma=0.0,                        # Negligible with balanced det
  total_noise_sigma=6316,
  feasible=True,
  quality_tier="excellent",                    # SNR >> 30 dB
  throughput_chain=[
    {"SLD / Swept Source": 1.0},
    {"Beam Splitter (Interferometer)": 0.50},
    {"Sample Arm (Scanning + Objective)": 0.75},
    {"Reference Arm": 0.90},
    {"Spectrometer / Balanced Detector": 0.80}
  ],
  noise_model="poisson",
  explanation="Shot-noise-limited OCT with balanced detection. 77.8 dB SNR "
              "corresponds to sensitivity of -118 dB, sufficient for retinal imaging."
)
```

---

## 4. MismatchAgent — Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"oct"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  oct:
    parameters:
      dispersion_mismatch:
        range: [-500.0, 500.0]
        typical_error: 50.0
        unit: "fs^2"
        description: "GVD mismatch between sample and reference arms"
      reference_offset:
        range: [-100.0, 100.0]
        typical_error: 15.0
        unit: "um"
        description: "Reference mirror position error causing depth offset"
      galvo_nonlinearity:
        range: [0.0, 0.05]
        typical_error: 0.01
        unit: "normalized"
        description: "Galvo scanner nonlinearity causing lateral distortion"
    severity_weights:
      dispersion_mismatch: 0.45
      reference_offset: 0.30
      galvo_nonlinearity: 0.25
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.45 * |50.0| / 1000.0      # dispersion:        0.0225
  + 0.30 * |15.0| / 200.0       # reference_offset:  0.0225
  + 0.25 * |0.01| / 0.05        # galvo_nonlin:      0.050
S = 0.095  # Low severity (well-aligned interferometer)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 0.95 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="oct",
  mismatch_family="grid_search",
  parameters={
    "dispersion_mismatch": {"typical_error": 50.0, "range": [-500, 500], "weight": 0.45},
    "reference_offset":    {"typical_error": 15.0, "range": [-100, 100], "weight": 0.30},
    "galvo_nonlinearity":  {"typical_error": 0.01, "range": [0.0, 0.05], "weight": 0.25}
  },
  severity_score=0.095,
  correction_method="grid_search",
  expected_improvement_db=0.95,
  explanation="Low mismatch severity. Galvo nonlinearity is the largest relative error "
              "but all parameters within acceptable range."
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
  oct:
    signal_prior_class: "wavelet_sparse"
    entries:
      - {cr: 1.0, noise: "shot_limited", solver: "fft_oct",
         recoverability: 0.90, expected_psnr_db: 36.2,
         provenance: {dataset_id: "duke_retinal_oct_2023", ...}}
      - {cr: 1.0, noise: "speckle", solver: "fft_oct",
         recoverability: 0.75, expected_psnr_db: 30.4, ...}
      - {cr: 1.0, noise: "shot_limited", solver: "deep_oct_denoiser",
         recoverability: 0.94, expected_psnr_db: 38.5, ...}
  ```

### Computation

```python
# 1. Compression ratio
#    x = (512, 512, 512) = 134,217,728 voxels (3D volume)
#    y = (512, 512, 1024) = 268,435,456 spectral samples
CR = prod(y_shape) / prod(x_shape) = 268435456 / 134217728 = 2.0
# Oversampled: 2x Nyquist in spectral dimension
# Effective CR for lookup: 1.0 (standard OCT, no undersampling)

# 2. Operator properties
#    OCT uses FFT for reconstruction: well-conditioned when properly sampled
#    Speckle noise is the main degradation (multiplicative, not additive)
#    Dispersion and DC terms are calibration issues, not compression issues

# 3. Calibration table lookup
#    Match: noise="shot_limited", solver="fft_oct", cr=1.0
#    -> recoverability=0.90, expected_psnr=36.2 dB

# 4. Best solver selection
#    deep_oct_denoiser: 38.5 dB > fft_oct: 36.2 dB
#    -> recommended: "fft_oct" (standard, reliable)
#    -> upgrade path: "deep_oct_denoiser" for +2.3 dB
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=1.0,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.wavelet_sparse,
  operator_diversity_score=0.9,
  condition_number_proxy=0.474,
  recoverability_score=0.90,
  recoverability_confidence=1.0,
  expected_psnr_db=36.2,
  expected_psnr_uncertainty_db=0.8,
  recommended_solver_family="fft_oct",
  verdict="excellent",                    # score >= 0.85
  calibration_table_entry={...},
  explanation="Excellent recoverability. Standard FFT reconstruction expected "
              "36.2 dB on Duke retinal OCT benchmark. No undersampling."
)
```

---

## 6. AnalysisAgent — Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(77.8 / 40, 1.0)   = 0.0     # Excellent SNR
mismatch_score    = 0.095                        = 0.095   # Low mismatch
compression_score = 1 - 0.90                     = 0.10    # Excellent recoverability
solver_score      = 0.10                         = 0.10    # FFT is well-understood

# Primary bottleneck
primary = "solver"   # max(0.0, 0.095, 0.10, 0.10) = tied; solver chosen
# NOTE: OCT's real bottleneck is speckle, which is a noise issue
# not captured in the photon score (speckle is multiplicative)

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.095*0.5) * (1 - 0.10*0.5) * (1 - 0.10*0.5)
  = 1.0 * 0.953 * 0.95 * 0.95
  = 0.860
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="solver",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.095, compression=0.10, solver=0.10
  ),
  suggestions=[
    Suggestion(
      text="Deep OCT denoiser can improve +2.3 dB over standard FFT",
      priority="medium",
      expected_gain_db=2.3
    ),
    Suggestion(
      text="Speckle averaging (N=4 frames) would improve SNR by ~6 dB",
      priority="medium",
      expected_gain_db=6.0
    )
  ],
  overall_verdict="excellent",           # P >= 0.85
  probability_of_success=0.860,
  explanation="Well-conditioned system. Standard FFT reconstruction is reliable. "
              "Speckle noise is the practical limitation for image quality."
)
```

---

## 7. AgentNegotiator — Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="excellent" | No veto |
| Severe mismatch without correction | severity=0.095 < 0.7 | No veto |
| All marginal | All excellent | No veto |
| Joint probability floor | P=0.860 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.90    # recoverability_score
P_mismatch       = 1.0 - 0.095 * 0.7 = 0.934

P_joint = 0.95 * 0.90 * 0.934 = 0.798
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.798
)
```

---

## 8. PreFlightReportBuilder — Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 512 * 512 * 1024 = 268435456
dim_factor   = total_pixels / (256 * 256) = 4096.0
solver_complexity = 0.5   # FFT (very fast, O(N log N))
cr_factor    = max(1.0, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 4096.0 * 0.5 * 0.125 = 512.0 seconds
# Note: dominated by data I/O for 512x512x1024 volume (~2 GB)
# Actual FFT computation per B-scan: ~0.02s -> 512 B-scans: ~10s
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="oct", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=512.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=[
    "Spectral interferograms (N_bscans x N_ascans x N_spectral)",
    "Reference spectrum for background subtraction (optional)",
    "Dispersion compensation coefficients (optional)"
  ]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# OCT forward model: y(k) = |E_r + E_s(k)|^2
#
# After balanced detection (DC + autocorrelation removed):
# y_bal(k) = 2 * Re(E_r* * E_s(k))
# E_s(k) = sum_z r(z) * exp(2j*k*z)  (backscattered field)
#
# This is a Fourier relationship:
# y_bal(k) <==> r(z) via IFFT
#
# Parameters:
#   n_spectral:   1024 spectral samples per A-scan
#   k_axis:       wavenumber axis (evenly sampled for swept-source)
#   n_depth:      512 (= n_spectral / 2, Nyquist)
#
# Input:  x = (512, 512, 512) tissue reflectivity volume
# Output: y = (512, 512, 1024) spectral interferograms

class OCTOperator(PhysicsOperator):
    def forward(self, x):
        """y(k) = 2*Re(sum_z r(z) * exp(2j*k*z))"""
        k = np.arange(n_spectral) * np.pi / n_spectral
        z = np.arange(n_depth)
        exp_matrix = np.exp(2j * np.outer(k, z))    # (n_spectral, n_depth)
        E_sample = x @ exp_matrix.T                  # (n_bscans, n_ascans, n_spectral)
        y = 2.0 * np.real(E_sample)
        return y

    def adjoint(self, y):
        """x_hat(z) = |IFFT(y(k))| -> A-scan magnitude"""
        depth_profile = np.fft.ifft(y, axis=-1)
        x_hat = np.abs(depth_profile[:, :, :n_depth])
        return x_hat

    def check_adjoint(self):
        """Verify <Ax, y> ~ <x, A*y> for random x, y"""
        # NOTE: OCT is nonlinear (|.|^2), but the balanced-detection
        # model is linear (cross-term only). Adjoint check passes for
        # the linearized model.
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided interferograms.npy:
y = np.load("interferograms.npy")     # (512, 512, 1024)

# If simulating:
# Ground truth: layered tissue phantom with curved retinal layers
x_true = np.zeros((128, 256))        # (n_alines, n_depth)
for layer in range(5):
    base_depth = 30 + layer * 45
    curve = 10 * np.sin(linspace(0, 2*pi, n_alines))
    reflectivity = random() * 0.5 + 0.3
    for a in range(n_alines):
        d = int(base_depth + curve[a])
        sigma_l = random() * 2 + 1
        profile = reflectivity * exp(-0.5 * ((depths - d) / sigma_l)^2)
        x_true[a] += profile

# Spectral interferogram (balanced detection cross-term)
k = np.arange(512) * pi / 512
exp_matrix = np.exp(2j * np.outer(k, z))
E_sample = x_true @ exp_matrix.T
y = 2.0 * np.real(E_sample)
y += np.random.randn(*y.shape) * 0.01  # Detector noise
```

### Step 9c: Reconstruction with FFT

```python
from pwm_core.recon.oct_solver import fft_recon, spectral_estimation

# Algorithm 1: Standard FFT Reconstruction (fast, standard)
recon_fft = fft_recon(
    spectral_data=y,          # (n_alines, n_spectral)
    window="hann",            # Hann window reduces sidelobes
    dc_subtract=True          # Remove DC term
)
# Process: DC subtract -> Hann window -> IFFT -> magnitude -> half-spectrum
# Output: (n_alines, n_depth) B-scan
# Expected PSNR: ~36.2 dB on Duke retinal benchmark

# Algorithm 2: MUSIC/ESPRIT Spectral Estimation (super-resolved)
recon_se = spectral_estimation(
    spectral_data=y,
    n_depth=256,              # Output depth samples
    n_components=10           # Estimated reflecting layers
)
# Eigendecomposition of autocorrelation matrix
# MUSIC pseudo-spectrum for super-resolution depth estimation
# Slower but resolves closely-spaced layers
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| FFT Recon | Traditional | 36.2 dB | No | `fft_recon(y, window="hann")` |
| Spectral Est. | Traditional | 37.0 dB | No | `spectral_estimation(y, n_components=10)` |
| Deep OCT Denoiser | Deep Learning | 38.5 dB | Yes | `deep_oct_denoiser(y, model_path=...)` |

### Step 9d: Metrics

```python
# PSNR
psnr = 10 * log10(max_val^2 / mse(recon, x_true))
# ~36.2 dB (FFT), ~38.5 dB (deep denoiser)

# SSIM
ssim = compute_ssim(recon, x_true)

# OCT-specific metrics:

# Axial resolution (measured from point spread function)
axial_res_um = 0.44 * lambda_c^2 / (n * delta_lambda)
#            = 0.44 * 1.06^2 / (1.37 * 0.1)
#            = 0.494 / 0.137 = 3.6 um (in tissue)

# Signal-to-noise ratio per A-line (dB scale)
snr_per_aline = 10 * log10(peak_signal / noise_floor)

# Speckle contrast ratio
C_speckle = std(region) / mean(region)
# For fully developed speckle: C = 1.0
# After averaging N frames: C = 1/sqrt(N)

# Layer segmentation accuracy (if applicable)
layer_dice = 2 * |seg & gt| / (|seg| + |gt|)
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
|   +-- y.npy              # Spectral interferograms (512, 512, 1024) + SHA256
|   +-- x_hat.npy          # Reconstructed volume (512, 512, 512) + SHA256
|   +-- x_true.npy         # Ground truth (if available) + SHA256
+-- metrics.json           # PSNR, SSIM, SNR/A-line, axial resolution, speckle
+-- operator.json          # Operator params (wavelength, bandwidth, k-axis hash)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized OCT pipeline with balanced detection (clean cross-term), no dispersion mismatch, and low noise. In practice, real OCT systems suffer from dispersion mismatch between the sample and reference arms (broadens the axial PSF), speckle noise (coherent artifact inherent to interferometry), DC and autocorrelation terms (in spectral-domain OCT without balanced detection), and motion artifacts from patient eye movement.

---

## Real Experiment: User Prompt

```
"Retinal OCT scan from our clinical SS-OCT system. There may be
 residual dispersion mismatch — the compensation coefficients were
 set months ago. Single B-scan, no averaging.
 Data: retina_scan.npy, 512 A-scans x 1024 spectral, lambda_c=1060nm."
```

**Key difference:** Dispersion mismatch, single-frame speckle, and no balanced detection assumed.

---

## R1. PlanAgent — Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.operator_correction,   # "dispersion mismatch" detected
#   has_measured_y=True,
#   operator_type=OperatorType.nonlinear_operator,
#   files=["retina_scan.npy"],
#   params={"n_ascans": 512, "n_spectral": 1024,
#           "center_wavelength_nm": 1060}
# )
```

---

## R2. PhotonAgent — Speckle-Dominated Regime

```python
# OCT speckle is not a photon budget issue — it is coherent interference
# from multiple scatterers within each resolution cell.
# With single-frame acquisition:
#   Speckle contrast C = 1.0 (fully developed)
#   Effective SNR degradation: ~5 dB from speckle

SNR_effective = 77.8 - 5.0 = 72.8 dB  # Still high, but speckle limits contrast

PhotonReport(
  n_photons_per_pixel=3.99e7,
  snr_db=72.8,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="excellent",
  explanation="Shot-noise limited but speckle-dominated. Single-frame OCT has "
              "fully developed speckle (C=1.0). Frame averaging recommended."
)
```

---

## R3. MismatchAgent — Dispersion Mismatch

```python
# Stale dispersion compensation coefficients
psi_true = {
    "dispersion_mismatch": +150.0,    # 150 fs^2 GVD mismatch (3x typical)
    "reference_offset":    -30.0,     # 30 um reference drift
    "galvo_nonlinearity":  +0.02,     # 2% nonlinearity
}

# Severity
S = 0.45 * |150.0| / 1000.0    # dispersion:    0.0675
  + 0.30 * |30.0| / 200.0      # ref_offset:    0.045
  + 0.25 * |0.02| / 0.05       # galvo:         0.10
S = 0.213  # Moderate severity

improvement_db = clip(10 * 0.213, 0, 20) = 2.13 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  severity_score=0.213,
  correction_method="grid_search",
  expected_improvement_db=2.13,
  explanation="Moderate mismatch. Dispersion compensation is stale (150 fs^2 "
              "GVD error) causing axial PSF broadening. Re-calibration needed."
)
```

---

## R4. RecoverabilityAgent — Speckle Regime

```python
# Calibration table: noise="speckle", solver="fft_oct", cr=1.0
# -> recoverability=0.75, expected_psnr=30.4 dB

RecoverabilityReport(
  recoverability_score=0.75,
  expected_psnr_db=30.4,
  verdict="good",
  explanation="Speckle reduces effective PSNR by ~5.8 dB compared to shot-limited. "
              "Deep denoiser can partially suppress speckle."
)
```

---

## R5. AnalysisAgent — Speckle is the Bottleneck

```python
photon_score      = 1 - min(72.8 / 40, 1.0)   = 0.0
mismatch_score    = 0.213
compression_score = 1 - 0.75                     = 0.25
solver_score      = 0.10

primary = "compression"  # max(0.0, 0.213, 0.25, 0.10) = compression
# Root cause: speckle noise inflates compression score

P = (1-0.0*0.5) * (1-0.213*0.5) * (1-0.25*0.5) * (1-0.10*0.5)
  = 1.0 * 0.894 * 0.875 * 0.95
  = 0.743
```

```python
SystemAnalysis(
  primary_bottleneck="mismatch",
  probability_of_success=0.743,
  overall_verdict="good",
  suggestions=[
    Suggestion(text="Re-calibrate dispersion compensation coefficients", priority="high"),
    Suggestion(text="Average 4-8 B-scans for speckle reduction (~3-4.5 dB gain)", priority="high"),
    Suggestion(text="Apply deep OCT denoiser for +8.1 dB over speckle-limited FFT", priority="medium"),
  ]
)
```

---

## R6. AgentNegotiator — Proceed

```python
P_joint = 0.95 * 0.75 * (1 - 0.213*0.7) = 0.95 * 0.75 * 0.851 = 0.606

NegotiationResult(
  vetoes=[],
  proceed=True,
  probability_of_success=0.606
)
```

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=65.0,       # Includes dispersion grid search
  proceed_recommended=True,
  warnings=[
    "Dispersion mismatch 150 fs^2 — axial resolution degraded from 3.6 um to ~7 um",
    "Single-frame speckle (C=1.0) limits contrast; averaging recommended",
    "Estimated PSNR 30.4 dB (speckle regime) vs 36.2 dB (shot-limited)"
  ]
)
```

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Clinical |
|--------|-----------|---------------|
| **Photon Agent** | | |
| SNR | 77.8 dB | 72.8 dB |
| Quality tier | excellent | excellent |
| Noise regime | shot_limited | shot_limited (speckle) |
| **Mismatch Agent** | | |
| Severity | 0.095 (low) | 0.213 (moderate) |
| Dominant error | galvo nonlin | **dispersion 150 fs^2** |
| Expected gain | +0.95 dB | +2.13 dB |
| **Recoverability Agent** | | |
| Score | 0.90 (excellent) | 0.75 (good) |
| Expected PSNR | 36.2 dB | 30.4 dB |
| Verdict | excellent | good |
| **Analysis Agent** | | |
| Primary bottleneck | solver | **mismatch** |
| P(success) | 0.860 | 0.743 |
| **Negotiator** | | |
| P_joint | 0.798 | 0.606 |
| **PreFlight** | | |
| Runtime | 512s | 65s |
| Warnings | 0 | 3 |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (FFT -> spectral estimation -> deep denoiser) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Adaptive:** OCT-specific awareness of speckle noise, dispersion mismatch, and interferometric detection physics.
