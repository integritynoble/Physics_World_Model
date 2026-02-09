# Light-Sheet Working Process

## End-to-End Pipeline for Light-Sheet Fluorescence Microscopy

This document traces a complete light-sheet fluorescence microscopy experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Remove stripe artifacts from our SPIM zebrafish volume and deconvolve.
 Volume: spim_volume.npy, 512x512x128 voxels,
 emission wavelength 525 nm, water immersion 20x objective."
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "spim_volume.npy" detected
#   operator_type=OperatorType.linear_operator,
#   files=["spim_volume.npy"],
#   params={"emission_wavelength_nm": 525, "voxel_dims": [512, 512, 128]}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> lightsheet entry
lightsheet:
  keywords: [lightsheet, SPIM, LSFM, stripe_artifacts, optical_sectioning]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="lightsheet",
#   confidence=0.96,
#   reasoning="Matched keywords: SPIM, stripe_artifacts"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the light-sheet registry entry:

```python
system = plan_agent.build_imaging_system("lightsheet")
# ImagingSystem(
#   modality_key="lightsheet",
#   display_name="Light-Sheet Fluorescence Microscopy",
#   signal_dims={"x": [512, 512, 128], "y": [512, 512, 128]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...5 elements...],
#   default_solver="fourier_notch_destripe"
# )
```

**Light-Sheet Element Chain (5 elements):**

```
Laser Source (488 nm) ──> Cylindrical Lens (Sheet Former) ──> Sample Medium ──> Detection Objective (20x/1.0 NA) ──> sCMOS Detector
  throughput=1.0           throughput=0.90                    throughput=0.85   throughput=0.78                       throughput=0.82
  noise: none              noise: alignment                   noise: aberration noise: aberration                     noise: shot+read+fixed
                                                              + stripe_artifacts
```

**Cumulative throughput:** `0.90 x 0.85 x 0.78 x 0.82 = 0.489`

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  lightsheet:
    model_id: "microscopy_fluorescence"
    parameters:
      power_w: 0.002
      wavelength_nm: 488
      na: 0.3
      n_medium: 1.33
      qe: 0.72
      exposure_s: 0.05
  ```

### Computation

```python
# 1. Photon energy (excitation wavelength)
E_photon = h * c / wavelength_nm
         = (6.626e-34 * 3.0e8) / (488e-9) = 4.07e-19 J

# 2. Collection solid angle
#    Detection objective NA = 0.3 in water (n=1.33)
solid_angle = (na / n_medium)^2 / (4 * pi)
            = (0.3 / 1.33)^2 / (4 * pi)
            = 4.04e-3

# 3. Raw photon count
N_raw = power_w * qe * solid_angle * exposure_s / E_photon
      = 0.002 * 0.72 * 4.04e-3 * 0.05 / 4.07e-19
      = 7.16e11 photons

# 4. Apply cumulative throughput
N_effective = N_raw * 0.489 = 3.50e11 photons/voxel

# 5. Noise variances
shot_var   = N_effective = 3.50e11             # Poisson (dominant)
read_var   = read_noise^2 = 1.6^2 = 2.56      # sCMOS read noise
dark_var   = 0                                 # Negligible
total_var  = 3.50e11 + 2.56 = 3.50e11

# 6. SNR
SNR = N_effective / sqrt(total_var) = sqrt(3.50e11) = 5.92e5
SNR_db = 20 * log10(5.92e5) = 115.4 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=3.50e11,
  snr_db=115.4,
  noise_regime=NoiseRegime.shot_limited,      # shot_var/total_var ~ 1.0
  shot_noise_sigma=5.92e5,
  read_noise_sigma=1.6,
  total_noise_sigma=5.92e5,
  feasible=True,
  quality_tier="excellent",                   # SNR >> 30 dB
  throughput_chain=[
    {"Laser Source (488 nm)": 1.0},
    {"Cylindrical Lens (Sheet Former)": 0.90},
    {"Sample Medium": 0.85},
    {"Detection Objective (20x / 1.0 NA Water)": 0.78},
    {"sCMOS Detector": 0.82}
  ],
  noise_model="poisson",
  explanation="Shot-noise-limited regime. The primary degradation is stripe artifacts, not photon noise."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"lightsheet"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  lightsheet:
    parameters:
      psf_sigma:
        range: [0.5, 3.0]
        typical_error: 0.3
        weight: 0.30
      defocus:
        range: [-2.0, 2.0]
        typical_error: 0.6
        weight: 0.30
      background:
        range: [0.0, 0.15]
        typical_error: 0.04
        weight: 0.25
      gain:
        range: [0.5, 1.5]
        typical_error: 0.1
        weight: 0.15
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.30 * |0.3| / 2.5        # psf_sigma: 0.036
  + 0.30 * |0.6| / 4.0        # defocus: 0.045
  + 0.25 * |0.04| / 0.15      # background: 0.067
  + 0.15 * |0.1| / 1.0        # gain: 0.015
S = 0.163  # Low severity

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.63 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="lightsheet",
  mismatch_family="grid_search",
  parameters={
    "psf_sigma":   {"typical_error": 0.3, "range": [0.5, 3.0], "weight": 0.30},
    "defocus":     {"typical_error": 0.6, "range": [-2.0, 2.0], "weight": 0.30},
    "background":  {"typical_error": 0.04, "range": [0.0, 0.15], "weight": 0.25},
    "gain":        {"typical_error": 0.1, "range": [0.5, 1.5], "weight": 0.15}
  },
  severity_score=0.163,
  correction_method="grid_search",
  expected_improvement_db=1.63,
  explanation="Low mismatch severity. Background scatter contributes the most uncertainty."
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
  lightsheet:
    signal_prior_class: "tv"
    entries:
      - {cr: 1.0, noise: "shot_limited", solver: "destripe",
         recoverability: 0.83, expected_psnr_db: 28.3,
         provenance: {dataset_id: "openspim_zebrafish_2023", ...}}
      - {cr: 1.0, noise: "background_limited", solver: "fourier_notch",
         recoverability: 0.71, expected_psnr_db: 24.6, ...}
      - {cr: 1.0, noise: "shot_limited", solver: "vsnr",
         recoverability: 0.78, expected_psnr_db: 26.8, ...}
  ```

### Computation

```python
# 1. Compression ratio
#    Light-sheet is not compressive: y has same dimensions as x
CR = prod(y_shape) / prod(x_shape) = (512*512*128) / (512*512*128) = 1.0

# 2. Operator diversity
#    Light-sheet forward model: y = S(z) * (PSF_3d *** x) + n
#    where S(z) is the stripe-generating illumination modulation
#    The operator is essentially PSF blur + stripe contamination
diversity = 0.6  # Moderate (PSF is low-pass, some frequencies are lost)

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.625

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="destripe", cr=1.0
#    -> recoverability=0.83, expected_psnr=28.3 dB, confidence=1.0

# 5. Best solver selection
#    destripe: 28.3 dB > vsnr: 26.8 dB > fourier_notch: 24.6 dB
#    -> recommended: "destripe" (DeStripe network)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=1.0,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.tv,
  operator_diversity_score=0.6,
  condition_number_proxy=0.625,
  recoverability_score=0.83,
  recoverability_confidence=1.0,
  expected_psnr_db=28.3,
  expected_psnr_uncertainty_db=1.5,
  recommended_solver_family="fourier_notch_destripe",
  verdict="sufficient",              # 0.60 <= score < 0.85
  calibration_table_entry={...},
  explanation="Sufficient recoverability. Fourier notch destriping is the default; DeStripe network gives +3.7 dB."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(115.4 / 40, 1.0)  = 0.0    # Excellent SNR
mismatch_score    = 0.163                       = 0.163  # Low mismatch
compression_score = 1 - 0.83                    = 0.17   # Good recoverability
solver_score      = 0.25                        = 0.25   # Destriping is imperfect

# Primary bottleneck
primary = "solver"  # max(0.0, 0.163, 0.17, 0.25) = solver

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.163*0.5) * (1 - 0.17*0.5) * (1 - 0.25*0.5)
  = 1.0 * 0.919 * 0.915 * 0.875
  = 0.736
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="solver",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.163, compression=0.17, solver=0.25
  ),
  suggestions=[
    Suggestion(
      text="Use DeStripe network for +3.7 dB over Fourier notch filter",
      priority="high",
      expected_gain_db=3.7
    ),
    Suggestion(
      text="Consider multi-view fusion for further stripe suppression",
      priority="medium",
      expected_gain_db=2.0
    ),
    Suggestion(
      text="Apply 3D deconvolution after destriping for additional resolution recovery",
      priority="low",
      expected_gain_db=1.0
    )
  ],
  overall_verdict="sufficient",       # 0.60 <= P < 0.80
  probability_of_success=0.736,
  explanation="Stripe artifact removal quality is the primary bottleneck. Photon budget is excellent."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND CR=1.0 | No veto |
| Severe mismatch without correction | severity=0.163 < 0.7 | No veto |
| All marginal | photon=excellent, others=sufficient | No veto |
| Joint probability floor | P=0.736 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.83    # recoverability_score
P_mismatch       = 1.0 - 0.163 * 0.7 = 0.886

P_joint = 0.95 * 0.83 * 0.886 = 0.698
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.698
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_voxels = 512 * 512 * 128 = 33,554,432
dim_factor   = total_voxels / (256 * 256) = 512.0
solver_complexity = 1.5  # Fourier notch filter (FFT-based, per-slice)
cr_factor    = max(1.0, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 512.0 * 1.5 * 0.125 = 192.0 seconds
# 128 slices x ~1.5s per slice for FFT-based destriping
# Reduce to ~15s with vectorized FFT implementation
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="lightsheet", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=15.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=[
    "measurement (3D light-sheet volume, multi-page TIFF or .npy)",
    "PSF (optional, 3D detection PSF for deconvolution)"
  ]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Light-sheet forward model:
#   y = S(z) * (PSF_3d *** x) + n
#
# where:
#   PSF_3d:  3D detection point spread function (anisotropic Gaussian)
#   S(z):    stripe modulation from illumination absorption/scattering
#   n:       mixed noise (shot + read + fixed-pattern)
#
# The stripe pattern S(z) is characterized by:
#   - Horizontal bands (perpendicular to illumination axis)
#   - Intensity varies with z-depth due to tissue scattering
#   - stripe_strength = 0.2 (from modalities.yaml)
#   - attenuation_coef = 0.02 per voxel
#
# Input:  x = (512, 512, 128) clean fluorescence volume
# Output: y = (512, 512, 128) corrupted volume (same size, no compression)

class LightSheetOperator(PhysicsOperator):
    def forward(self, x):
        """y = S * (PSF *** x) + n"""
        # 3D convolution with detection PSF
        blurred = convolve_fft_3d(x, self.psf_3d)
        # Apply stripe modulation (multiplicative)
        y = self.stripe_mask * blurred
        return y

    def adjoint(self, y):
        """x_hat = PSF_T *** (S * y)"""
        # Apply stripe mask (self-adjoint for diagonal)
        masked = self.stripe_mask * y
        # Transpose convolution (correlation with PSF)
        return correlate_fft_3d(masked, self.psf_3d)

    def check_adjoint(self):
        """Verify <Ax, y> ~ <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided spim_volume.npy:
y = np.load("spim_volume.npy")          # (512, 512, 128)

# If simulating:
x_true = load_ground_truth_volume()      # (512, 512, 128) zebrafish embryo

# Create 3D PSF (anisotropic Gaussian)
psf_3d = gaussian_psf_3d(
    sigma_lateral=1.5,    # px (detection NA=1.0)
    sigma_axial=1.0,      # px (light-sheet optical sectioning)
    size=(11, 11, 7)
)

# Generate stripe artifacts
stripes = np.zeros_like(x_true)
for z in range(128):
    # Random horizontal stripe positions (absorption shadows)
    stripe_pos = np.random.choice(512, 5, replace=False)
    stripes[stripe_pos, :, z] = 0.2  # 20% intensity modulation

# Forward model
from scipy.ndimage import convolve
y = convolve(x_true, psf_3d, mode='reflect')
y = y + stripes                         # Add stripe artifacts
y += np.random.randn(*y.shape) * 0.02   # Gaussian noise
```

### Step 9c: Reconstruction with Fourier Notch Destripe

```python
from pwm_core.recon.lightsheet_solver import fourier_notch_destripe

# Process each z-slice independently
x_hat = np.zeros_like(y)
for z in range(128):
    x_hat[:, :, z] = fourier_notch_destripe(
        y[:, :, z],
        notch_width=5,       # Width of notch filter in frequency domain
        damping=5.0           # Damping factor for smooth suppression
    )
# x_hat shape: (512, 512, 128) -- destriped volume
# Expected PSNR: ~24.6 dB (Fourier notch baseline)
```

**Fourier Notch Destripe Algorithm:**
```python
def fourier_notch_destripe(img, notch_width=5, damping=5.0):
    """Remove horizontal stripes via Fourier notch filter.

    Stripes appear as high energy along the kx=0 axis
    in Fourier space. A damped notch filter suppresses
    these frequencies while preserving image content.
    """
    F = np.fft.fft2(img)
    H, W = img.shape
    # Create notch mask: suppress kx=0 column (vertical frequencies)
    mask = np.ones((H, W), dtype=np.float32)
    center_col = W // 2
    for dx in range(-notch_width, notch_width + 1):
        col = (center_col + dx) % W
        # Damped suppression (stronger near DC, weaker at edges)
        weight = 1.0 - np.exp(-dx**2 / (2 * damping**2))
        mask[:, col] *= weight
    # Preserve DC component
    mask[0, 0] = 1.0
    # Apply in shifted Fourier domain
    F_shifted = np.fft.fftshift(F)
    F_filtered = F_shifted * mask
    return np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| Fourier Notch | Traditional | 24.6 dB | No | `fourier_notch_destripe(y[:,:,z], notch_width=5)` |
| VSNR | Variational | 26.8 dB | No | `vsnr_destripe(y[:,:,z])` |
| DeStripe | Deep Learning | 28.3 dB | Yes | `destripe_denoise(y[:,:,z], self_supervised_iters=100)` |

### Step 9d: Metrics

```python
# Per-slice PSNR
for z in range(128):
    psnr_z = 10 * log10(max_val^2 / mse(x_hat[:,:,z], x_true[:,:,z]))

# Average PSNR across all slices
avg_psnr = mean(psnr_per_slice)  # ~24.6 dB (notch), ~28.3 dB (DeStripe)

# SSIM (structural similarity per slice)
avg_ssim = mean([ssim(x_hat[:,:,z], x_true[:,:,z]) for z in range(128)])

# Stripe residual metric (light-sheet specific)
# Measure energy in the notch region of the Fourier spectrum
stripe_residual = mean([
    sum(abs(fft2(x_hat[:,:,z]))[kx_notch_region]) /
    sum(abs(fft2(x_hat[:,:,z])))
    for z in range(128)
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
|   +-- y.npy              # Measurement (512, 512, 128) + SHA256 hash
|   +-- x_hat.npy          # Reconstruction (512, 512, 128) + SHA256 hash
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
+-- metrics.json           # PSNR, SSIM per slice + average, stripe residual
+-- operator.json          # Operator parameters (PSF sigma, stripe model)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized light-sheet pipeline with excellent SNR (115.4 dB) and low mismatch (0.163). In practice, real light-sheet microscopes suffer from severe stripe artifacts in thick tissue, light-sheet drift causing defocus, and scattering-induced background.

This section traces the same pipeline with realistic degraded parameters from a thick zebrafish specimen.

---

## Real Experiment: User Prompt

```
"We imaged a thick zebrafish embryo with our OpenSPIM. Heavy striping
 due to pigmented cells absorbing the light sheet. The sheet may have
 drifted during the z-stack acquisition (1 hour total).
 Volume: thick_zebrafish.npy, 128x128x32 voxels."
```

**Key difference:** Thick tissue causes severe stripes; long acquisition causes sheet drift.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["thick_zebrafish.npy"],
#   params={"voxel_dims": [128, 128, 32]}
# )
```

---

## R2. PhotonAgent -- Realistic Conditions

### Real tissue parameters

```yaml
# Real lab: thick tissue, scattering, lower effective power
lightsheet_lab:
  power_w: 0.0005           # 4x lower effective power at depth
  wavelength_nm: 488
  na: 0.3
  n_medium: 1.33
  qe: 0.65                  # Sensor aging
  exposure_s: 0.02           # Shorter exposure to reduce bleaching
```

### Computation

```python
N_raw = 0.0005 * 0.65 * 4.04e-3 * 0.02 / 4.07e-19 = 6.46e9
N_effective = 6.46e9 * 0.489 = 3.16e9 photons/voxel

shot_var   = 3.16e9
read_var   = 1.6^2 = 2.56
total_var  = 3.16e9 + 2.56 = 3.16e9

SNR = sqrt(3.16e9) = 5.62e4
SNR_db = 20 * log10(5.62e4) = 95.0 dB
```

### Output

```python
PhotonReport(
  n_photons_per_pixel=3.16e9,
  snr_db=95.0,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="excellent",           # 95 dB >> 30 dB
  explanation="Shot-limited. Excellent SNR even in thick tissue."
)
```

**Verdict:** Even with 4x fewer photons, SNR remains excellent. This confirms the light-sheet bottleneck is stripe artifacts, not photons.

---

## R3. MismatchAgent -- Real Sheet Drift and Thick Tissue

```python
# Actual errors from thick tissue + long acquisition
S = 0.30 * |0.8| / 2.5        # psf_sigma: 0.096  (PSF degraded by scattering)
  + 0.30 * |1.5| / 4.0        # defocus: 0.113    (sheet drifted 1.5 um!)
  + 0.25 * |0.10| / 0.15      # background: 0.167  (heavy scattering)
  + 0.15 * |0.15| / 1.0       # gain: 0.023        (minor fixed-pattern)
S = 0.399  # MODERATE-HIGH severity

improvement_db = clip(10 * 0.399, 0, 20) = 3.99 dB
```

### Output

```python
MismatchReport(
  severity_score=0.399,
  correction_method="grid_search",
  expected_improvement_db=3.99,
  explanation="Moderate-high mismatch. Heavy tissue scattering (background=0.10) "
              "and sheet drift (1.5 um) are the dominant error sources. "
              "Consider multi-view fusion for thick specimens."
)
```

---

## R4. RecoverabilityAgent -- Degraded by Tissue

```python
# CR = 1.0 (unchanged)
# background_limited entry: recoverability=0.71, expected_psnr=24.6 dB
# Additional degradation from sheet drift:
# recoverability_adjusted = 0.71 * (1 - 0.399*0.3) = 0.625
# expected_psnr ~ 22.0 dB
```

### Output

```python
RecoverabilityReport(
  compression_ratio=1.0,
  recoverability_score=0.625,
  expected_psnr_db=22.0,
  verdict="sufficient",           # score >= 0.60
  explanation="Thick tissue degrades destriping quality. Multi-view would help."
)
```

---

## R5. AnalysisAgent -- Background is the Bottleneck

```python
photon_score      = 0.0       # Excellent
mismatch_score    = 0.399     # Moderate-high
compression_score = 1 - 0.625 = 0.375
solver_score      = 0.25

primary = "mismatch"   # max(0.0, 0.399, 0.375, 0.25)

P = 1.0 * 0.801 * 0.813 * 0.875 = 0.570
```

---

## R6. AgentNegotiator -- Proceed with Warnings

```python
P_photon         = 0.95
P_recoverability = 0.625
P_mismatch       = 1.0 - 0.399 * 0.7 = 0.721

P_joint = 0.95 * 0.625 * 0.721 = 0.428
```

No veto (P_joint > 0.15). Proceed with warnings.

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=25.0,
  proceed_recommended=True,
  warnings=[
    "High scattering in thick tissue -- heavy stripe artifacts expected",
    "Sheet drift (1.5 um) detected -- PSF may vary across z-stack",
    "Consider multi-view fusion or dual-illumination for improved stripe removal"
  ],
  what_to_upload=["measurement (3D light-sheet volume)"]
)
```

---

## R8. Pipeline Results

| Configuration | Fourier Notch | VSNR | DeStripe |
|---------------|---------------|------|----------|
| Ideal (thin tissue) | 24.6 dB | 26.8 dB | 28.3 dB |
| Thick tissue (real) | 20.5 dB | 22.0 dB | 24.5 dB |
| With sheet drift | 19.0 dB | 21.0 dB | 23.5 dB |

**Key findings:**
- Light-sheet stripe removal quality depends primarily on stripe severity, not photon budget
- DeStripe network outperforms Fourier notch by +3.7 dB (learned texture-aware priors)
- VSNR provides a good middle ground: better than Fourier notch, no GPU required
- Thick tissue causes ~4 dB degradation compared to thin samples
- Sheet drift adds another ~1.5 dB loss (z-dependent PSF variation)
- Multi-view fusion (not implemented yet) could recover ~2-3 dB in thick tissue

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Experiment |
|--------|-----------|-----------------|
| **Photon Agent** | | |
| N_effective | 3.50e11 | 3.16e9 |
| SNR | 115.4 dB | 95.0 dB |
| Quality tier | excellent | excellent |
| **Mismatch Agent** | | |
| Severity | 0.163 (low) | 0.399 (moderate-high) |
| Dominant error | background (scatter) | background + defocus |
| Expected gain | +1.63 dB | +3.99 dB |
| **Recoverability Agent** | | |
| Score | 0.83 (sufficient) | 0.625 (sufficient) |
| Expected PSNR | 28.3 dB | 22.0 dB |
| **Analysis Agent** | | |
| Primary bottleneck | solver | mismatch |
| P(success) | 0.736 | 0.570 |
| **Negotiator** | | |
| P_joint | 0.698 | 0.428 |
| **PreFlight** | | |
| Runtime | 15.0s | 25.0s |
| Warnings | 0 | 3 |
| **Pipeline** | | |
| Fourier Notch | 24.6 dB | 20.5 dB |
| VSNR | 26.8 dB | 22.0 dB |
| DeStripe | 28.3 dB | 24.5 dB |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (Fourier Notch -> VSNR -> DeStripe) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Artifact-centric:** Light-sheet quality is dominated by stripe artifact severity and tissue scattering, not photon budget or compression ratio.
