# Confocal 3D Z-Stack Working Process

## End-to-End Pipeline for Volumetric Confocal Deconvolution

This document traces a complete confocal 3D z-stack experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Deconvolve this confocal z-stack of mitochondria (MitoTracker Red).
 63x/1.4 NA oil objective, 561 nm laser, 64 z-slices at 0.3 um spacing.
 Measurement: mito_zstack.tif"
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "mito_zstack.tif" detected
#   operator_type=OperatorType.linear_operator,
#   files=["mito_zstack.tif"],
#   params={"excitation_wavelength_nm": 561, "numerical_aperture": 1.4,
#           "magnification": 63, "n_slices": 64, "z_step_um": 0.3}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> confocal_3d entry
confocal_3d:
  keywords: [confocal, 3d, z_stack, volumetric, deconvolution]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="confocal_3d",
#   confidence=0.96,
#   reasoning="Matched keywords: confocal, 3d, z_stack, deconvolution"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the confocal_3d registry entry:

```python
system = plan_agent.build_imaging_system("confocal_3d")
# ImagingSystem(
#   modality_key="confocal_3d",
#   display_name="Confocal 3D Z-Stack",
#   signal_dims={"x": [256, 256, 64], "y": [256, 256, 64]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...5 elements...],
#   default_solver="richardson_lucy_3d"
# )
```

**Confocal 3D Element Chain (5 elements):**

```
Laser Source (561 nm) --> Scanning Mirrors (Galvo) --> Objective Lens (63x/1.4 NA Oil) --> Confocal Pinhole (1 AU) --> PMT Detector
  throughput=1.0            throughput=0.92              throughput=0.75                      throughput=0.70           throughput=0.25
  noise: none               noise: alignment             noise: aberration                    noise: alignment          noise: shot+read+thermal
  power_mw=3                z_step_um=0.3                immersion=oil
                            scan_rate=8000 Hz            psf_sigma_lat=1.0 px
                                                         psf_sigma_ax=3.5 px
```

**Cumulative throughput:** `0.92 x 0.75 x 0.70 x 0.25 = 0.121`

**Key 3D physics:** The confocal PSF is highly anisotropic in 3D:
- **Lateral resolution:** sigma_lat = 1.0 px (~100 nm at 63x mag)
- **Axial resolution:** sigma_ax = 3.5 px (~350 nm effective)
- **Axial/lateral ratio:** 3.5x -- the PSF is elongated along z, creating the well-known "missing cone" problem in Fourier space.

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  confocal_3d:
    model_id: "microscopy_fluorescence"
    parameters:
      power_w: 0.001
      wavelength_nm: 488
      na: 1.2
      n_medium: 1.33
      qe: 0.45
      exposure_s: 0.01
  ```

### Computation

```python
# 1. Photon energy (using actual laser wavelength 561 nm for this experiment)
E_photon = h * c / wavelength_nm
         = (6.626e-34 * 3e8) / (561e-9)
         = 3.544e-19 J

# 2. Collection solid angle (water immersion objective in registry)
solid_angle = (na / n_medium)^2 / (4 * pi)
            = (1.2 / 1.33)^2 / (4 * pi)
            = 0.8133 / 12.566
            = 0.0647

# 3. Raw photon count per voxel
#    For z-stacks, exposure_s is the dwell time per voxel
N_raw = power_w * qe * solid_angle * exposure_s / E_photon
      = 0.001 * 0.45 * 0.0647 * 0.01 / 3.544e-19
      = 8.21e11 photons

# 4. Apply cumulative throughput (0.121)
N_effective = 8.21e11 * 0.121 = 9.93e10 photons/voxel

# 5. Noise variances
shot_var   = N_effective = 9.93e10             # Poisson
dark_var   = 50 * 0.01 = 0.5                   # dark_current_cps * exposure
thermal_var = 100                               # PMT thermal
total_var  = 9.93e10 + 0.5 + 100

# 6. SNR
SNR = N_effective / sqrt(total_var) ~ sqrt(N_effective) = 3.15e5
SNR_db = 20 * log10(3.15e5) = 110.0 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=9.93e10,
  snr_db=110.0,
  noise_regime=NoiseRegime.shot_limited,    # shot_var/total_var >> 0.9
  shot_noise_sigma=3.15e5,
  read_noise_sigma=0.0,                     # PMT
  total_noise_sigma=3.15e5,
  feasible=True,
  quality_tier="excellent",                 # SNR > 30 dB
  throughput_chain=[
    {"Laser Source (561 nm)": 1.0},
    {"Scanning Mirrors (Galvo)": 0.92},
    {"Objective Lens (63x/1.4 NA Oil)": 0.75},
    {"Confocal Pinhole (1 AU)": 0.70},
    {"PMT Detector": 0.25}
  ],
  noise_model="poisson",
  explanation="Shot-noise-limited. Excellent SNR per voxel for 3D deconvolution. "
              "Total acquisition time for 64-slice stack: ~34 minutes at 512x512."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"confocal_3d"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  confocal_3d:
    parameters:
      psf_sigma:   {range: [0.3, 2.5], typical_error: 0.25, weight: 0.30}
      defocus:     {range: [-3.0, 3.0], typical_error: 0.8, weight: 0.35}
      background:  {range: [0.0, 0.12], typical_error: 0.03, weight: 0.20}
      gain:        {range: [0.5, 1.5], typical_error: 0.1, weight: 0.15}
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.30 * |0.25| / 2.2    # psf_sigma:  0.034
  + 0.35 * |0.8| / 6.0     # defocus:    0.047
  + 0.20 * |0.03| / 0.12   # background: 0.050
  + 0.15 * |0.1| / 1.0     # gain:       0.015
S = 0.146  # Low-moderate severity

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.46 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="confocal_3d",
  mismatch_family="grid_search",
  parameters={
    "psf_sigma":  {"typical_error": 0.25, "range": [0.3, 2.5], "weight": 0.30},
    "defocus":    {"typical_error": 0.8, "range": [-3.0, 3.0], "weight": 0.35},
    "background": {"typical_error": 0.03, "range": [0.0, 0.12], "weight": 0.20},
    "gain":       {"typical_error": 0.1, "range": [0.5, 1.5], "weight": 0.15}
  },
  severity_score=0.146,
  correction_method="grid_search",
  expected_improvement_db=1.46,
  explanation="Low-moderate mismatch. Depth-dependent spherical aberration (defocus) is "
              "the primary concern for 3D stacks -- the PSF changes with depth."
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
  confocal_3d:
    signal_prior_class: "tv"
    entries:
      - {cr: 1.0, noise: "shot_limited", solver: "richardson_lucy_3d",
         recoverability: 0.85, expected_psnr_db: 29.2,
         provenance: {dataset_id: "bii_confocal_3d_mito_2023", ...}}
      - {cr: 1.0, noise: "shot_limited", solver: "care_restore_3d",
         recoverability: 0.92, expected_psnr_db: 33.1, ...}
  ```

### Computation

```python
# 1. Compression ratio
CR = prod(y_shape) / prod(x_shape) = (256 * 256 * 64) / (256 * 256 * 64) = 1.0
# No compression -- 3D confocal is a voxel-to-voxel mapping.
# The challenge is 3D deconvolution with an anisotropic PSF.

# 2. Operator diversity
# The 3D confocal PSF has a "missing cone" in Fourier space:
# Frequencies near the kz axis are not transferred, creating
# axial elongation artifacts. This limits diversity.
diversity = 0.55  # Lower than 2D confocal due to missing cone

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.645

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="richardson_lucy_3d", cr=1.0
#    -> recoverability=0.85, expected_psnr=29.2 dB

# 5. Best solver selection
#    care_restore_3d: 33.1 dB > richardson_lucy_3d: 29.2 dB
#    -> recommended: "richardson_lucy_3d" (default) or "care_restore_3d" (best)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=1.0,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.tv,
  operator_diversity_score=0.55,
  condition_number_proxy=0.645,
  recoverability_score=0.85,
  recoverability_confidence=1.0,
  expected_psnr_db=29.2,
  expected_psnr_uncertainty_db=0.7,
  recommended_solver_family="richardson_lucy_3d",
  verdict="excellent",              # score >= 0.85
  calibration_table_entry={...},
  explanation="Excellent recoverability. 3D RL expected 29.2 dB; CARE-3D reaches 33.1 dB "
              "on BII 3D mitochondria benchmark. Missing cone limits axial resolution."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(110.0 / 40, 1.0)  = 0.0    # Excellent SNR
mismatch_score    = 0.146                       = 0.146  # Low-moderate
compression_score = 1 - 0.85                    = 0.15   # Excellent recoverability
solver_score      = 0.2                         = 0.2    # Default

# Primary bottleneck
primary = "solver"  # max(0.0, 0.146, 0.15, 0.2) = solver

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.146*0.5) * (1 - 0.15*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.927 * 0.925 * 0.90
  = 0.771
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="solver",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.146, compression=0.15, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Use CARE-3D for +3.9 dB over 3D Richardson-Lucy",
      priority="medium",
      expected_gain_db=3.9
    ),
    Suggestion(
      text="Measure the 3D PSF with sub-diffraction beads for accurate axial deconvolution",
      priority="medium",
      expected_gain_db=1.2
    ),
    Suggestion(
      text="Consider depth-dependent PSF model for thick specimens (>10 um)",
      priority="low",
      expected_gain_db=0.8
    )
  ],
  overall_verdict="good",           # P >= 0.70
  probability_of_success=0.771,
  explanation="System well-configured for 3D deconvolution. Solver choice is the primary "
              "lever. The missing cone in 3D confocal is the fundamental limitation."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND CR=1.0 | No veto |
| Severe mismatch without correction | severity=0.146 < 0.7 | No veto |
| All marginal | All excellent/good | No veto |
| Joint probability floor | P=0.771 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95   # tier_prob["excellent"]
P_recoverability = 0.85   # recoverability_score
P_mismatch       = 1.0 - 0.146 * 0.7 = 0.898

P_joint = 0.95 * 0.85 * 0.898 = 0.725
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.725
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_voxels = 256 * 256 * 64 = 4,194,304
dim_factor   = total_voxels / (256 * 256) = 64.0
solver_complexity = 1.5  # 3D RL (iterative, CPU, 3D FFT is expensive)
cr_factor    = max(1.0, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 64.0 * 1.5 * 0.125 = 24.0 seconds
# 3D RL with 40 iterations on 256x256x64: ~20-30 seconds on modern CPU
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="confocal_3d", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=24.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=["measurement (3D confocal z-stack, multi-page TIFF or .npy)"]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Confocal 3D forward model: y(x,y,z) = PSF_3d(x,y,z) *** x(x,y,z) + n
#
# The 3D confocal PSF is the product of excitation and emission PSFs:
#   h_3d(x,y,z) = h_exc(x,y,z) * h_det(x,y,z)
#
# Modelled as an anisotropic 3D Gaussian:
#   h_3d(r,z) = exp(-r^2/(2*sigma_lat^2)) * exp(-z^2/(2*sigma_ax^2))
#
# Parameters:
#   sigma_lat = 1.0 px (lateral, ~100 nm at 63x)
#   sigma_ax  = 3.5 px (axial, ~350 nm effective, limited by missing cone)
#   z_step    = 0.3 um (Nyquist sampling for axial resolution ~600 nm)
#
# The 3D PSF in Fourier space (OTF) has a "missing cone":
#   Frequencies where |kz| > 2*NA/lambda are zero.
#   This means axial frequencies beyond the cutoff cannot be recovered,
#   regardless of deconvolution algorithm.
#
# Input:  x = (256, 256, 64) volumetric fluorescence distribution
# Output: y = (256, 256, 64) blurred z-stack

class Confocal3DOperator(PhysicsOperator):
    def forward(self, x):
        """y = PSF_3d *** x (3D convolution)"""
        return fftconvolve(x, self.psf_3d, mode='same')

    def adjoint(self, y):
        """x_hat = PSF_3d_flipped *** y (3D correlation)"""
        return fftconvolve(y, self.psf_3d[::-1, ::-1, ::-1], mode='same')

    def check_adjoint(self):
        """Verify <Ax, y> ~ <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided measurement:
y = load_tiff_stack("mito_zstack.tif")    # (256, 256, 64)

# If simulating:
x_true = load_ground_truth_3d()            # (256, 256, 64) from BII dataset
y = operator.forward(x_true)               # (256, 256, 64) blurred
y += np.random.randn(*y.shape) * 0.03      # Additive Gaussian noise
```

### Step 9c: Reconstruction with 3D Richardson-Lucy

```python
from pwm_core.recon import run_richardson_lucy_3d

x_hat = run_richardson_lucy_3d(
    y=y,                      # (256, 256, 64) blurred z-stack
    psf_3d=psf_3d,            # (21, 21, 41) 3D confocal PSF
    n_iters=40,               # 40 iterations (more needed for 3D)
    regularization="tikhonov",
    reg_param=0.001
)
# x_hat shape: (256, 256, 64) -- deconvolved volume
# Expected PSNR: ~29.2 dB (BII 3D mitochondria benchmark)
```

**3D Richardson-Lucy algorithm:**

```python
# 3D RL update rule (identical structure to 2D, but with 3D FFTs):
#
# x_{k+1} = x_k * (h_3d_flipped *** (y / (h_3d *** x_k + eps)))
#
# Key differences from 2D:
# 1. 3D FFT is O(N^3 log N) instead of O(N^2 log N) -- much slower
# 2. Missing cone frequencies remain zero -- no information to recover
# 3. Regularization (Tikhonov or TV) is more critical to prevent
#    noise amplification along the axial direction
# 4. More iterations needed (40 vs 30) because axial convergence is slower
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| 3D Richardson-Lucy | Traditional | 29.2 dB | No | `run_richardson_lucy_3d(y, psf_3d, n_iters=40)` |
| CARE-3D (slice-wise) | Deep Learning | 33.1 dB | Yes | `care_restore_3d(y, model_path="weights/care_3d.pth")` |
| 3D Wiener | Traditional | 27.5 dB | No | `wiener_deconv_3d(y, psf_3d, snr=80)` |

### Step 9d: Metrics

```python
# Per-slice PSNR
for z in range(64):
    psnr_z = 10 * log10(max_val^2 / mse(x_hat[:,:,z], x_true[:,:,z]))

# Average volumetric PSNR
avg_psnr = mean(psnr_per_slice)  # ~29.2 dB

# SSIM per slice and average
avg_ssim = mean([ssim(x_hat[:,:,z], x_true[:,:,z]) for z in range(64)])

# Axial resolution metric
# Theoretical axial resolution: lambda / (n - sqrt(n^2 - NA^2))
# = 561 / (1.33 - sqrt(1.33^2 - 1.2^2)) = 561 / (1.33 - 0.575) = 743 nm
# After deconvolution: ~500-600 nm (improved but still limited by missing cone)

# Isotropy ratio (axial FWHM / lateral FWHM)
# Before deconvolution: ~3.5x
# After 3D RL: ~2.5x
# After CARE-3D: ~2.0x
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
|   +-- y.npy              # Measurement (256, 256, 64) + SHA256 hash
|   +-- x_hat.npy          # Reconstruction (256, 256, 64) + SHA256 hash
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
|   +-- psf_3d.npy         # 3D PSF (21, 21, 41) + SHA256 hash
+-- metrics.json           # PSNR, SSIM per slice + average, isotropy ratio
+-- operator.json          # Operator parameters (sigma_lat, sigma_ax, z_step)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized confocal 3D pipeline (SNR 110.0 dB, mismatch 0.146). In practice, deep 3D imaging faces severe challenges: depth-dependent spherical aberration causes the PSF to degrade with depth, scattering reduces signal, and photobleaching occurs during the long acquisition time required for a full z-stack.

---

## Real Experiment: User Prompt

```
"3D confocal z-stack of thick tissue section (brain organoid, ~50 um deep).
 The deeper slices look significantly blurrier and dimmer than the surface.
 63x/1.4 NA oil, 561 nm, 64 slices at 0.3 um spacing.
 Measurement: organoid_deep.tif"
```

**Key difference:** Thick specimen with depth-dependent PSF degradation, scattering, and attenuation.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["organoid_deep.tif"],
#   params={"excitation_wavelength_nm": 561, "numerical_aperture": 1.4,
#           "n_slices": 64, "z_step_um": 0.3}
# )
```

---

## R2. PhotonAgent -- Depth-Dependent Signal Loss

### Real parameters

```yaml
confocal_3d_deep:
  power_w: 0.001
  wavelength_nm: 561
  na: 1.4
  n_medium: 1.515
  qe: 0.25                 # PMT
  exposure_s: 0.005         # 5 ms dwell (longer to compensate)
```

### Computation

```python
# At the surface (z=0): full signal
N_surface = 9.93e10 * (0.005/0.01) = 4.97e10 photons/voxel
SNR_surface_db = 20 * log10(sqrt(4.97e10)) = 107.0 dB

# At depth z=50 um: exponential attenuation from scattering
# Mean free path in brain tissue ~ 100 um (at 561 nm)
# Attenuation = exp(-2 * z / mfp) = exp(-2 * 50 / 100) = 0.368  (round trip)
N_deep = 4.97e10 * 0.368 = 1.83e10 photons/voxel
SNR_deep_db = 20 * log10(sqrt(1.83e10)) = 102.6 dB

# Average across stack
SNR_avg = (107.0 + 102.6) / 2 = 104.8 dB
```

### Output

```python
PhotonReport(
  n_photons_per_pixel=3.4e10,                    # Geometric mean
  snr_db=104.8,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="excellent",
  explanation="Shot-limited even at depth. Signal attenuates by ~2.7x at 50 um "
              "but PMT still collects sufficient photons. Depth-dependent "
              "correction (attenuation compensation) recommended."
)
```

---

## R3. MismatchAgent -- Depth-Dependent Aberration

```python
# Depth-dependent mismatch: PSF degrades significantly with depth
psi_true = {
    "psf_sigma": +0.8,    # PSF broadens 2x at 50 um depth due to RI mismatch
    "defocus":   +2.5,     # Spherical aberration shifts focus by 2.5 um at depth
    "background": 0.08,    # Scattered light background increases with depth
    "gain":       -0.30,   # Signal attenuation not compensated by gain adjustment
}

# Severity computation
S = 0.30 * |0.8| / 2.2     # psf_sigma:  0.109
  + 0.35 * |2.5| / 6.0     # defocus:    0.146
  + 0.20 * |0.08| / 0.12   # background: 0.133
  + 0.15 * |0.30| / 1.0    # gain:       0.045
S = 0.433  # HIGH severity

improvement_db = clip(10 * 0.433, 0, 20) = 4.33 dB
```

### Output

```python
MismatchReport(
  severity_score=0.433,
  correction_method="grid_search",
  expected_improvement_db=4.33,
  explanation="High mismatch severity for deep tissue. Spherical aberration from "
              "refractive index mismatch (oil n=1.515 vs tissue n=1.38) causes "
              "depth-dependent PSF degradation. Standard 3D RL with a single PSF "
              "will fail for deep slices."
)
```

---

## R4. RecoverabilityAgent

```python
# With high mismatch, 3D RL performance degrades significantly:
# Calibration lookup -> recoverability=0.80, expected_psnr=27.5 dB

RecoverabilityReport(
  compression_ratio=1.0,
  recoverability_score=0.80,
  expected_psnr_db=27.5,
  verdict="good",
  explanation="Good overall but deep slices will be worse. "
              "Depth-variant deconvolution could improve by ~2 dB."
)
```

---

## R5. AnalysisAgent

```python
photon_score      = 1 - min(104.8 / 40, 1.0)  = 0.0
mismatch_score    = 0.433
compression_score = 1 - 0.80                    = 0.20
solver_score      = 0.2

primary = "mismatch"  # max(0.0, 0.433, 0.20, 0.2) = mismatch

P = 1.0 * 0.784 * 0.90 * 0.90 = 0.635

SystemAnalysis(
  primary_bottleneck="mismatch",
  probability_of_success=0.635,
  suggestions=[
    Suggestion(text="Use depth-variant 3D PSF model (PSF varies per z-slice)",
               priority="critical", expected_gain_db=3.0),
    Suggestion(text="Apply CARE-3D which learns depth-dependent correction",
               priority="high", expected_gain_db=4.0),
    Suggestion(text="Consider adaptive optics or water immersion to reduce RI mismatch",
               priority="medium", expected_gain_db=2.5),
    Suggestion(text="Apply attenuation compensation (exponential gain with depth)",
               priority="medium", expected_gain_db=1.5)
  ],
  overall_verdict="marginal"
)
```

---

## R6. AgentNegotiator

```python
P_photon         = 0.95
P_recoverability = 0.80
P_mismatch       = 1.0 - 0.433 * 0.7 = 0.697

P_joint = 0.95 * 0.80 * 0.697 = 0.530

NegotiationResult(
  vetoes=[],
  proceed=True,
  probability_of_success=0.530
)
```

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=30.0,
  proceed_recommended=True,
  warnings=[
    "Mismatch severity 0.433 -- depth-dependent aberration in thick specimen",
    "PSF varies with depth -- standard single-PSF deconvolution will degrade deep slices",
    "Consider depth-variant PSF model or CARE-3D"
  ],
  what_to_upload=["measurement (3D z-stack, multi-page TIFF or .npy)"]
)
```

---

## R8. Pipeline Runner -- With Depth Correction

### Step R8a: Standard 3D RL (Single PSF)

```python
x_rl, _ = run_richardson_lucy_3d(y_deep, psf_3d_surface, n_iters=40)
# Average PSNR = 27.5 dB
# Surface PSNR = 29.0 dB  (good)
# Deep PSNR   = 24.0 dB   (poor -- PSF mismatch at depth)
```

### Step R8b: Attenuation Compensation + 3D RL

```python
# Compensate for depth-dependent signal loss
attenuation = np.exp(-2 * z_positions / mfp)  # (64,) array
y_compensated = y_deep / attenuation[None, None, :]

x_rl_comp, _ = run_richardson_lucy_3d(y_compensated, psf_3d_surface, n_iters=40)
# Average PSNR = 28.5 dB  (+1.0 dB from attenuation compensation)
```

### Step R8c: CARE-3D (Slice-wise)

```python
from pwm_core.recon.care_unet import care_train_quick
import torch

# Train on middle slice, apply to all slices
mid_z = 32
model = care_train_quick(y_deep[:,:,mid_z], x_true[:,:,mid_z], epochs=100)

recon_3d = np.zeros_like(y_deep)
for z in range(64):
    x_in = torch.from_numpy(y_deep[:,:,z]).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        recon_3d[:,:,z] = model(x_in).squeeze().numpy()
# Average PSNR = 33.1 dB  (+3.9 dB over 3D RL)
```

### Step R8d: Final Comparison

| Configuration | 3D RL | CARE-3D | Notes |
|---------------|-------|---------|-------|
| Surface slices (z=0-10) | **29.0 dB** | **33.5 dB** | Both good |
| Mid-depth (z=20-40) | **28.0 dB** | **33.1 dB** | RL starts degrading |
| Deep slices (z=50+) | **24.0 dB** | **32.5 dB** | RL severely degraded |
| Overall average | **27.5 dB** | **33.1 dB** | CARE: +5.6 dB |

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal (thin sample) | Real (thick tissue) |
|--------|---------------------|---------------------|
| **Photon Agent** | | |
| N_effective | 9.93e10 | 3.4e10 (average) |
| SNR | 110.0 dB | 104.8 dB |
| Quality tier | excellent | excellent |
| **Mismatch Agent** | | |
| Severity | 0.146 (low) | 0.433 (**high**) |
| Dominant error | defocus (typical) | depth aberration |
| Expected gain | +1.46 dB | +4.33 dB |
| **Recoverability Agent** | | |
| Score | 0.85 (excellent) | 0.80 (good) |
| Expected PSNR | 29.2 dB | 27.5 dB |
| **Negotiator** | | |
| P_joint | 0.725 | 0.530 |
| **Pipeline** | | |
| 3D RL PSNR | 29.2 dB | 27.5 dB |
| CARE-3D PSNR | 33.1 dB | **33.1 dB** |
| Deep-slice RL PSNR | 29.2 dB | **24.0 dB** |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (3D RL -> CARE-3D -> 3D Wiener) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Depth-aware:** The pipeline recognizes that 3D confocal imaging in thick specimens suffers from depth-dependent PSF degradation and recommends depth-variant correction or deep learning approaches that implicitly learn depth-dependent priors.
