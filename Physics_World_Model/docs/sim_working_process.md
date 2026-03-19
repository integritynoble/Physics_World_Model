# Structured Illumination Microscopy (SIM) Working Process

## End-to-End Pipeline for Super-Resolution SIM Reconstruction

This document traces a complete structured illumination microscopy experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct a super-resolution image from these SIM raw frames.
 9 frames (3 angles x 3 phases), 100x/1.49 NA oil objective, 488 nm laser.
 Measurement: sim_raw_stack.tif"
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "sim_raw_stack.tif" detected
#   operator_type=OperatorType.linear_operator,
#   files=["sim_raw_stack.tif"],
#   params={"n_angles": 3, "n_phases": 3, "numerical_aperture": 1.49,
#           "excitation_wavelength_nm": 488, "magnification": 100}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> sim entry
sim:
  keywords: [SIM, super_resolution, patterned_illumination, frequency_mixing]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="sim",
#   confidence=0.97,
#   reasoning="Matched keywords: SIM, super_resolution (3 angles x 3 phases context)"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the SIM registry entry:

```python
system = plan_agent.build_imaging_system("sim")
# ImagingSystem(
#   modality_key="sim",
#   display_name="Structured Illumination Microscopy",
#   signal_dims={"x": [512, 512], "y": [512, 512, 9]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...5 elements...],
#   default_solver="wiener_sim"
# )
```

**SIM Element Chain (5 elements):**

```
Coherent Laser (488 nm) --> SLM (Pattern Generator) --> Objective Lens (100x/1.49 NA Oil) --> Emission Filter --> sCMOS Detector
  throughput=1.0              throughput=0.60               throughput=0.70                        throughput=0.90     throughput=0.82
  noise: none                 noise: alignment+fixed_pattern  noise: aberration                    noise: none         noise: shot+read+fixed_pattern
  power_mw=20                 3 angles, 3 phases             immersion=oil                         center=525 nm       pixel_size=6.5 um
  coherence=50 mm             k_pattern=0.1 cyc/px           NA=1.49, psf_sigma=1.5 px                                 read_noise=1.6 e-
```

**Cumulative throughput:** `0.60 x 0.70 x 0.90 x 0.82 = 0.310`

**Key SIM physics:** The structured illumination pattern creates frequency mixing between the illumination pattern and the sample structure. Each raw frame captures:
```
y_k(r) = PSF ** [I_k(r) * x(r)] + n
```
where `I_k(r) = 1 + m*cos(2*pi*k_p*r + phi_k)` is the sinusoidal illumination pattern with modulation depth `m`, spatial frequency `k_p`, and phase `phi_k`.

The 9 raw frames (3 orientations x 3 phases) encode frequency information up to **2x the diffraction limit**, achieving ~100 nm lateral resolution (vs ~200 nm conventional).

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  sim:
    model_id: "microscopy_fluorescence"
    parameters:
      power_w: 0.005
      wavelength_nm: 488
      na: 1.2
      n_medium: 1.515
      qe: 0.72
      exposure_s: 0.02
  ```

### Computation

```python
# 1. Photon energy
E_photon = h * c / wavelength_nm
         = (6.626e-34 * 3e8) / (488e-9)
         = 4.074e-19 J

# 2. Collection solid angle
solid_angle = (na / n_medium)^2 / (4 * pi)
            = (1.2 / 1.515)^2 / (4 * pi)
            = 0.6272 / 12.566
            = 0.0499

# 3. Raw photon count per frame
N_raw = power_w * qe * solid_angle * exposure_s / E_photon
      = 0.005 * 0.72 * 0.0499 * 0.02 / 4.074e-19
      = 8.81e11 photons/frame

# 4. Apply cumulative throughput (0.310)
N_effective = 8.81e11 * 0.310 = 2.73e11 photons/pixel/frame

# 5. Total photons across 9 frames
N_total = 9 * N_effective = 2.46e12 photons/pixel (total dose)

# 6. Noise variances (per frame)
shot_var   = N_effective = 2.73e11                   # Poisson
read_var   = read_noise^2 = 1.6^2 = 2.56           # Gaussian (sCMOS)
total_var  = 2.73e11 + 2.56

# 7. SNR (per frame)
SNR_frame = N_effective / sqrt(total_var) ~ sqrt(N_effective) = 5.22e5
SNR_frame_db = 20 * log10(5.22e5) = 114.4 dB

# 8. Effective SNR after SIM reconstruction
#    SIM reconstruction amplifies noise by ~sqrt(9) (combining 9 frames)
#    but also recovers 2x bandwidth, so effective SNR:
SNR_sim = SNR_frame / sqrt(2)  # Noise amplification in frequency space
SNR_sim_db = SNR_frame_db - 3.0 = 111.4 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=2.73e11,
  snr_db=114.4,
  noise_regime=NoiseRegime.shot_limited,    # shot_var/total_var >> 0.9
  shot_noise_sigma=5.22e5,
  read_noise_sigma=1.6,
  total_noise_sigma=5.22e5,
  feasible=True,
  quality_tier="excellent",                 # SNR > 30 dB
  throughput_chain=[
    {"Coherent Laser (488 nm)": 1.0},
    {"SLM (Pattern Generator)": 0.60},
    {"Objective Lens (100x/1.49 NA Oil)": 0.70},
    {"Emission Filter": 0.90},
    {"sCMOS Detector": 0.82}
  ],
  noise_model="poisson",
  explanation="Shot-noise-limited. Excellent SNR per frame (114.4 dB). "
              "SIM reconstruction introduces ~3 dB noise amplification from "
              "frequency unmixing. Total dose across 9 frames is 2.46e12 photons."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"sim"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  sim:
    parameters:
      psf_sigma:   {range: [0.3, 2.0], typical_error: 0.15, weight: 0.35}
      defocus:     {range: [-1.0, 1.0], typical_error: 0.3, weight: 0.35}
      background:  {range: [0.0, 0.10], typical_error: 0.02, weight: 0.15}
      gain:        {range: [0.7, 1.3], typical_error: 0.05, weight: 0.15}
    correction_method: "grid_search"
  ```

### Computation

```python
# SIM is highly sensitive to parameter errors because reconstruction
# operates in Fourier space where small errors cause artifacts.

# Severity score (weighted normalized errors)
S = 0.35 * |0.15| / 1.7    # psf_sigma:  0.031
  + 0.35 * |0.3| / 2.0     # defocus:    0.053
  + 0.15 * |0.02| / 0.10   # background: 0.030
  + 0.15 * |0.05| / 0.6    # gain:       0.013
S = 0.127  # Low severity

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.27 dB
```

**Important note:** For SIM, even "low" mismatch in the illumination pattern parameters (pattern frequency, pattern orientation, pattern phase) can cause severe artifacts. These are not captured by the standard PSF/defocus mismatch model -- they are handled by the SIM-specific pattern estimation step during reconstruction.

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="sim",
  mismatch_family="grid_search",
  parameters={
    "psf_sigma":  {"typical_error": 0.15, "range": [0.3, 2.0], "weight": 0.35},
    "defocus":    {"typical_error": 0.3, "range": [-1.0, 1.0], "weight": 0.35},
    "background": {"typical_error": 0.02, "range": [0.0, 0.10], "weight": 0.15},
    "gain":       {"typical_error": 0.05, "range": [0.7, 1.3], "weight": 0.15}
  },
  severity_score=0.127,
  correction_method="grid_search",
  expected_improvement_db=1.27,
  explanation="Low OTF/PSF mismatch severity. However, SIM is additionally sensitive to "
              "illumination pattern parameter errors (frequency, angle, phase) which are "
              "estimated during the reconstruction process itself."
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
  sim:
    signal_prior_class: "tv"
    entries:
      - {cr: 0.33, noise: "shot_limited", solver: "wiener_sim",
         recoverability: 0.81, expected_psnr_db: 27.4,
         provenance: {dataset_id: "fairsim_bpae_2023", ...}}
      - {cr: 0.33, noise: "shot_limited", solver: "hifi_sim",
         recoverability: 0.88, expected_psnr_db: 30.9, ...}
  ```

### Computation

```python
# 1. Compression ratio
#    SIM acquires 9 raw frames to reconstruct 1 super-resolution image:
#    y = (512, 512, 9) -> x = (512, 512) at 2x resolution = (1024, 1024)
#    But at original resolution: y_total = 512*512*9 pixels, x = 512*512 pixels
CR = prod(x_shape) / prod(y_shape) = (512 * 512) / (512 * 512 * 9) = 0.111
# Or equivalently at 2x output: (1024*1024) / (512*512*9) = 0.444
# Registry uses CR = 0.33 (effective information compression)

# 2. Operator diversity
# SIM illumination patterns provide excellent frequency diversity:
#   3 angles x 3 phases = 9 measurements
#   Each angle shifts different frequency bands into the passband
#   The pattern frequency k_p extends the OTF support by k_p in each direction
diversity = 0.85  # High diversity from structured patterns

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.541

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="wiener_sim", cr=0.33
#    -> recoverability=0.81, expected_psnr=27.4 dB

# 5. Best solver selection
#    hifi_sim: 30.9 dB > wiener_sim: 27.4 dB
#    -> recommended: "wiener_sim" (default, fast) or "hifi_sim" (best quality)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.33,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.tv,
  operator_diversity_score=0.85,
  condition_number_proxy=0.541,
  recoverability_score=0.81,
  recoverability_confidence=1.0,
  expected_psnr_db=27.4,
  expected_psnr_uncertainty_db=0.9,
  recommended_solver_family="wiener_sim",
  verdict="good",                # score >= 0.75
  calibration_table_entry={...},
  explanation="Good recoverability. Wiener-SIM expected 27.4 dB; HiFi-SIM reaches 30.9 dB "
              "on FairSIM BPAE benchmark. Pattern diversity is excellent (0.85)."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(114.4 / 40, 1.0)  = 0.0    # Excellent SNR
mismatch_score    = 0.127                       = 0.127  # Low mismatch
compression_score = 1 - 0.81                    = 0.19   # Good recoverability
solver_score      = 0.2                         = 0.2    # Default

# Primary bottleneck
primary = "solver"  # max(0.0, 0.127, 0.19, 0.2) = solver

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.127*0.5) * (1 - 0.19*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.937 * 0.905 * 0.90
  = 0.763
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="solver",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.127, compression=0.19, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Use HiFi-SIM for +3.5 dB over Wiener-SIM with better artifact suppression",
      priority="medium",
      expected_gain_db=3.5
    ),
    Suggestion(
      text="Verify illumination pattern parameters before reconstruction",
      priority="medium",
      expected_gain_db=1.0
    ),
    Suggestion(
      text="DL-SIM can achieve real-time super-resolution with GPU",
      priority="low",
      expected_gain_db=2.0
    )
  ],
  overall_verdict="good",           # P >= 0.70
  probability_of_success=0.763,
  explanation="System well-configured for SIM reconstruction. Solver choice is the "
              "primary lever. Illumination pattern accuracy is critical for artifact-free results."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND CR=0.33 | No veto (good SNR) |
| Severe mismatch without correction | severity=0.127 < 0.7 | No veto |
| All marginal | All good/excellent | No veto |
| Joint probability floor | P=0.763 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95   # tier_prob["excellent"]
P_recoverability = 0.81   # recoverability_score
P_mismatch       = 1.0 - 0.127 * 0.7 = 0.911

P_joint = 0.95 * 0.81 * 0.911 = 0.701
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.701
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 512 * 512 * 9 = 2,359,296  # 9 input frames
dim_factor   = total_pixels / (256 * 256) = 36.0
solver_complexity = 1.5  # Wiener-SIM (FFT-based, moderate)
cr_factor    = max(0.33, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 36.0 * 1.5 * 0.125 = 13.5 seconds
# Wiener-SIM on 9 frames of 512x512: ~10-15 seconds on CPU
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="sim", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=13.5,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=["measurement (SIM raw stack with 9 frames, multi-page TIFF)"]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# SIM forward model: y_k = PSF ** (I_k * x) + n, k = 1..9
#
# The structured illumination pattern for angle theta and phase phi:
#   I_k(r) = 1 + m * cos(2*pi*k_p*(r . e_theta) + phi_k)
#
# where:
#   m    = modulation depth (~0.8-1.0 for good SLM)
#   k_p  = pattern spatial frequency (0.1 cycles/pixel ~ near OTF cutoff)
#   e_theta = unit vector at angle theta
#   phi_k   = phase step (0, 2*pi/3, 4*pi/3 for 3-phase)
#
# In Fourier space, the illumination creates three copies of X(k):
#   Y_k(k) = OTF(k) * [X(k) + (m/2)*e^{j*phi_k}*X(k-k_p) + (m/2)*e^{-j*phi_k}*X(k+k_p)]
#
# The three phase steps for each angle allow separation of the three bands:
#   X(k), X(k-k_p), X(k+k_p)
#
# With 3 angles, the extended OTF support covers ~2x the conventional cutoff.
#
# Parameters:
#   n_angles = 3, n_phases = 3 (total 9 frames)
#   k_p = 0.1 cycles/pixel (near the OTF cutoff for maximum extension)
#   angles = [0, pi/3, 2*pi/3] radians
#   PSF sigma = 1.5 px (100x/1.49 NA oil)
#
# Input:  x = (512, 512) sample fluorescence distribution
# Output: y = (512, 512, 9) stack of patterned images

class SIMOperator(PhysicsOperator):
    def forward(self, x):
        """Generate 9 SIM raw frames from sample x"""
        y = np.zeros((self.H, self.W, self.n_angles * self.n_phases))
        idx = 0
        for a in range(self.n_angles):
            theta = a * np.pi / self.n_angles
            for p in range(self.n_phases):
                phi = p * 2 * np.pi / self.n_phases
                # Illumination pattern
                X, Y = np.meshgrid(np.arange(self.W), np.arange(self.H))
                pattern = 1 + self.m * np.cos(
                    2 * np.pi * self.k_p *
                    (X * np.cos(theta) + Y * np.sin(theta)) + phi
                )
                # Modulate sample and blur with PSF
                y[:, :, idx] = fftconvolve(x * pattern, self.psf, mode='same')
                idx += 1
        return y

    def adjoint(self, y):
        """Correlate each frame with PSF and sum"""
        x_hat = np.zeros((self.H, self.W))
        idx = 0
        for a in range(self.n_angles):
            for p in range(self.n_phases):
                x_hat += fftconvolve(
                    y[:, :, idx], self.psf[::-1, ::-1], mode='same'
                )
                idx += 1
        return x_hat / (self.n_angles * self.n_phases)

    def check_adjoint(self):
        """Verify <Ax, y> ~ <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided measurement:
y = load_tiff_stack("sim_raw_stack.tif")    # (512, 512, 9)

# If simulating:
x_true = load_ground_truth()                 # (512, 512) fine structures
n_angles, n_phases = 3, 3
k_p = 0.15                                   # Pattern frequency

# Generate 9 SIM patterns
patterns = np.zeros((512, 512, 9), dtype=np.float32)
idx = 0
for a in range(n_angles):
    theta = a * np.pi / n_angles
    for p in range(n_phases):
        phi = p * 2 * np.pi / n_phases
        X, Y = np.meshgrid(np.arange(512), np.arange(512))
        illumination = 0.5 + 0.5 * np.cos(
            2 * np.pi * k_p * (X * np.cos(theta) + Y * np.sin(theta)) + phi
        )
        patterns[:, :, idx] = x_true * illumination
        idx += 1

# Add noise
patterns += np.random.randn(512, 512, 9).astype(np.float32) * 0.02
```

### Step 9c: Reconstruction with Wiener-SIM

```python
from pwm_core.recon import run_sim_reconstruction as run_wiener_sim

class SIMPhysics:
    def __init__(self):
        self.n_angles = 3
        self.n_phases = 3
        self.k = 0.15  # Pattern frequency

physics = SIMPhysics()
patterns_transposed = y.transpose(2, 0, 1)  # (9, 512, 512)

x_hat, info = run_wiener_sim(
    y=patterns_transposed,      # (9, 512, 512) raw SIM frames
    physics=physics,
    cfg={"wiener_param": 0.001}
)
# x_hat shape: (512, 512) super-resolution image (or (1024, 1024) at 2x)
# Expected PSNR: ~27.4 dB (FairSIM BPAE benchmark)
```

**Wiener-SIM algorithm:**

```python
# Step 1: Separate frequency bands (per angle)
# For each angle theta with 3 phases [phi_0, phi_1, phi_2]:
#
#   [Y_0]   [1  exp(j*phi_0)  exp(-j*phi_0)] [S_0]
#   [Y_1] = [1  exp(j*phi_1)  exp(-j*phi_1)] [S_1]    (3x3 mixing matrix)
#   [Y_2]   [1  exp(j*phi_2)  exp(-j*phi_2)] [S_2]
#
# Invert: [S_0, S_1, S_2] = M^{-1} [Y_0, Y_1, Y_2]
# S_0 = FT(x), S_1 = FT(x shifted by +k_p), S_2 = FT(x shifted by -k_p)
#
# Step 2: Shift bands to correct positions in Fourier space
# S_1 -> shift by -k_p (move back to original position)
# S_2 -> shift by +k_p
#
# Step 3: Combine all bands (3 angles x 3 bands = 9 components)
# using generalized Wiener filter:
#
#   X_hat(k) = Sum_j [ OTF_j*(k) * S_j(k) ] / [ Sum_j |OTF_j(k)|^2 + w^2 ]
#
# where w is the Wiener regularization parameter.
#
# Step 4: Inverse FFT to get super-resolution image
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| Wiener-SIM | Traditional | 27.4 dB | No | `run_wiener_sim(y, physics, {"wiener_param": 0.001})` |
| HiFi-SIM | Iterative | 30.9 dB | No | `hifi_sim_2d(y, n_angles=3, n_phases=3)` |
| DL-SIM | Deep Learning | 29.0 dB | Yes | `dl_sim_reconstruct(y, n_angles=3, n_phases=3)` |

### Step 9d: Metrics

```python
# PSNR (at original resolution for fair comparison)
if x_hat.shape != x_true.shape:
    # HiFi-SIM/DL-SIM may output 2x resolution
    from scipy.ndimage import zoom
    x_hat_ds = zoom(x_hat, 0.5, order=1)
    psnr = 10 * log10(max_val^2 / mse(x_hat_ds, x_true))
else:
    psnr = 10 * log10(max_val^2 / mse(x_hat, x_true))
# ~ 27.4 dB (Wiener-SIM)

# SSIM
ssim_val = ssim(x_hat, x_true)

# Resolution metric: SIM theoretical resolution
# Conventional: d = lambda / (2 * NA) = 488 / (2 * 1.49) = 164 nm
# SIM (2x):     d_SIM = lambda / (2 * (NA + NA_pattern))
#             ~ lambda / (4 * NA) = 488 / (4 * 1.49) = 82 nm
# Practical SIM resolution: ~100-120 nm (limited by pattern contrast)

# Fourier ring correlation (FRC) for resolution estimation
# FRC crossing at 1/7 threshold gives effective resolution
frc_resolution_nm = compute_frc(x_hat, x_hat_independent)
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
|   +-- y.npy              # Raw SIM stack (512, 512, 9) + SHA256 hash
|   +-- x_hat.npy          # Super-resolution image (512, 512) or (1024, 1024)
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
+-- metrics.json           # PSNR, SSIM, FRC resolution, pattern parameters
+-- operator.json          # Pattern parameters (k_p, angles, phases, OTF)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized SIM pipeline (SNR 114.4 dB, mismatch 0.127). In practice, SIM is extremely sensitive to illumination pattern imperfections, out-of-focus background, and photobleaching between frames. These cause characteristic "honeycomb" artifacts in the reconstructed image.

---

## Real Experiment: User Prompt

```
"SIM raw data from our home-built system. The pattern contrast seems low
 and there may be some drift between frames. Also seeing some out-of-focus
 haze. 3 angles x 3 phases, 100x/1.49 NA oil, 488 nm.
 Data: sim_homebuilt.tif"
```

**Key difference:** Home-built system with imperfect pattern generation, inter-frame drift, and out-of-focus background.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["sim_homebuilt.tif"],
#   params={"n_angles": 3, "n_phases": 3, "numerical_aperture": 1.49,
#           "excitation_wavelength_nm": 488}
# )
```

---

## R2. PhotonAgent -- Reduced Pattern Contrast

```yaml
# Home-built SIM with lower pattern contrast and photobleaching
sim_homebuilt:
  power_w: 0.003           # Slightly lower power
  wavelength_nm: 488
  na: 1.49
  n_medium: 1.515
  qe: 0.72
  exposure_s: 0.03          # Longer exposure to compensate
  modulation_depth: 0.5     # Pattern contrast only 50% (vs ideal 100%)
```

### Computation

```python
# Photon count per frame
N_raw = 0.003 * 0.72 * 0.0499 * 0.03 / 4.074e-19 = 7.93e11
N_effective = 7.93e11 * 0.310 = 2.46e11 photons/pixel/frame

# SNR per frame
SNR_frame = sqrt(2.46e11) = 4.96e5
SNR_frame_db = 20 * log10(4.96e5) = 113.9 dB

# But effective SIM SNR is reduced by low modulation depth:
# At modulation m=0.5, the frequency-shifted bands have amplitude m/2=0.25
# vs m/2=0.5 at full modulation. SNR of shifted bands:
SNR_shifted = SNR_frame * (0.5 / 1.0) = 2.48e5
SNR_shifted_db = 107.9 dB  # Still adequate, but 6 dB worse for high-freq info

PhotonReport(
  n_photons_per_pixel=2.46e11,
  snr_db=113.9,
  noise_regime=NoiseRegime.shot_limited,
  quality_tier="excellent",
  explanation="Adequate photon budget. However, low modulation depth (m=0.5) "
              "reduces SNR of super-resolution frequency bands by 6 dB. "
              "High-frequency features will be noisier."
)
```

---

## R3. MismatchAgent -- Pattern Imperfections

```python
# SIM-specific mismatch: pattern errors + OTF errors
psi_true = {
    "psf_sigma": +0.3,    # OTF slightly broader (home-built alignment)
    "defocus":   +0.5,     # Slight axial offset
    "background": 0.08,    # Out-of-focus haze (thick sample, no confocal rejection)
    "gain":       -0.10,   # Photobleaching causes ~10% signal drop frame 1->9
}

# Additional SIM-specific errors (not in standard mismatch model):
# - Pattern frequency error: k_p off by 2% -> stripe artifacts
# - Pattern angle error: 0.5 degrees -> directional artifacts
# - Inter-frame drift: 0.3 px -> ghosting artifacts

# Standard severity
S = 0.35 * |0.3| / 1.7     # psf_sigma:  0.062
  + 0.35 * |0.5| / 2.0     # defocus:    0.088
  + 0.15 * |0.08| / 0.10   # background: 0.120
  + 0.15 * |0.10| / 0.6    # gain:       0.025
S = 0.295  # Moderate severity

# SIM-specific penalty (pattern errors add ~0.1 to severity)
S_effective = S + 0.10 = 0.395  # High-moderate

improvement_db = clip(10 * 0.395, 0, 20) = 3.95 dB
```

### Output

```python
MismatchReport(
  severity_score=0.395,
  correction_method="grid_search",
  expected_improvement_db=3.95,
  explanation="High-moderate mismatch. Out-of-focus background (8%) is the dominant "
              "PSF-related error. Additionally, SIM pattern imperfections (low modulation, "
              "drift, frequency error) will cause honeycomb artifacts unless corrected "
              "during reconstruction. HiFi-SIM is recommended for artifact suppression."
)
```

---

## R4. RecoverabilityAgent

```python
# With moderate mismatch + low modulation:
# Calibration lookup -> noise="photon_starved" (effective, due to low modulation)
# -> recoverability=0.62, expected_psnr=22.7 dB

RecoverabilityReport(
  compression_ratio=0.33,
  recoverability_score=0.62,
  expected_psnr_db=22.7,
  verdict="marginal",
  explanation="Low modulation depth reduces effective SNR of super-resolution bands. "
              "Wiener-SIM will produce artifacts; HiFi-SIM recommended."
)
```

---

## R5. AnalysisAgent

```python
photon_score      = 1 - min(113.9 / 40, 1.0)  = 0.0
mismatch_score    = 0.395
compression_score = 1 - 0.62                    = 0.38
solver_score      = 0.15

primary = "mismatch"  # max(0.0, 0.395, 0.38, 0.15) = mismatch

P = 1.0 * 0.803 * 0.81 * 0.925 = 0.601

SystemAnalysis(
  primary_bottleneck="mismatch",
  probability_of_success=0.601,
  suggestions=[
    Suggestion(text="Use HiFi-SIM with iterative pattern parameter refinement",
               priority="critical", expected_gain_db=4.0),
    Suggestion(text="Apply flat-field correction to remove out-of-focus haze before SIM",
               priority="high", expected_gain_db=2.0),
    Suggestion(text="Correct inter-frame drift using cross-correlation registration",
               priority="high", expected_gain_db=1.5),
    Suggestion(text="Calibrate SLM pattern with a uniform fluorescent slide",
               priority="medium", expected_gain_db=2.0)
  ],
  overall_verdict="marginal"
)
```

---

## R6. AgentNegotiator

```python
P_photon         = 0.95
P_recoverability = 0.62
P_mismatch       = 1.0 - 0.395 * 0.7 = 0.724

P_joint = 0.95 * 0.62 * 0.724 = 0.426

NegotiationResult(
  vetoes=[],
  proceed=True,
  probability_of_success=0.426
)
```

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=25.0,
  proceed_recommended=True,
  warnings=[
    "Pattern modulation depth appears low (m~0.5) -- super-resolution contrast will be reduced",
    "Mismatch severity 0.395 -- out-of-focus background and pattern errors expected",
    "HiFi-SIM recommended over Wiener-SIM for artifact suppression"
  ],
  what_to_upload=["measurement (SIM raw stack, 9 frames, multi-page TIFF)"]
)
```

---

## R8. Pipeline Runner

### Step R8a: Wiener-SIM (Standard)

```python
x_wiener, _ = run_wiener_sim(y_homebuilt.transpose(2,0,1), physics, {"wiener_param": 0.001})
# PSNR = 22.7 dB
# Visible honeycomb artifacts from pattern parameter errors
# Out-of-focus haze reduces contrast
```

### Step R8b: Pre-processing + HiFi-SIM

```python
# Step 1: Flat-field correction (remove background haze)
background = estimate_sim_background(y_homebuilt, method="notch_filter")
y_corrected = y_homebuilt - background[:, :, None]

# Step 2: Inter-frame drift correction
from scipy.ndimage import shift as ndshift
for i in range(1, 9):
    drift = cross_correlate_drift(y_corrected[:,:,0], y_corrected[:,:,i])
    y_corrected[:,:,i] = ndshift(y_corrected[:,:,i], -drift)

# Step 3: HiFi-SIM with iterative pattern estimation
from pwm_core.recon.sim_solver import hifi_sim_2d
x_hifi = hifi_sim_2d(
    y_corrected.transpose(2,0,1),
    n_angles=3,
    n_phases=3
)
# PSNR = 28.5 dB  (+5.8 dB over raw Wiener-SIM)
# HiFi-SIM iteratively refines pattern parameters during reconstruction
```

### Step R8c: Final Comparison

| Configuration | Wiener-SIM | HiFi-SIM | Notes |
|---------------|------------|----------|-------|
| Raw data, no correction | **22.7 dB** | **25.0 dB** | Honeycomb artifacts visible |
| Background + drift corrected | **25.5 dB** | **28.5 dB** | Artifacts reduced |
| Ideal conditions (reference) | **27.4 dB** | **30.9 dB** | Upper bound |

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real (Home-Built) |
|--------|-----------|---------------------|
| **Photon Agent** | | |
| N_effective/frame | 2.73e11 | 2.46e11 |
| SNR per frame | 114.4 dB | 113.9 dB |
| Quality tier | excellent | excellent |
| Modulation depth | 1.0 | **0.5** |
| **Mismatch Agent** | | |
| Severity | 0.127 (low) | 0.395 (**high-moderate**) |
| Dominant error | defocus | background + pattern errors |
| **Recoverability Agent** | | |
| Score | 0.81 (good) | 0.62 (marginal) |
| Expected PSNR | 27.4 dB | 22.7 dB |
| Verdict | good | **marginal** |
| **Negotiator** | | |
| P_joint | 0.701 | 0.426 |
| **Pipeline** | | |
| Wiener-SIM PSNR | 27.4 dB | 22.7 dB |
| HiFi-SIM PSNR | 30.9 dB | **28.5 dB** (corrected) |
| Resolution | ~100 nm | ~130 nm (reduced by low m) |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (Wiener-SIM -> HiFi-SIM -> DL-SIM) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Frequency-aware:** The pipeline understands that SIM reconstruction operates in Fourier space and that illumination pattern accuracy is critical for artifact-free super-resolution. It recommends iterative solvers (HiFi-SIM) when pattern quality is uncertain.
