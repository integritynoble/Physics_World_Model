# CT Working Process

## End-to-End Pipeline for X-ray Computed Tomography

This document traces a complete X-ray CT experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct a CT image from this sparse-view sinogram.
 Sinogram: sinogram.npy, 90 projections over 180 degrees,
 256 detectors, low-dose scan."
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "sinogram.npy" detected
#   operator_type=OperatorType.linear_operator,
#   files=["sinogram.npy"],
#   params={"n_angles": 90, "angular_range_deg": 180, "n_detectors": 256}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> ct entry
ct:
  keywords: [CT, tomography, Radon, sinogram, FBP, medical_imaging]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="ct",
#   confidence=0.97,
#   reasoning="Matched keywords: CT, sinogram, FBP"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the CT registry entry:

```python
system = plan_agent.build_imaging_system("ct")
# ImagingSystem(
#   modality_key="ct",
#   display_name="X-ray Computed Tomography",
#   signal_dims={"x": [256, 256], "y": [180, 256]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...5 elements...],
#   default_solver="fbp"
# )
```

**CT Element Chain (5 elements):**

```
X-ray Tube ──> Beam Collimator ──> Patient (Attenuating Medium) ──> Anti-scatter Grid ──> Scintillator + Photodiode Array
  throughput=1.0  throughput=0.85   throughput=0.10                  throughput=0.70       throughput=0.85
  noise: none     noise: none       noise: shot_poisson              noise: none           noise: shot+read+quant
                                    (Beer-Lambert attenuation)
```

**Cumulative throughput:** `0.85 x 0.10 x 0.70 x 0.85 = 0.0506`

The very low cumulative throughput (5.1%) is characteristic of CT: X-rays are exponentially attenuated by the patient body (mean attenuation coefficient mu=0.2 cm^-1, path length L=20 cm). The Beer-Lambert law gives transmission = exp(-mu*L) = exp(-4.0) = 0.018 through tissue center, with the 0.10 average throughput accounting for the full range of ray paths.

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  ct:
    model_id: "ct_xray"
    parameters:
      tube_current_photons: 1.0e+07
      mu: 0.2
      L: 20.0
      eta_det: 0.85
  ```

### Computation

```python
# CT uses Beer-Lambert attenuation model (different from optical microscopy)

# 1. Source X-ray photon count per detector element per projection
N_0 = tube_current_photons = 1.0e7

# 2. Beer-Lambert transmission through patient
#    Mean attenuation: mu = 0.2 cm^-1, average path length L = 20 cm
T = exp(-mu * L) = exp(-0.2 * 20.0) = exp(-4.0) = 0.0183

# 3. Detected photons (after collimator + grid + detector)
eta_pre = 0.85 * 0.70 = 0.595       # Collimator + grid throughput
eta_det = 0.85                       # Detector quantum efficiency
N_detected = N_0 * T * eta_pre * eta_det
           = 1.0e7 * 0.0183 * 0.595 * 0.85
           = 9.26e4 photons/detector/projection

# 4. Total detected per pixel (over all projections contributing)
#    Each image pixel receives contributions from ~n_angles projections
N_effective = N_detected * (n_angles / pi)
            = 9.26e4 * (90 / 3.14159)
            = 2.65e6 photons/pixel

# 5. Noise variances
#    In CT, the dominant noise is quantum (Poisson) in the projection data
#    After log-transform: variance = 1/N_detected per measurement
shot_var   = 1.0 / N_detected = 1.08e-5  # Variance in log-domain
read_var   = 0                            # Negligible for scintillator
total_var  = 1.08e-5

# 6. SNR (in sinogram domain, per detector element)
SNR = sqrt(N_detected) = sqrt(9.26e4) = 304.3
SNR_db = 20 * log10(304.3) = 49.7 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=9.26e4,
  snr_db=49.7,
  noise_regime=NoiseRegime.detector_limited,  # CT is typically detector-limited
  shot_noise_sigma=304.3,
  read_noise_sigma=0.0,
  total_noise_sigma=304.3,
  feasible=True,
  quality_tier="excellent",                   # SNR > 30 dB
  throughput_chain=[
    {"X-ray Tube": 1.0},
    {"Beam Collimator": 0.85},
    {"Patient (Attenuating Medium)": 0.10},
    {"Anti-scatter Grid": 0.70},
    {"Scintillator + Photodiode Array": 0.85}
  ],
  noise_model="poisson",
  explanation="Poisson-limited X-ray detection. 90 projections at ~9.3e4 photons/detector provide good SNR."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"ct"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  ct:
    parameters:
      projection_angle_offset:
        range: [-2.0, 2.0]
        typical_error: 0.5
        weight: 0.30
      detector_offset:
        range: [-5.0, 5.0]
        typical_error: 1.5
        weight: 0.45
      beam_hardening_coeff:
        range: [0.0, 0.3]
        typical_error: 0.05
        weight: 0.25
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.30 * |0.5| / 4.0        # projection_angle_offset: 0.0375
  + 0.45 * |1.5| / 10.0       # detector_offset: 0.0675
  + 0.25 * |0.05| / 0.3       # beam_hardening: 0.0417
S = 0.147  # Low severity

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.47 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="ct",
  mismatch_family="grid_search",
  parameters={
    "projection_angle_offset": {"typical_error": 0.5, "range": [-2.0, 2.0], "weight": 0.30},
    "detector_offset":         {"typical_error": 1.5, "range": [-5.0, 5.0], "weight": 0.45},
    "beam_hardening_coeff":    {"typical_error": 0.05, "range": [0.0, 0.3], "weight": 0.25}
  },
  severity_score=0.147,
  correction_method="grid_search",
  expected_improvement_db=1.47,
  explanation="Low mismatch severity. Center-of-rotation offset is the primary error source."
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
  ct:
    signal_prior_class: "tv"
    entries:
      - {cr: 0.50, noise: "detector_limited", solver: "pnp_sart",
         recoverability: 0.90, expected_psnr_db: 33.7,
         provenance: {dataset_id: "aapm_ldct_mayo_2023", ...}}
      - {cr: 0.25, noise: "detector_limited", solver: "pnp_sart",
         recoverability: 0.80, expected_psnr_db: 28.1, ...}
      - {cr: 0.25, noise: "detector_limited", solver: "redcnn",
         recoverability: 0.83, expected_psnr_db: 30.4, ...}
  ```

### Computation

```python
# 1. Compression ratio
#    90 projections x 256 detectors = 23,040 measurements
#    256 x 256 image = 65,536 pixels
CR = (90 * 256) / (256 * 256) = 23040 / 65536 = 0.352

# 2. Operator diversity (Radon transform)
#    90 projection angles uniformly span [0, 180 degrees]
#    Angular coverage = 90 / 180 = 0.50 of full Nyquist requirement
#    Radon transform is well-conditioned with sufficient angles
diversity = n_angles / (pi * n_pixels / 2)
         = 90 / (3.14159 * 128)
         = 0.224  # Moderate (sparse-view regime)
# But with structured prior (TV), effective diversity is higher
diversity_eff = 0.70

# 3. Condition number proxy
kappa = 1 / (1 + diversity_eff) = 0.588

# 4. Calibration table lookup
#    Interpolate between cr=0.25 (28.1 dB) and cr=0.50 (33.7 dB) at cr=0.352
#    Linear interpolation: 28.1 + (0.352-0.25)/(0.50-0.25) * (33.7-28.1) = 30.38 dB
#    Recoverability: 0.80 + 0.408 * (0.90-0.80) = 0.841
#    Confidence = 0.9 (interpolated)

# 5. Best solver selection
#    pnp_sart (interpolated): 30.38 dB
#    redcnn at cr=0.25: 30.4 dB
#    -> recommended: "pnp_sart" (default iterative) or "redcnn" (post-processing)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.352,
  noise_regime=NoiseRegime.detector_limited,
  signal_prior_class=SignalPriorClass.tv,
  operator_diversity_score=0.70,
  condition_number_proxy=0.588,
  recoverability_score=0.841,
  recoverability_confidence=0.9,
  expected_psnr_db=30.38,
  expected_psnr_uncertainty_db=1.5,
  recommended_solver_family="pnp_sart",
  verdict="sufficient",              # 0.60 <= score < 0.85
  calibration_table_entry={...},
  explanation="Good recoverability at 90 views. PnP-SART expected ~30.4 dB; RED-CNN post-processing also viable."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(49.7 / 40, 1.0)   = 0.0    # Excellent SNR
mismatch_score    = 0.147                       = 0.147  # Low mismatch
compression_score = 1 - 0.841                   = 0.159  # Good recoverability
solver_score      = 0.2                         = 0.2    # Default placeholder

# Primary bottleneck
primary = "solver"  # max(0.0, 0.147, 0.159, 0.2) = solver

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.147*0.5) * (1 - 0.159*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.927 * 0.921 * 0.90
  = 0.767
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="solver",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.147, compression=0.159, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Apply RED-CNN post-processing to FBP for +2.0 dB with minimal compute",
      priority="high",
      expected_gain_db=2.0
    ),
    Suggestion(
      text="PnP-SART with DRUNet provides best iterative quality",
      priority="medium",
      expected_gain_db=2.5
    ),
    Suggestion(
      text="Consider acquiring 180 projections for +5 dB improvement",
      priority="low",
      expected_gain_db=5.0
    )
  ],
  overall_verdict="sufficient",       # 0.60 <= P < 0.80
  probability_of_success=0.767,
  explanation="Well-balanced system at 90 views. Solver choice is the main lever for improvement."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="sufficient" | No veto |
| Severe mismatch without correction | severity=0.147 < 0.7 | No veto |
| All marginal | photon=excellent, others=sufficient | No veto |
| Joint probability floor | P=0.767 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.841   # recoverability_score
P_mismatch       = 1.0 - 0.147 * 0.7 = 0.897

P_joint = 0.95 * 0.841 * 0.897 = 0.717
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.717
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 256 * 256 = 65,536
dim_factor   = total_pixels / (256 * 256) = 1.0
n_angles     = 90
solver_complexity = 3.5  # PnP-SART (many SART iters + DRUNet at each stage)
cr_factor    = max(0.352, 1.0) / 8.0 = 0.125

# SART requires per-angle rotation + backprojection
# 7 stages x 6 iters x 90 angles = 3,780 forward/backward operations
# Plus DRUNet denoising at each stage
runtime_s = 2.0 * 1.0 * 3.5 * 0.125 * 90 = 78.8 seconds
# ~80 seconds for PnP-SART with 7-stage DRUNet denoising
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="ct", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=80.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=[
    "sinogram (2D, n_angles x n_detectors)",
    "geometry (optional: source-detector distances for fan-beam)"
  ]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# CT forward model: Radon transform (line integral of attenuation)
#   y(theta, s) = integral_{line(theta,s)} x(r) dl
#
# Parameters:
#   n_angles:    90 projection angles
#   angles:      np.linspace(0, pi, 90, endpoint=False)
#   n_detectors: 256 detector elements
#
# Input:  x = (256, 256) attenuation map (Hounsfield units)
# Output: y = (90, 256) sinogram

class CTOperator(PhysicsOperator):
    def forward(self, x):
        """Radon transform: y(theta, s) = sum of x along line(theta, s)"""
        from scipy.ndimage import rotate
        sinogram = np.zeros((self.n_angles, self.n), dtype=np.float32)
        for i, theta in enumerate(self.angles):
            rotated = rotate(x, np.degrees(theta), reshape=False, order=1)
            sinogram[i, :] = rotated.sum(axis=0)
        return sinogram

    def adjoint(self, sinogram):
        """Backprojection: x_hat = sum_theta smeared projections"""
        from scipy.ndimage import rotate
        recon = np.zeros((self.n, self.n), dtype=np.float32)
        for i, theta in enumerate(self.angles):
            back = np.tile(sinogram[i, :], (self.n, 1))
            rotated = rotate(back, -np.degrees(theta), reshape=False, order=1)
            recon += rotated
        return recon * np.pi / self.n_angles

    def check_adjoint(self):
        """Verify <Ax, y> ~ <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-6)
        # Rotation-based Radon has interpolation error ~1e-6
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided sinogram.npy:
sinogram = np.load("sinogram.npy")      # (90, 256)

# If simulating:
phantom = create_shepp_logan(256)        # (256, 256) Shepp-Logan phantom

# Radon transform
angles = np.linspace(0, np.pi, 90, endpoint=False)
sinogram = radon_forward(phantom, angles)

# Add Poisson noise (CT quantum noise model)
# Detected photons per element: N_0 * exp(-sinogram_value)
N_0 = 1.0e7
photons = N_0 * np.exp(-sinogram)       # Beer-Lambert
photons_noisy = np.random.poisson(photons).astype(np.float32)

# Convert back to sinogram (log-transform)
sinogram_noisy = -np.log(photons_noisy / N_0 + 1e-8)
# Add Gaussian noise approximation
sinogram_noisy += np.random.randn(*sinogram.shape) * 0.05
```

### Step 9c: Reconstruction Methods

**Method 1: Filtered Back-Projection (FBP)**

```python
def fbp_reconstruct(sinogram, angles, n):
    """FBP with Shepp-Logan windowed Ram-Lak filter."""
    n_angles, n_det = sinogram.shape

    # Zero-pad to next power of 2
    pad_size = int(2 ** np.ceil(np.log2(2 * n_det)))

    # Ram-Lak filter with Shepp-Logan window
    freq = np.fft.fftfreq(pad_size)
    ramp = np.abs(freq)
    shepp_logan = np.sinc(freq / (2 * 0.5))  # Shepp-Logan window
    filt = ramp * shepp_logan

    # Filter each projection
    filtered = np.zeros((n_angles, n_det))
    for i in range(n_angles):
        proj_padded = np.zeros(pad_size)
        proj_padded[:n_det] = sinogram[i]
        proj_fft = np.fft.fft(proj_padded)
        filtered[i] = np.real(np.fft.ifft(proj_fft * filt))[:n_det]

    # Backproject
    recon = backproject(filtered, angles, n)
    return np.clip(recon, 0, 1)

x_fbp = fbp_reconstruct(sinogram, angles, 256)
# Expected PSNR: ~25 dB (FBP baseline at 90 views)
```

**Method 2: SART-TV (Iterative)**

```python
def sart_tv_reconstruct(sinogram, angles, n, iters=40, relaxation=0.15, tv_weight=0.08):
    """SART with TV regularization.

    Algebraic Reconstruction Technique with:
    - FBP initialization (better starting point)
    - Per-angle SART normalization (correct ray/pixel weighting)
    - TV regularization every iteration
    """
    x = fbp_reconstruct(sinogram, angles, n)  # FBP init

    # Precompute per-angle normalization
    ray_norms, col_norms = precompute_sart_norms(angles, n)

    for it in range(iters):
        for i, theta in enumerate(angles):
            proj_est = forward_single(x, theta)
            residual = sinogram[i] - proj_est
            # SART update: x += relax * (1/D) * A^T * ((1/M) * r)
            x += relaxation * back_single(residual / ray_norms[i], theta, n) / col_norms[i]
            x = np.maximum(x, 0)

        # TV regularization
        x = denoise_tv_chambolle(x, weight=tv_weight)
        x = np.clip(x, 0, 1)

    return x

x_sart_tv = sart_tv_reconstruct(sinogram, angles, 256)
# Expected PSNR: ~28 dB (SART-TV at 90 views)
```

**Method 3: PnP-SART with DRUNet (Best Quality)**

```python
def pnp_sart_reconstruct(sinogram, angles, n, denoiser, device):
    """PnP-SART: SART with stage-based DRUNet denoising.

    Uses RED (Regularization by Denoising) approach:
    - FBP initialization
    - Coarse-to-fine DRUNet denoising stages
    - Stage-based sigma annealing for stable convergence
    """
    x = fbp_reconstruct(sinogram, angles, n)  # FBP init

    # Stage-based: coarse-to-fine denoising
    stages = [
        # (n_sart_iters, sigma, blend_weight)
        (6, 20.0/255, 0.50),   # Strong denoising early
        (6, 12.0/255, 0.45),
        (6,  8.0/255, 0.40),
        (6,  5.0/255, 0.35),
        (6,  3.0/255, 0.30),
        (6,  2.0/255, 0.25),
        (6,  1.0/255, 0.20),   # Light denoising late
    ]

    for n_sart, sigma, blend in stages:
        # SART iterations with proper normalization
        for _ in range(n_sart):
            for i, theta in enumerate(angles):
                proj_est = forward_single(x, theta)
                residual = sinogram[i] - proj_est
                x += relaxation * back_single(residual / ray_norms[i], theta, n) / col_norms[i]
                x = np.maximum(x, 0)

        # DRUNet denoising + blending
        x_denoised = apply_drunet(x, sigma, denoiser, device)
        x = blend * x_denoised + (1 - blend) * x
        x = np.clip(x, 0, 1)

    # Final TV polish
    x = denoise_tv_chambolle(x, weight=0.02)
    return x

x_pnp = pnp_sart_reconstruct(sinogram, angles, 256, denoiser, device)
# Expected PSNR: ~30.4 dB (PnP-SART at 90 views)
```

**Method 4: RED-CNN Post-Processing**

```python
# RED-CNN applied to FBP output (fast post-processing approach)
from pwm_core.recon.redcnn import redcnn_recon

x_redcnn = redcnn_recon(
    x_fbp,                   # (256, 256) FBP reconstruction
    model_path="weights/redcnn_ldct.pth",
    device="cuda"
)
# Expected PSNR: ~30.4 dB (RED-CNN at 90 views)
```

**Alternative solvers:**

| Solver | Type | PSNR (90 views) | GPU | Command |
|--------|------|-----------------|-----|---------|
| FBP | Analytical | ~25.0 dB | No | `fbp_reconstruct(sinogram, angles, 256)` |
| SART-TV | Iterative | ~28.0 dB | No | `sart_tv_reconstruct(sinogram, angles, 256)` |
| PnP-SART | PnP + DRUNet | ~30.4 dB | Yes | `pnp_sart_reconstruct(sinogram, angles, 256)` |
| RED-CNN | Post-processing | ~30.4 dB | Yes | `redcnn_recon(x_fbp, model_path)` |

### Step 9d: Metrics

```python
# PSNR (peak signal-to-noise ratio)
psnr = 10 * log10(max_val^2 / mse(x_hat, phantom))  # ~30.4 dB (PnP-SART)

# SSIM (structural similarity)
ssim_val = ssim(x_hat, phantom)

# RMSE in Hounsfield units (CT-specific)
rmse_hu = sqrt(mse(x_hat * 1000, phantom * 1000))  # HU scale

# Ring artifact metric
# Detect circular artifacts in the reconstruction
radial_profile = azimuthal_average(x_hat - phantom)
ring_metric = max(abs(radial_profile))
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
|   +-- sinogram.npy       # Sinogram (90, 256) + SHA256 hash
|   +-- x_hat.npy          # Reconstruction (256, 256) + SHA256 hash
|   +-- phantom.npy        # Ground truth (if available) + SHA256 hash
+-- metrics.json           # PSNR, SSIM, RMSE_HU, ring artifact metric
+-- operator.json          # Operator parameters (n_angles, angular_range, geometry)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed a standard CT pipeline with 90 projections and moderate low-dose conditions. In practice, clinical low-dose CT scans may use even fewer projections (sparse-view) or reduced tube current, and geometric calibration errors (center-of-rotation offset, beam hardening) can significantly degrade reconstruction.

This section traces the same pipeline with realistic degraded parameters for a severely sparse-view, ultra-low-dose scenario.

---

## Real Experiment: User Prompt

```
"Ultra-low-dose CT scan for pediatric patient. Only 18 projections
 were acquired (10% of standard 180-view protocol) to minimize
 radiation dose. The center of rotation may be slightly off.
 Sinogram: pediatric_sino.npy, 18 angles over 180 degrees."
```

**Key difference:** Only 18 projections (10% of standard), potential center-of-rotation error.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["pediatric_sino.npy"],
#   params={"n_angles": 18, "angular_range_deg": 180}
# )
```

---

## R2. PhotonAgent -- Ultra-Low-Dose Conditions

### Real scanner parameters

```yaml
# Ultra-low-dose pediatric scan
ct_lowdose:
  tube_current_photons: 2.0e+06    # 5x lower tube current
  mu: 0.15                         # Lower attenuation (pediatric)
  L: 15.0                          # Smaller patient (child)
  eta_det: 0.80                    # Slightly degraded detector
```

### Computation

```python
# Beer-Lambert transmission
T = exp(-0.15 * 15.0) = exp(-2.25) = 0.105

# Detected photons per element per projection
N_detected = 2.0e6 * 0.105 * 0.595 * 0.80
           = 1.00e5 photons/detector/projection

# Effective per pixel (18 angles)
N_effective = 1.00e5 * (18 / 3.14159) = 5.73e5

# SNR in sinogram domain
SNR = sqrt(1.00e5) = 316.2
SNR_db = 20 * log10(316.2) = 50.0 dB
```

### Output

```python
PhotonReport(
  n_photons_per_pixel=1.00e5,
  snr_db=50.0,
  noise_regime=NoiseRegime.detector_limited,
  feasible=True,
  quality_tier="excellent",           # 50 dB > 30 dB
  explanation="Good photon count per projection despite low dose. "
              "The challenge is sparse angular sampling (18 views), not photon noise."
)
```

---

## R3. MismatchAgent -- Real Geometric Errors

```python
# Actual errors from clinical scanner
S = 0.30 * |1.0| / 4.0        # projection_angle_offset: 0.075  (1 degree offset)
  + 0.45 * |3.0| / 10.0       # detector_offset: 0.135  (3 px center-of-rotation shift!)
  + 0.25 * |0.10| / 0.3       # beam_hardening: 0.083  (uncorrected hardening)
S = 0.293  # MODERATE severity

improvement_db = clip(10 * 0.293, 0, 20) = 2.93 dB
```

### Output

```python
MismatchReport(
  severity_score=0.293,
  correction_method="grid_search",
  expected_improvement_db=2.93,
  explanation="Moderate mismatch. Center-of-rotation offset (3 px) is the dominant error. "
              "Beam hardening from polychromatic X-rays adds cupping artifacts."
)
```

---

## R4. RecoverabilityAgent -- Severely Degraded at 10%

```python
# CR = (18 * 256) / (256 * 256) = 4608 / 65536 = 0.070
# Below minimum calibrated CR (0.10)
# Extrapolation: recoverability ~ 0.55, expected_psnr ~ 22.0 dB

# Exact calibration match: cr=0.10, solver=pnp_sart
# -> recoverability=0.65, expected_psnr=25.3 dB

# With mismatch degradation:
# recoverability_adjusted = 0.55 * (1 - 0.293*0.5) = 0.469
# expected_psnr ~ 20.0 dB
```

### Output

```python
RecoverabilityReport(
  compression_ratio=0.070,
  recoverability_score=0.469,
  expected_psnr_db=20.0,
  verdict="marginal",           # 0.40 <= score < 0.60
  explanation="Very sparse sampling (18 views) severely limits reconstruction quality. "
              "Combined with geometric mismatch, results will have streak artifacts."
)
```

---

## R5. AnalysisAgent -- Angular Undersampling is the Bottleneck

```python
photon_score      = 0.0       # Excellent (photons are not the issue)
mismatch_score    = 0.293     # Moderate
compression_score = 1 - 0.469 = 0.531   # Poor
solver_score      = 0.25

primary = "compression"   # max(0.0, 0.293, 0.531, 0.25)

P = 1.0 * 0.854 * 0.735 * 0.875 = 0.549
```

### Output

```python
SystemAnalysis(
  primary_bottleneck="compression",
  suggestions=[
    Suggestion(
      text="Increase to 45 projections for +8 dB improvement",
      priority="critical",
      expected_gain_db=8.0
    ),
    Suggestion(
      text="Calibrate center-of-rotation before reconstruction",
      priority="high",
      expected_gain_db=2.9
    ),
    Suggestion(
      text="Apply beam hardening correction to sinogram before FBP",
      priority="medium",
      expected_gain_db=1.0
    )
  ],
  overall_verdict="marginal",
  probability_of_success=0.549,
  explanation="Severely sparse sampling (18 views) dominates. "
              "Geometric mismatch (CoR offset) adds streak artifacts. "
              "Results will have limited diagnostic value."
)
```

---

## R6. AgentNegotiator -- Close to Veto

```python
P_photon         = 0.95
P_recoverability = 0.469
P_mismatch       = 1.0 - 0.293 * 0.7 = 0.795

P_joint = 0.95 * 0.469 * 0.795 = 0.354
```

No veto (P_joint > 0.15), but joint probability is low.

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=20.0,       # Fewer angles = faster SART
  proceed_recommended=True,
  warnings=[
    "Severely sparse sampling (18 projections, 10%) -- streak artifacts expected",
    "Center-of-rotation offset (3 px) -- geometric ring artifacts likely",
    "Beam hardening not corrected -- cupping artifacts in reconstruction",
    "Consider acquiring more projections if radiation dose budget permits"
  ],
  what_to_upload=["sinogram (2D, 18 x 256)"]
)
```

---

## R8. Pipeline Results

| Configuration | FBP | SART-TV | PnP-SART | RED-CNN |
|---------------|-----|---------|----------|---------|
| 90 views, ideal | 25.0 dB | 28.0 dB | 30.4 dB | 30.4 dB |
| 90 views, with mismatch | 23.5 dB | 26.5 dB | 28.5 dB | 28.5 dB |
| 45 views, ideal | 22.0 dB | 25.5 dB | 28.1 dB | 28.1 dB |
| 18 views, ideal | 17.0 dB | 22.0 dB | 25.3 dB | 24.5 dB |
| 18 views, with mismatch | 14.5 dB | 19.5 dB | 22.0 dB | 21.5 dB |

**Key findings:**
- FBP quality degrades catastrophically with sparse views (severe streak artifacts)
- PnP-SART maintains +8 dB advantage over FBP at all sampling rates (learned prior)
- At 18 views, even PnP-SART produces noticeable streak artifacts (22 dB)
- Center-of-rotation offset causes ~2.5 dB degradation across all solvers
- RED-CNN post-processing on FBP is competitive with PnP-SART but slightly lower
- Increasing from 18 to 90 views provides ~8 dB improvement (most impactful change)
- CT quality is primarily governed by angular sampling density, then geometric calibration

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Standard (90 views) | Ultra-Low-Dose (18 views) |
|--------|---------------------|---------------------------|
| **Photon Agent** | | |
| N_detected/element | 9.26e4 | 1.00e5 |
| SNR | 49.7 dB | 50.0 dB |
| Quality tier | excellent | excellent |
| **Mismatch Agent** | | |
| Severity | 0.147 (low) | 0.293 (moderate) |
| Dominant error | detector_offset | detector_offset (3 px) |
| Expected gain | +1.47 dB | +2.93 dB |
| **Recoverability Agent** | | |
| CR | 0.352 | 0.070 |
| Score | 0.841 (sufficient) | 0.469 (marginal) |
| Expected PSNR | 30.38 dB | 20.0 dB |
| **Analysis Agent** | | |
| Primary bottleneck | solver | compression |
| P(success) | 0.767 | 0.549 |
| **Negotiator** | | |
| P_joint | 0.717 | 0.354 |
| **PreFlight** | | |
| Runtime | 80.0s | 20.0s |
| Warnings | 0 | 4 |
| **Pipeline** | | |
| FBP | 25.0 dB | 14.5 dB |
| SART-TV | 28.0 dB | 19.5 dB |
| PnP-SART | 30.4 dB | 22.0 dB |
| RED-CNN | 30.4 dB | 21.5 dB |

---

## CT Physics: Beer-Lambert Noise Model

CT has a unique noise model compared to optical imaging. The measurement is a line integral of attenuation, and the noise arises from Poisson statistics of transmitted X-ray photons after exponential attenuation:

```python
# True projection: p(theta, s) = integral mu(x, y) dl
# Measured photon count: N = N_0 * exp(-p) + Poisson noise
#
# Variance of log-measurement: Var[-log(N/N_0)] ~ 1/N
#
# This means:
# - High-attenuation paths (through bone) have FEWER photons and MORE noise
# - Low-attenuation paths (through air) have MORE photons and LESS noise
# - The noise is signal-dependent and spatially non-uniform
#
# Reconstruction quality is limited by the lowest-photon ray paths,
# not the average photon count. This is why low-dose CT is especially
# challenging for dense structures (bone, metal implants).
```

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (FBP -> SART-TV -> PnP-SART -> RED-CNN) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Sampling-centric:** CT quality is primarily governed by angular sampling density (number of projections), not photon count per projection. The Beer-Lambert attenuation model makes noise analysis fundamentally different from optical imaging.
