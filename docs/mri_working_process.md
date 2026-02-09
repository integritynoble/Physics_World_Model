# MRI Working Process

## End-to-End Pipeline for Magnetic Resonance Imaging

This document traces a complete accelerated MRI experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct a knee MRI from undersampled multi-coil k-space data.
 K-space: kspace.npy (8 coils), Mask: sampling_mask.npy, 4x acceleration."
```

---

## 2. PlanAgent --- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "kspace.npy" detected
#   operator_type=OperatorType.linear_operator,
#   files=["kspace.npy", "sampling_mask.npy"],
#   params={"acceleration": 4, "n_coils": 8}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> mri entry
mri:
  keywords: [MRI, Fourier, k_space, undersampling, parallel_imaging]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="mri",
#   confidence=0.97,
#   reasoning="Matched keywords: MRI, k_space, undersampling"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the MRI registry entry:

```python
system = plan_agent.build_imaging_system("mri")
# ImagingSystem(
#   modality_key="mri",
#   display_name="Magnetic Resonance Imaging",
#   signal_dims={"x": [320, 320], "y": [320, 320]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...4 elements...],
#   default_solver="sense"
# )
```

**MRI Element Chain (4 elements):**

```
Main Magnet (3T) ----> RF Excitation Coil ----> Gradient Coils (k-space) ----> Receive Coil Array
  throughput=1.0        throughput=0.95          throughput=1.0               throughput=0.90
  noise: none           noise: none              noise: alignment             noise: thermal+read
                        flip_angle=90 deg        sampling_rate=0.25
                        pulse=sinc               center_fraction=0.08
```

**Cumulative throughput:** `1.0 x 0.95 x 1.0 x 0.90 = 0.855`

**Forward model:** `y_c = M * F * S_c * x + n_c, c = 1..N_coils`

where F is the 2D Fourier transform, M is the undersampling mask, and S_c are coil sensitivity maps.

---

## 3. PhotonAgent --- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  mri:
    model_id: "mri_thermal"
    parameters:
      B0: 3.0
      voxel_mm3: 1.0
      n_averages: 1
      bandwidth_hz: 200000
  ```

### Computation

MRI noise is fundamentally different from optical imaging --- it is dominated by thermal noise in the receive coils rather than photon shot noise.

```python
# 1. MRI signal model (Boltzmann magnetization)
# M_0 proportional to gamma^2 * hbar^2 * B0 * rho / (4 * k_B * T)
#   gamma = 2.675e8 rad/s/T (proton gyromagnetic ratio)
#   rho ~ 6.7e28 protons/m^3 (water at 37 C)
#   B0 = 3.0 T, T = 310 K
#
# Signal per voxel (arbitrary scanner units):
S_voxel = 1.0  # normalized

# 2. Thermal noise model
# noise_sigma = k_B * T / (mu_0 * gamma^2 * B0 * V_coil * Q * BW)^(1/2)
# In practice, noise scales as sqrt(bandwidth):
noise_sigma = 1.0 / np.sqrt(bandwidth_hz * n_averages)
#            = 1.0 / sqrt(200000 * 1)
#            = 0.00224

# 3. Apply throughput chain
S_effective = S_voxel * 0.855 = 0.855

# 4. Multi-coil SNR (root-sum-of-squares combination)
# SNR_rss = sqrt(n_coils) * S_effective / noise_sigma
SNR = np.sqrt(8) * 0.855 / 0.00224
#   = 2.828 * 0.855 / 0.00224
#   = 1079.4

SNR_db = 20 * np.log10(1079.4) = 60.7  # dB

# 5. Undersampled SNR (acceleration penalty)
# Undersampling by R=4 reduces SNR by sqrt(R) (SENSE g-factor)
SNR_accel = SNR / np.sqrt(4) = 539.7
SNR_accel_db = 20 * np.log10(539.7) = 54.6  # dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=None,             # N/A for MRI (not photon-based)
  snr_db=54.6,
  noise_regime=NoiseRegime.read_limited, # thermal noise dominates
  shot_noise_sigma=0.0,                 # no shot noise in MRI
  read_noise_sigma=0.00224,             # thermal noise per coil
  total_noise_sigma=0.00224,
  feasible=True,
  quality_tier="excellent",              # SNR > 30 dB
  throughput_chain=[
    {"Main Magnet (3T)": 1.0},
    {"RF Excitation Coil": 0.95},
    {"Gradient Coils": 1.0},
    {"Receive Coil Array": 0.90}
  ],
  noise_model="gaussian",               # thermal noise is Gaussian
  explanation="Read-limited regime (thermal noise). 8-coil parallel imaging "
              "provides sqrt(8) SNR gain. Excellent SNR for reconstruction."
)
```

---

## 4. MismatchAgent --- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"mri"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  mri:
    parameters:
      coil_sensitivity_error:
        range: [0.0, 0.5]
        typical_error: 0.1
        unit: "normalized"
        description: "Coil sensitivity map estimation error (ESPIRiT residual)"
      k_space_trajectory_error:
        range: [0.0, 0.05]
        typical_error: 0.01
        unit: "1/FOV"
        description: "Gradient timing error causing k-space trajectory deviation"
      B0_inhomogeneity:
        range: [0.0, 100.0]
        typical_error: 20.0
        unit: "Hz"
        description: "Static field inhomogeneity causing geometric distortion"
    severity_weights:
      coil_sensitivity_error: 0.45
      k_space_trajectory_error: 0.30
      B0_inhomogeneity: 0.25
    correction_method: "gradient_descent"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.45 * |0.1| / 0.5       # coil_sensitivity: 0.090
  + 0.30 * |0.01| / 0.05     # k_trajectory:     0.060
  + 0.25 * |20.0| / 100.0    # B0_inhom:         0.050
S = 0.200  # Moderate severity (typical clinical scanner)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 2.00 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="mri",
  mismatch_family="gradient_descent",
  parameters={
    "coil_sensitivity_error": {"typical_error": 0.1, "range": [0.0, 0.5], "weight": 0.45},
    "k_space_trajectory_error": {"typical_error": 0.01, "range": [0.0, 0.05], "weight": 0.30},
    "B0_inhomogeneity": {"typical_error": 20.0, "range": [0.0, 100.0], "weight": 0.25}
  },
  severity_score=0.200,
  correction_method="gradient_descent",
  expected_improvement_db=2.00,
  explanation="Moderate mismatch severity under typical conditions. "
              "Coil sensitivity estimation error is the dominant source."
)
```

---

## 5. RecoverabilityAgent --- Can We Reconstruct?

**File:** `agents/recoverability_agent.py` (912 lines)

### Input
- `ImagingSystem` (signal_dims for CR calculation)
- `PhotonReport` (noise regime)
- Calibration table from `compression_db.yaml`:
  ```yaml
  mri:
    signal_prior_class: "wavelet_sparse"
    entries:
      - {cr: 0.25, noise: "read_limited", solver: "pnp_admm",
         recoverability: 0.88, expected_psnr_db: 40.1,
         provenance: {dataset_id: "fastmri_knee_multicoil_2023", ...}}
      - {cr: 0.25, noise: "read_limited", solver: "varnet",
         recoverability: 0.95, expected_psnr_db: 48.2, ...}
      - {cr: 0.125, noise: "read_limited", solver: "pnp_admm",
         recoverability: 0.70, expected_psnr_db: 34.2, ...}
      - {cr: 0.125, noise: "read_limited", solver: "varnet",
         recoverability: 0.86, expected_psnr_db: 38.4, ...}
  ```

### Computation

```python
# 1. Compression ratio
# For MRI, CR = fraction of k-space lines sampled
# 4x acceleration -> 25% of k-space sampled
CR = 1 / 4 = 0.25

# 2. Operator diversity (variable-density sampling)
# Center fraction = 0.08 -> fully sampled low frequencies
# Outer k-space: random Cartesian lines
# Variable density provides good incoherence
diversity = 0.85  # high diversity from variable-density pattern

# 3. Condition number proxy
# Multi-coil parallel imaging reduces effective condition number
# g-factor penalty for SENSE: kappa ~ g_max
kappa = 1 / (1 + diversity * np.sqrt(8)) = 0.293

# 4. Calibration table lookup
#    Match: noise="read_limited", solver="varnet", cr=0.25 (exact match)
#    -> recoverability=0.95, expected_psnr=48.2 dB, confidence=1.0

# 5. Best solver selection
#    varnet: 48.2 dB >> pnp_admm: 40.1 dB
#    -> recommended: "varnet"
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.25,
  noise_regime=NoiseRegime.read_limited,
  signal_prior_class=SignalPriorClass.wavelet_sparse,
  operator_diversity_score=0.85,
  condition_number_proxy=0.293,
  recoverability_score=0.95,
  recoverability_confidence=1.0,
  expected_psnr_db=48.2,
  expected_psnr_uncertainty_db=1.0,
  recommended_solver_family="varnet",
  verdict="excellent",             # score >= 0.85
  calibration_table_entry={...},
  explanation="Excellent recoverability. VarNet expected 48.2 dB on fastMRI benchmark. "
              "Multi-coil data provides strong parallel imaging constraints."
)
```

---

## 6. AnalysisAgent --- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(54.6 / 40, 1.0)   = 0.0     # Excellent SNR
mismatch_score    = 0.200                        = 0.200   # Moderate mismatch
compression_score = 1 - 0.95                     = 0.05    # Excellent recoverability
solver_score      = 0.15                         = 0.15    # VarNet well-characterized

# Primary bottleneck
primary = "mismatch"  # max(0.0, 0.200, 0.05, 0.15) = mismatch

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.200*0.5) * (1 - 0.05*0.5) * (1 - 0.15*0.5)
  = 1.0 * 0.90 * 0.975 * 0.925
  = 0.812
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="mismatch",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.200, compression=0.05, solver=0.15
  ),
  suggestions=[
    Suggestion(
      text="Use ESPIRiT for coil sensitivity estimation to reduce mismatch",
      priority="medium",
      expected_gain_db=1.5
    ),
    Suggestion(
      text="VarNet provides +8.1 dB over PnP-ADMM at 4x acceleration",
      priority="high",
      expected_gain_db=8.1
    )
  ],
  overall_verdict="excellent",       # P >= 0.80
  probability_of_success=0.812,
  explanation="System is well-configured. Coil sensitivity calibration is the primary "
              "bottleneck; use ESPIRiT autocalibration from ACS lines."
)
```

---

## 7. AgentNegotiator --- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="excellent" | No veto |
| Severe mismatch without correction | severity=0.200 < 0.7 | No veto |
| All marginal | All excellent/sufficient | No veto |
| Joint probability floor | P=0.812 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.95    # recoverability_score
P_mismatch       = 1.0 - 0.200 * 0.7 = 0.860

P_joint = 0.95 * 0.95 * 0.860 = 0.776
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],               # No vetoes
  proceed=True,
  probability_of_success=0.776
)
```

---

## 8. PreFlightReportBuilder --- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 320 * 320 = 102,400
dim_factor   = total_pixels / (256 * 256) = 1.5625
solver_complexity = 3.0  # VarNet (12-cascade unrolled network)
cr_factor    = max(0.25, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 1.5625 * 3.0 * 0.125 = 1.17 seconds
# Multi-coil overhead: x8 coils
runtime_s *= 8 = 9.4 seconds
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="mri", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=9.4,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=[
    "multi-coil k-space data (complex, [n_coils, H, W])",
    "undersampling mask (binary, [H, W])",
    "coil sensitivity maps (optional, [n_coils, H, W])"
  ]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# MRI forward model: y_c = M * F * S_c * x + n_c
#
# Parameters:
#   mask:     (H, W) binary undersampling pattern  [loaded from sampling_mask.npy]
#   n_coils:  8                                     [from kspace shape]
#   sens:     (8, H, W) complex coil sensitivities  [estimated via ESPIRiT]
#
# Input:  x = (320, 320) complex MR image
# Output: y = (8, 320, 320) undersampled multi-coil k-space

class MRIOperator(PhysicsOperator):
    def forward(self, x):
        """y_c(kx,ky) = M(kx,ky) * FFT2(S_c(x,y) * image(x,y))"""
        y = np.zeros((n_coils, H, W), dtype=np.complex64)
        for c in range(n_coils):
            coil_img = sens[c] * x
            coil_kspace = np.fft.fftshift(np.fft.fft2(coil_img))
            y[c] = mask * coil_kspace
        return y

    def adjoint(self, y):
        """x_hat = sum_c conj(S_c) * IFFT2(M * y_c)"""
        x = np.zeros((H, W), dtype=np.complex64)
        for c in range(n_coils):
            filled_kspace = mask * y[c]
            coil_img = np.fft.ifft2(np.fft.ifftshift(filled_kspace))
            x += np.conj(sens[c]) * coil_img
        return x

    def check_adjoint(self):
        """Verify <Ax, y> ~ <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided kspace.npy:
y = np.load("kspace.npy")           # (8, 320, 320) complex
mask = np.load("sampling_mask.npy") # (320, 320) binary

# Estimate coil sensitivities from ACS (auto-calibration signal)
center = n // 6  # 8% center fraction
acs = kspace_full[:, H//2-center:H//2+center, W//2-center:W//2+center]
sens = espirit(acs, n_maps=1)       # ESPIRiT autocalibration

# If simulating:
x_true = load_ground_truth()         # (320, 320) complex knee image
y = operator.forward(x_true)         # (8, 320, 320) complex k-space
y += (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape)) * noise_sigma
```

### Step 9c: Reconstruction

**Algorithm 1: Zero-Filled SENSE (baseline)**

```python
# Simple inverse FFT with mask
x_zf = operator.adjoint(y)
# PSNR ~ 25-28 dB (aliasing artifacts from undersampling)
```

**Algorithm 2: PnP-ADMM with DRUNet (default)**

```python
from pwm_core.recon.pnp import pnp_admm_mri

x_hat = pnp_admm_mri(
    masked_kspace=y,           # (8, 320, 320) complex
    sampling_mask=mask,        # (320, 320) binary
    sens_maps=sens,            # (8, 320, 320) complex
    max_iter=50,
    rho=0.1,
    denoiser="drunet",
    sigma_schedule=(25/255, 3/255),  # annealing
    device="cuda"
)
# x_hat shape: (320, 320) -- reconstructed MR image
# Expected PSNR: ~40.1 dB (fastMRI benchmark, 4x)
```

**PnP-ADMM iteration (k-space data consistency):**

```python
# ADMM splitting: min_x ||M*F*S*x - y||^2 + rho/2 ||x - z + u||^2
#
# x-update (data consistency in k-space):
#   X = (M * Y + rho * F(z - u)) / (M + rho)
#   x = IFFT2(X)
#
# z-update (denoising):
#   z = DRUNet(x + u, sigma_t)   # sigma annealed over iterations
#
# u-update (dual variable):
#   u = u + x - z
```

**Alternative solvers:**

| Solver | Type | PSNR (4x) | GPU | Command |
|--------|------|-----------|-----|---------|
| Zero-filled | Traditional | ~27 dB | No | `operator.adjoint(y)` |
| PnP-ADMM | Plug-and-Play | 40.1 dB | Yes | `pnp_admm_mri(y, mask, sens)` |
| VarNet | Deep Unrolled | 48.2 dB | Yes | `varnet_mri(y, mask, sens)` |
| MoDL | Deep Unrolled | 39.5 dB | Yes | `modl_mri(y, mask, sens)` |

### Step 9d: Metrics

```python
# PSNR (magnitude image)
x_mag = np.abs(x_hat)
x_true_mag = np.abs(x_true)
psnr = 10 * np.log10(x_true_mag.max()**2 / np.mean((x_mag - x_true_mag)**2))

# SSIM (structural similarity on magnitude)
ssim = structural_similarity(x_mag, x_true_mag)

# NMSE (normalized mean squared error -- MRI-specific)
nmse = np.sum((x_hat - x_true)**2) / np.sum(x_true**2)

# HFEN (High-Frequency Error Norm -- MRI-specific)
# Captures edge fidelity lost in smoothing by deep networks
from scipy.ndimage import laplace
hfen = np.linalg.norm(laplace(x_mag) - laplace(x_true_mag)) / np.linalg.norm(laplace(x_true_mag))
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
|   +-- y.npy              # Multi-coil k-space (8, 320, 320) complex + SHA256
|   +-- x_hat.npy          # Reconstruction (320, 320) complex + SHA256
|   +-- x_true.npy         # Ground truth (if available) + SHA256
|   +-- sens_maps.npy      # Coil sensitivities (8, 320, 320) complex + SHA256
+-- metrics.json           # PSNR, SSIM, NMSE, HFEN
+-- operator.json          # Operator parameters (mask hash, n_coils, sens hash)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (PnP-ADMM -> VarNet -> MoDL) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **MRI-specific:** Gaussian noise model (not Poisson), complex-valued signals, multi-coil data consistency.
