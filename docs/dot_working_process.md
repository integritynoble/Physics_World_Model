# DOT Working Process

## End-to-End Pipeline for Diffuse Optical Tomography

This document traces a complete DOT experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct absorption maps from DOT boundary measurements.
 Measurements: boundary_data.npy, Geometry: source_detector_pos.json,
 16 sources, 16 detectors, 3 wavelengths (750/800/850 nm)."
```

---

## 2. PlanAgent — Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "boundary_data.npy" detected
#   operator_type=OperatorType.linear_operator,
#   files=["boundary_data.npy", "source_detector_pos.json"],
#   params={"n_sources": 16, "n_detectors": 16,
#           "wavelengths_nm": [750, 800, 850]}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> dot entry
dot:
  keywords: [DOT, diffuse_optical, NIR, tissue_imaging, absorption, scattering]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="dot",
#   confidence=0.95,
#   reasoning="Matched keywords: DOT, diffuse_optical, absorption"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the DOT registry entry:

```python
system = plan_agent.build_imaging_system("dot")
# ImagingSystem(
#   modality_key="dot",
#   display_name="Diffuse Optical Tomography",
#   signal_dims={"x": [64, 64, 64], "y": [256]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...3 elements...],
#   default_solver="born_approx"
# )
```

**DOT Element Chain (3 elements):**

```
NIR Laser Sources ---------> Tissue Medium ---------> Boundary Photodetectors
  throughput=1.0               throughput=0.01          throughput=0.70
  noise: none                  noise: shot_poisson      noise: shot+read+thermal
  3 wavelengths (750/800/850)  mu_a=0.01/mm             APD detectors (16)
  20 mW per source (16)       mu_s'=1.0/mm             0.5 A/W responsivity
  CW modulation                n=1.37, 60mm thick       10 pW/sqrt(Hz) NEP
                               diffusion model           200 MHz bandwidth
```

**Cumulative throughput:** `0.01 x 0.70 = 0.007`

Note: The extremely low tissue throughput (1%) is physically correct. NIR light at 750-850 nm is multiply scattered in tissue, and only a small fraction of photons reach boundary detectors after traversing 60 mm of tissue. This is the defining characteristic of DOT.

**Forward model equation:**
```
y = J(mu_a, mu_s') * delta_mu_a

J: Jacobian (sensitivity) matrix  [n_measurements x n_voxels]
J[m,v] = -V * G(r_s, r_v) * G(r_v, r_d) / G(r_s, r_d)

where:
  G(r1, r2) = exp(-k_d * |r1-r2|) / (4*pi*D*|r1-r2|)  (Green's function)
  k_d = sqrt(mu_a / D)                                   (effective wavenumber)
  D = 1 / (3 * (mu_a + mu_s'))                           (diffusion coefficient)
  V = voxel volume
```

---

## 3. PhotonAgent — SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  dot:
    model_id: "nir_tomography"
    parameters:
      power_w: 0.05
      wavelength_nm: 800
      na: 0.5
      n_medium: 1.4
      qe: 0.60
      exposure_s: 1.0
  ```

### Computation

```python
# 1. Photon energy at 800 nm
E_photon = h * c / wavelength_nm
#        = 6.626e-34 * 3e8 / (800e-9)
#        = 2.48e-19 J

# 2. Source power per source
P_per_source = 0.020  # 20 mW per laser diode (from modalities.yaml)

# 3. Collection efficiency
# DOT uses fiber-coupled detectors on tissue surface
# Collection NA ~ 0.5 (large-core fiber)
solid_angle = (na / n_medium)^2 / (4 * pi)
#           = (0.5 / 1.4)^2 / (4 * pi)
#           = 0.1276^2 = 0.01276 / 12.566
#           = 0.01015

# 4. Photons reaching boundary (through 60mm tissue)
# Tissue throughput: 1% (from element chain)
# This accounts for diffusion: photons undergo ~1000 scattering events
# Mean free path: 1/mu_s' = 1 mm, tissue thickness: 60 mm
N_raw = P_per_source * qe * solid_angle * exposure_s / E_photon
#     = 0.020 * 0.60 * 0.01015 * 1.0 / 2.48e-19
#     = 1.218e-4 / 2.48e-19
#     = 4.91e14 photons (launched into tissue)

# Through tissue (1% throughput)
N_through = 4.91e14 * 0.01 = 4.91e12 photons at boundary

# Per detector (16 detectors, but not all pairs useful)
# Effective: photons scale as exp(-mu_eff * d_sd) where d_sd is source-detector distance
# Near detectors: ~1e10 photons, far detectors: ~1e6 photons
N_near = 1e10
N_far  = 1e6

# 5. Detector noise
# APD NEP = 10 pW/sqrt(Hz), bandwidth = 200 MHz
noise_power = 10e-12 * sqrt(200e6) = 1.41e-7 W
# In photon terms: 1.41e-7 / (2.48e-19 * 200e6) = 2.85 photons/sample... negligible
# Dominant noise: shot noise from detected photons

# 6. SNR
# Worst case (far source-detector pair):
SNR_far = sqrt(N_far) = 1000
SNR_far_db = 20 * log10(1000) = 60.0 dB

# Best case (near pair):
SNR_near = sqrt(N_near) = 1e5
SNR_near_db = 20 * log10(1e5) = 100.0 dB

# Average across all 256 source-detector pairs:
SNR_avg_db = 75.0 dB  # Geometric mean
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=4.91e12,              # Total photons at boundary
  snr_db=75.0,
  noise_regime=NoiseRegime.shot_limited,     # Shot noise dominates
  shot_noise_sigma=2.22e6,
  read_noise_sigma=1000,                     # APD noise equivalent
  total_noise_sigma=2.22e6,
  feasible=True,
  quality_tier="excellent",                  # Average SNR >> 30 dB
  throughput_chain=[
    {"NIR Laser Sources": 1.0},
    {"Tissue Medium": 0.01},
    {"Boundary Photodetectors": 0.70}
  ],
  noise_model="poisson",
  explanation="Shot-noise-limited for near source-detector pairs. Far pairs "
              "(>40mm separation) have lower SNR but still adequate. "
              "Tissue scattering limits throughput to ~1%."
)
```

---

## 4. MismatchAgent — Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"dot"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  dot:
    parameters:
      source_position:
        range: [-3.0, 3.0]
        typical_error: 0.5
        unit: "mm"
        description: "Illumination fiber tip position error from optode placement"
      detector_position:
        range: [-3.0, 3.0]
        typical_error: 0.5
        unit: "mm"
        description: "Detection fiber position error from optode-tissue coupling"
      scattering_coeff:
        range: [0.5, 2.0]
        typical_error: 0.2
        unit: "1/mm"
        description: "Reduced scattering coefficient error from tissue heterogeneity"
    severity_weights:
      source_position: 0.30
      detector_position: 0.30
      scattering_coeff: 0.40
    correction_method: "gradient_descent"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.30 * |0.5| / 6.0       # source_position:     0.025
  + 0.30 * |0.5| / 6.0       # detector_position:   0.025
  + 0.40 * |0.2| / 1.5       # scattering_coeff:    0.053
S = 0.103  # Low severity

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.03 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="dot",
  mismatch_family="gradient_descent",
  parameters={
    "source_position":   {"typical_error": 0.5, "range": [-3, 3], "weight": 0.30},
    "detector_position": {"typical_error": 0.5, "range": [-3, 3], "weight": 0.30},
    "scattering_coeff":  {"typical_error": 0.2, "range": [0.5, 2.0], "weight": 0.40}
  },
  severity_score=0.103,
  correction_method="gradient_descent",
  expected_improvement_db=1.03,
  explanation="Low mismatch severity. Scattering coefficient uncertainty is the "
              "largest error source but within acceptable bounds for Born approximation."
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
  dot:
    signal_prior_class: "tv"
    entries:
      - {cr: 0.001, noise: "shot_limited", solver: "diffusion_model_dot",
         recoverability: 0.62, expected_psnr_db: 28.4,
         provenance: {dataset_id: "ucl_dot_breast_phantom_2023", ...}}
      - {cr: 0.001, noise: "detector_limited", solver: "tikhonov_dot",
         recoverability: 0.42, expected_psnr_db: 25.1, ...}
      - {cr: 0.001, noise: "shot_limited", solver: "tv_dot",
         recoverability: 0.55, expected_psnr_db: 27.0, ...}
  ```

### Computation

```python
# 1. Compression ratio
#    x = (64, 64, 64) = 262,144 voxels (3D absorption map)
#    y = (256,) = 256 measurements (16 sources x 16 detectors)
CR = prod(y_shape) / prod(x_shape) = 256 / 262144 = 0.000977 ~ 0.001
# SEVERELY ILL-POSED: 256 measurements to reconstruct 262,144 unknowns!
# This is the defining challenge of DOT.

# 2. Operator diversity
#    The Jacobian has exponentially decaying sensitivity:
#    Surface voxels: high sensitivity, deep voxels: near-zero sensitivity
#    Source-detector pairs provide tomographic diversity but
#    the system is vastly underdetermined
diversity = 0.2  # Low — severely ill-posed system

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.833  # High (poorly conditioned)

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="diffusion_model_dot", cr=0.001
#    -> recoverability=0.62, expected_psnr=28.4 dB

# 5. Best solver selection
#    diffusion_model_dot: 28.4 dB > tv_dot: 27.0 dB > tikhonov: 25.1 dB
#    -> recommended: "diffusion_model_dot" (best for DOT)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.001,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.tv,
  operator_diversity_score=0.2,
  condition_number_proxy=0.833,
  recoverability_score=0.62,
  recoverability_confidence=0.85,
  expected_psnr_db=28.4,
  expected_psnr_uncertainty_db=2.0,
  recommended_solver_family="diffusion_model_dot",
  verdict="sufficient",                  # 0.50 <= score < 0.70
  calibration_table_entry={...},
  explanation="Sufficient recoverability despite extreme ill-posedness (CR=0.001). "
              "DOT fundamentally limited by diffusion blurring. TV regularization "
              "preserves edges but spatial resolution is ~5-10 mm."
)
```

---

## 6. AnalysisAgent — Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(75.0 / 40, 1.0)   = 0.0     # Excellent SNR
mismatch_score    = 0.103                        = 0.103   # Low mismatch
compression_score = 1 - 0.62                     = 0.38    # Moderate recoverability
solver_score      = 0.25                         = 0.25    # Regularization sensitivity

# Primary bottleneck
primary = "compression"  # max(0.0, 0.103, 0.38, 0.25) = compression
# This correctly identifies the fundamental DOT limitation:
# 256 measurements for 262,144 unknowns is extremely underdetermined

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.103*0.5) * (1 - 0.38*0.5) * (1 - 0.25*0.5)
  = 1.0 * 0.949 * 0.81 * 0.875
  = 0.672
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="compression",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.103, compression=0.38, solver=0.25
  ),
  suggestions=[
    Suggestion(
      text="Use diffusion model prior for best DOT reconstruction (+1.4 dB over Tikhonov)",
      priority="high",
      expected_gain_db=1.4
    ),
    Suggestion(
      text="Increase source-detector count (32x32=1024 pairs) for 4x more measurements",
      priority="medium",
      expected_gain_db=3.0
    ),
    Suggestion(
      text="Add frequency-domain modulation for phase measurements (doubles data)",
      priority="low",
      expected_gain_db=2.0
    )
  ],
  overall_verdict="sufficient",          # 0.50 <= P < 0.70
  probability_of_success=0.672,
  explanation="Compression is the fundamental bottleneck. DOT is inherently "
              "severely ill-posed (CR=0.001). Spatial resolution limited to "
              "~5-10 mm by photon diffusion. Prior knowledge essential."
)
```

---

## 7. AgentNegotiator — Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" BUT verdict="sufficient" | **Close but no veto** |
| Severe mismatch without correction | severity=0.103 < 0.7 | No veto |
| All marginal | photon=excellent, compression=sufficient | No veto |
| Joint probability floor | P=0.672 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.62    # recoverability_score
P_mismatch       = 1.0 - 0.103 * 0.7 = 0.928

P_joint = 0.95 * 0.62 * 0.928 = 0.547
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.547
)
```

---

## 8. PreFlightReportBuilder — Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
# Jacobian construction dominates: O(n_src * n_det * n_vox)
n_voxels = 64 * 64 * 64 = 262144
n_pairs = 16 * 16 = 256
n_green_calls = n_pairs * n_voxels = 67,108,864

# Jacobian build: ~30s (Green's function evaluations)
# Born/Tikhonov solve: CG on (262144 x 262144) system, ~200 iters, ~10s
# L-BFGS-TV: 100 iterations with gradient, ~60s
runtime_s = 30.0 + 60.0 = 90.0 seconds
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="dot", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=90.0,
  proceed_recommended=True,
  warnings=[
    "Severely ill-posed (CR=0.001): 256 measurements for 262,144 unknowns",
    "Spatial resolution fundamentally limited to ~5-10 mm by photon diffusion"
  ],
  what_to_upload=[
    "Boundary measurement vector (N_source_detector_pairs)",
    "Source and detector positions on tissue surface (JSON)",
    "Background optical properties (mu_a, mu_s') if known"
  ]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# DOT forward model: y = J * delta_mu_a
#
# The Jacobian J is derived from the diffusion equation via the
# Born approximation. Each element J[m,v] represents the sensitivity
# of measurement m to an absorption change at voxel v.
#
# J[m,v] = -V * G(r_s, r_v) * G(r_v, r_d) / G(r_s, r_d)
#
# where G is the Green's function of the diffusion equation:
# G(r1, r2) = exp(-k_d * |r1-r2|) / (4*pi*D*|r1-r2|)
# k_d = sqrt(mu_a_bg / D)
# D = 1 / (3 * (mu_a_bg + mu_s'))
#
# Parameters:
#   source_positions:    (16, 3) source fiber locations
#   detector_positions:  (16, 3) detector fiber locations
#   volume_shape:        (64, 64, 64) reconstruction grid
#   mu_a_bg:             0.01 /mm (background absorption)
#   mu_s_prime:          1.0 /mm (reduced scattering)
#   voxel_volume:        1.0 mm^3
#
# Input:  x = (64, 64, 64) absorption coefficient map [delta_mu_a]
# Output: y = (256,) boundary measurements

class DOTOperator(PhysicsOperator):
    def __init__(self, src_pos, det_pos, vox_pos, mu_a_bg, mu_s_prime):
        self.jacobian = _build_jacobian(
            src_pos, det_pos, vox_pos, voxel_volume=1.0,
            mu_a_bg=mu_a_bg, mu_s_prime=mu_s_prime
        )

    def forward(self, x):
        """y = J * delta_mu_a"""
        delta_mu_a = (x - self.mu_a_bg).ravel()
        return self.jacobian @ delta_mu_a

    def adjoint(self, y):
        """delta_mu_a_hat = J^T * y"""
        return (self.jacobian.T @ y).reshape(self.volume_shape)

    def check_adjoint(self):
        """Verify <Jx, y> ~ <x, J^T y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided boundary_data.npy:
y = np.load("boundary_data.npy")     # (256,) or (n_wavelengths, 256)
geometry = json.load(open("source_detector_pos.json"))

# If simulating:
# Ground truth: 3D absorption phantom with spherical inclusions
x_true = np.ones((16, 16, 16)) * 0.01    # Background mu_a = 0.01/mm

# Spherical inclusions (tumor-like absorbers)
for inclusion in range(3):
    center = random_position()
    radius = random(2, 5)  # 2-5 mm
    mu_a_inclusion = random(0.02, 0.06)  # 2-6x background
    mask = (xx-cx)^2 + (yy-cy)^2 + (zz-cz)^2 < r^2
    x_true[mask] = mu_a_inclusion

# Source/detector positions on z=0 and z=nz-1 faces
source_positions = grid_on_face(z=0, n=12)
detector_positions = grid_on_face(y=0, n=12)

# Forward model
jacobian = _build_jacobian(source_pos, detector_pos, voxel_pos, voxel_volume=1.0)
delta_mu_a = (x_true - 0.01).ravel()
y = jacobian @ delta_mu_a
y += np.random.randn(*y.shape) * np.abs(y).max() * 0.01  # 1% noise
```

### Step 9c: Reconstruction with Born/Tikhonov

```python
from pwm_core.recon.dot_solver import born_approx, lbfgs_tv, _build_jacobian

# Normalize Jacobian for better conditioning
col_norms = np.sqrt(np.sum(jacobian**2, axis=0))
col_norms = np.maximum(col_norms, 1e-10)
J_normalized = jacobian / col_norms[np.newaxis, :]

# Algorithm 1: Born/Tikhonov (fast, standard)
recon_norm = born_approx(
    y=y,                           # (256,) boundary measurements
    jacobian=J_normalized,          # (256, 4096) normalized Jacobian
    alpha=1.0,                      # Tikhonov regularization parameter
    max_cg_iters=200                # Conjugate gradient iterations
)
# Solves: x = (J^T J + alpha*I)^{-1} J^T y via CG
# Denormalize: x_actual = x_norm / col_norms
recon_born = (recon_norm / col_norms).reshape(volume_shape) + 0.01
# Expected PSNR: ~25.0 dB on UCL breast phantom benchmark

# Algorithm 2: L-BFGS with Total Variation (higher quality)
recon_tv_flat = lbfgs_tv(
    y=y,
    jacobian=jacobian,
    volume_shape=(16, 16, 16),
    lambda_tv=0.001,                # TV regularization weight
    max_iters=100                   # L-BFGS-B iterations
)
# Minimizes: ||Jx - y||^2 + lambda_tv * TV(x)
# TV gradient computed via 3D finite differences
# Non-negativity constraint via bounds
# Expected PSNR: ~28.4 dB with diffusion model prior
recon_tv = recon_tv_flat.reshape(volume_shape) + 0.01
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| Born/Tikhonov | Traditional | 25.1 dB | No | `born_approx(y, J, alpha=1.0)` |
| TV-regularized | Traditional | 27.0 dB | No | `lbfgs_tv(y, J, shape, lambda_tv=0.001)` |
| Diffusion Model | Deep Learning | 28.4 dB | Yes | `diffusion_model_dot(y, J, shape)` |

### Step 9d: Metrics

```python
# PSNR on absorption map
psnr = 10 * log10(max_val^2 / mse(recon, x_true))
# ~25.1 dB (Tikhonov), ~28.4 dB (diffusion model)

# DOT-specific metrics:

# Contrast recovery: how much of the true contrast is recovered
# CR = (mu_a_inclusion_recon - mu_a_bg_recon) / (mu_a_inclusion_true - mu_a_bg_true)
contrast_recovery = (mean(recon[inclusion]) - mean(recon[bg])) / \
                    (mean(x_true[inclusion]) - mean(x_true[bg]))
# Typical: 30-60% for deep inclusions

# Localization accuracy (centroid error)
centroid_error_mm = norm(centroid(recon[inclusion]) - centroid(x_true[inclusion]))

# Size accuracy (volume of inclusion at FWHM)
fwhm_volume = count(recon > 0.5 * max(recon)) * voxel_volume
volume_error = abs(fwhm_volume - true_volume) / true_volume

# Absorption accuracy
mu_a_error = abs(mean(recon[inclusion]) - mean(x_true[inclusion]))
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
|   +-- y.npy              # Boundary measurements (256,) + SHA256 hash
|   +-- x_hat.npy          # Reconstructed volume (64, 64, 64) + SHA256 hash
|   +-- jacobian.npy        # Jacobian matrix (256, 262144) + SHA256 hash
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
+-- metrics.json           # PSNR, contrast recovery, localization, volume error
+-- operator.json          # Operator params (geometry hash, mu_a_bg, mu_s')
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized DOT pipeline with known background optical properties and well-placed optodes. In practice, real DOT systems face optode placement uncertainty (hand-placed fibers on patient), unknown background scattering (tissue heterogeneity), and contact coupling variations (fiber-tissue interface losses vary by 10-50%).

---

## Real Experiment: User Prompt

```
"Breast DOT scan for tumor detection. Optodes placed by hand on patient
 surface — positions are approximate (measured with ruler, ~2mm accuracy).
 Background scattering unknown. Some fibers may have poor contact.
 Data: breast_measurements.npy, Geometry: optode_positions.csv,
 12 sources, 12 detectors."
```

**Key difference:** Hand-placed optodes, unknown scattering, and contact coupling variation.

---

## R1. PlanAgent — Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.operator_correction,   # "positions approximate" detected
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["breast_measurements.npy", "optode_positions.csv"],
#   params={"n_sources": 12, "n_detectors": 12}
# )
```

---

## R2. PhotonAgent — Contact Coupling Variation

```python
# With 12x12=144 source-detector pairs (vs 256 ideal)
# Some fibers have poor contact: 20-50% signal loss
# Average coupling loss: 30%

N_effective_avg = 4.91e12 * 0.70 = 3.44e12  # 30% coupling loss
# Per source-detector pair, SNR varies widely:
# Good contact: SNR ~ 70 dB
# Poor contact: SNR ~ 40 dB
# Average: ~55 dB

PhotonReport(
  n_photons_per_pixel=3.44e12,
  snr_db=55.0,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="excellent",
  explanation="Shot-limited but coupling variation introduces systematic errors. "
              "Some source-detector pairs degraded by poor fiber contact."
)
```

---

## R3. MismatchAgent — Position Uncertainty + Scattering

```python
# Actual errors from hand-placement
psi_true = {
    "source_position":   2.0,     # 2 mm error (4x typical)
    "detector_position": 2.0,     # 2 mm error (4x typical)
    "scattering_coeff":  0.4,     # Unknown: assumed 1.0, actual 0.6 /mm
}

# Severity
S = 0.30 * |2.0| / 6.0       # source_position:     0.100
  + 0.30 * |2.0| / 6.0       # detector_position:   0.100
  + 0.40 * |0.4| / 1.5       # scattering_coeff:    0.107
S = 0.307  # Moderate severity

improvement_db = clip(10 * 0.307, 0, 20) = 3.07 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  severity_score=0.307,
  correction_method="gradient_descent",
  expected_improvement_db=3.07,
  explanation="Moderate severity. Optode position uncertainty (2 mm) and unknown "
              "scattering coefficient both contribute significantly. Jacobian must "
              "be recomputed with corrected parameters."
)
```

---

## R4. RecoverabilityAgent — Degraded by Fewer Pairs and Mismatch

```python
# Fewer measurements: 144 (12x12) vs 256 (16x16)
# CR = 144 / 262144 = 0.00055 (even worse than ideal)
# Calibration table: noise="detector_limited", solver="tikhonov_dot"
# -> recoverability=0.42, expected_psnr=25.1 dB

RecoverabilityReport(
  recoverability_score=0.42,
  expected_psnr_db=25.1,
  verdict="marginal",
  explanation="Marginal recoverability. Fewer source-detector pairs (144 vs 256), "
              "operator mismatch, and coupling variation all degrade reconstruction. "
              "Strong regularization essential."
)
```

---

## R5. AnalysisAgent — Everything is a Bottleneck

```python
photon_score      = 1 - min(55.0 / 40, 1.0)   = 0.0
mismatch_score    = 0.307
compression_score = 1 - 0.42                     = 0.58
solver_score      = 0.25

primary = "compression"  # max(0.0, 0.307, 0.58, 0.25)

P = (1-0.0*0.5) * (1-0.307*0.5) * (1-0.58*0.5) * (1-0.25*0.5)
  = 1.0 * 0.847 * 0.71 * 0.875
  = 0.526
```

```python
SystemAnalysis(
  primary_bottleneck="compression",
  probability_of_success=0.526,
  overall_verdict="marginal",
  suggestions=[
    Suggestion(text="Calibrate optode positions with 3D tracking system", priority="critical"),
    Suggestion(text="Estimate background scattering from homogeneous reference measurement", priority="high"),
    Suggestion(text="Apply coupling correction by normalizing per-channel intensities", priority="high"),
    Suggestion(text="Use diffusion model prior for maximum regularization benefit", priority="medium"),
  ]
)
```

---

## R6. AgentNegotiator — Conditional Proceed

```python
P_joint = 0.95 * 0.42 * (1 - 0.307*0.7) = 0.95 * 0.42 * 0.785 = 0.313

NegotiationResult(
  vetoes=[],
  proceed=True,         # P > 0.15
  probability_of_success=0.313
)
```

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=450.0,       # Includes scattering calibration + recompute Jacobian
  proceed_recommended=True,
  warnings=[
    "Severely ill-posed with only 144 measurements for 262,144 voxels (CR=0.00055)",
    "Optode positions uncertain (~2 mm) — Jacobian will be recalculated after calibration",
    "Background scattering unknown — will estimate from reference measurements",
    "Coupling variation 20-50% — per-channel normalization applied",
    "Expect spatial resolution ~10-15 mm (worse than ideal 5-10 mm)"
  ]
)
```

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Clinical |
|--------|-----------|---------------|
| **Photon Agent** | | |
| N_effective | 4.91e12 | 3.44e12 |
| SNR | 75.0 dB | 55.0 dB |
| Quality tier | excellent | excellent |
| **Mismatch Agent** | | |
| Severity | 0.103 (low) | 0.307 (moderate) |
| Dominant error | scattering | **position + scattering** |
| Expected gain | +1.03 dB | +3.07 dB |
| **Recoverability Agent** | | |
| Score | 0.62 (sufficient) | 0.42 (marginal) |
| Expected PSNR | 28.4 dB | 25.1 dB |
| Verdict | sufficient | **marginal** |
| **Analysis Agent** | | |
| Primary bottleneck | compression | compression |
| P(success) | 0.672 | 0.526 |
| **Negotiator** | | |
| P_joint | 0.547 | 0.313 |
| **PreFlight** | | |
| Runtime | 90s | 450s |
| Warnings | 2 | 5 |
| **Measurements** | 256 (16x16) | 144 (12x12) |
| **Effective CR** | 0.001 | 0.00055 |

---

## DOT-Specific: Understanding the Ill-Posedness

DOT is the most ill-posed modality in the PWM registry. Key physics:

```python
# Why DOT is so challenging:
#
# 1. Extreme compression: CR = 0.001 (256 measurements, 262,144 unknowns)
#    Compare to CASSI (CR=0.036) or FPM (CR=0.055)
#
# 2. Exponential sensitivity decay:
#    G(r1, r2) ~ exp(-k_d * |r1-r2|) / |r1-r2|
#    Surface voxels: J ~ 1e-3
#    Deep voxels:    J ~ 1e-9  (1,000,000x weaker!)
#
# 3. Diffusion limit on resolution:
#    Best possible resolution ~ 1/mu_s' = 1 mm (scattering mean free path)
#    Practical resolution: 5-15 mm depending on depth
#
# 4. Regularization controls everything:
#    alpha too large: over-smoothed, features disappear
#    alpha too small: noise-dominated, spurious artifacts
#    lambda_tv too large: staircase artifacts
#    lambda_tv too small: insufficient denoising
#
# This is why the recoverability score (0.62 ideal, 0.42 real) is the
# lowest of all 26 modalities in the PWM registry.
```

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (Tikhonov -> TV -> diffusion model) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Adaptive:** DOT-specific awareness of extreme ill-posedness, depth-dependent sensitivity, and mandatory strong regularization.
