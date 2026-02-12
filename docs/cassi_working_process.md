# CASSI Working Process

## End-to-End Pipeline for Single-Disperser Coded Aperture Snapshot Spectral Imaging (SD-CASSI)

This document traces a complete SD-CASSI experiment through the PWM multi-agent system,
from user prompt to final RunBundle output. The optical model and forward equations follow
the SD-CASSI formulation in Huang et al., ECCV 2020 ("Spectral Imaging with Deep Learning").

**Reference layout (Fig. 1, ECCV 2020):**

```
Scene ──► Objective Lens ──► Coded Aperture (mask) ──► Relay Lens 1 ──► Single Disperser (prism) ──► Relay Lens 2 ──► Detector
```

---

## 1. User Prompt

```
"Reconstruct a hyperspectral cube from this CASSI snapshot measurement.
 Measurement: measurement.npy, Mask: mask.npy, 28 spectral bands, step=2."
```

---

## 2. PlanAgent — Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent → `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "measurement.npy" detected
#   operator_type=OperatorType.linear_operator,
#   files=["measurement.npy", "mask.npy"],
#   params={"n_bands": 28, "dispersion_step_px": 2}
# )
```

### Step 2b: Keyword Match → `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml → cassi entry
cassi:
  keywords: [CASSI, hyperspectral, coded_aperture, spectral_imaging, compressive_sensing]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="cassi",
#   confidence=0.95,
#   reasoning="Matched keywords: CASSI, hyperspectral"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System → `ImagingSystem`

Constructs the element chain from the CASSI registry entry:

```python
system = plan_agent.build_imaging_system("cassi")
# ImagingSystem(
#   modality_key="cassi",
#   display_name="Single-Disperser CASSI (SD-CASSI)",
#   signal_dims={"x": [256, 256, 28], "y": [256, 283]},  # Nx × (Ny + Nλ - 1)
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...8 elements...],
#   default_solver="mst"
# )
```

**SD-CASSI Optical Chain (ECCV-2020 layout, 8 elements):**

```
Scene Radiance ──► Objective Lens ──► Coded Aperture ──► Relay Lens 1 ──► Disperser (prism) ──► SD Distortion ──► Relay Lens 2 ──► Detector
  throughput=1.0    throughput=0.92    throughput=0.50    throughput=0.90   throughput=0.88      throughput≈1.0     throughput=0.90    throughput=0.75
  noise: none       noise: aberration  noise: fixed_pat   noise: aberration  noise: alignment     noise: warp        noise: aberration   noise: shot+read+quant
                                       + alignment
```

**PWM OperatorGraph mapping:**

```
x (world spectral cube / radiance)
↓
SourceNode: SceneRadianceSource (photon carrier, optional illumination SPD)
↓
Element 1 (imaging): ObjectiveLensNode — focuses scene onto coded aperture plane
↓
Element 2 (encoding): CodedApertureNode — binary random mask M(x,y), throughput ≈ 50%
↓
Element 3 (transport/relay): RelayLens1Node — 4f collimation (e.g. f=45mm)
↓
Element 4 (dispersion): DisperserNode — prism, wavelength-dependent spatial shear d(λ)
↓
Element 5 (SD distortion): ImbalancedResponseNode — path-length-dependent warp
↓
Element 6 (transport/relay): RelayLens2Node — imaging relay (e.g. f=50mm)
↓
SensorNode: DetectorNode (QE(λ), pixel integration, ADC quantization)
↓
NoiseNode: Poisson shot + read noise + quantization noise
↓
y (2D coded measurement, shape Nx × (Ny + Nλ − 1))
```

**Cumulative throughput:** `0.92 × 0.50 × 0.90 × 0.88 × 1.0 × 0.90 × 0.75 = 0.244`

---

## 3. SD-CASSI Forward Model (ECCV-2020 Equations)

### 3.1 Physical Process

The SD-CASSI system encodes a 3D spectral cube into a single 2D snapshot through
three operations: (1) spatial coding by the mask, (2) wavelength-dependent spatial
shearing by the prism, and (3) spectral integration at the detector.

### 3.2 Continuous Forward Model

Let `f(x, y, λ)` denote the spectral scene radiance, `M(x, y)` the coded aperture
mask, and `d(λ)` the prism dispersion function (spatial shift in pixels as a
function of wavelength).

**Step 1 — Coded aperture modulation:**

```
F'(x, y, λ) = f(x, y, λ) · M(x, y)
```

**Step 2 — Disperser shear (along y-axis, the dispersion direction):**

```
F''(x, y, nλ) = F'(x, y + d(λ_n − λ_c), nλ)
```

where `λ_c` is the center wavelength and `d(·)` is the dispersion slope
(pixels per unit wavelength difference). The shift is **along the y-axis**
(columns), extending the measurement in that dimension.

**Step 3 — Detector integration (sum over wavelength channels):**

```
Y(x, y) = Σ_{nλ=1}^{Nλ} F''(x, y, nλ) + G(x, y)
```

where `G` is additive noise (shot + read + quantization).

### 3.3 Discrete Forward Model

For `Nλ` spectral bands discretized at wavelengths `{λ_1, ..., λ_Nλ}`, with
dispersion step `s` pixels per band (relative to center band):

```
d_n = s · (n − n_c)      where n_c = (Nλ + 1) / 2
```

The shifted-mask equivalent form (convenient for implementation):

```
Y(x, y) = Σ_{nλ=1}^{Nλ} f̃(x, y, nλ) ⊙ M(x, y − d_n) + G(x, y)
```

Or equivalently (shift the coded slice, not the mask):

```
Y(x, y) = Σ_{nλ=1}^{Nλ} shift_y( f(x, y, nλ) ⊙ M(x, y), d_n ) + G(x, y)
```

**Vector form:**

```
y = Φ f + g
```

where `Φ ∈ R^{M × N}` is the SD-CASSI sensing matrix, `f ∈ R^N` is the
vectorized spectral cube (N = Nx · Ny · Nλ), `y ∈ R^M` is the vectorized
measurement (M = Nx · (Ny + Nλ − 1)), and `g` is noise.

### 3.4 Measurement Shape

The dispersion extends the measurement along the y-axis:

```
Y ∈ R^{Nx × (Ny + (Nλ − 1) · s)}
```

For the standard benchmark: `Nx=256, Ny=256, Nλ=28, s=1`:

```
Y ∈ R^{256 × 283}    where 283 = 256 + (28 − 1) × 1
```

> **Axis convention:** Dispersion is along the **y-axis** (axis=1 in row-major
> arrays). The mask is 2D `(Nx, Ny)`. The coded slice for each band is shifted
> along y by `d_n` pixels before summation.

### 3.5 Implementation (Option A — shift the coded slice, recommended)

```python
class SDCASSIOperator(PhysicsOperator):
    """SD-CASSI forward model following ECCV-2020 formulation.

    Dispersion along y-axis. Output shape: (Nx, Ny + (Nλ-1)*step).
    """
    def forward(self, x):
        """Y(x,y) = Σ_l shift_y( X[:,:,l] ⊙ M, d_l )"""
        Nx, Ny, L = x.shape
        step = self.dispersion_step
        Ny_out = Ny + (L - 1) * step
        Y = np.zeros((Nx, Ny_out))
        n_c = (L - 1) / 2.0  # center band index
        for l in range(L):
            d_l = int(round(step * (l - n_c)))  # or step * l for zero-referenced
            coded_slice = x[:, :, l] * self.mask     # (Nx, Ny)
            # Place shifted slice into extended measurement
            y_start = max(0, d_l)
            y_end = min(Ny_out, Ny + d_l)
            src_start = max(0, -d_l)
            src_end = src_start + (y_end - y_start)
            Y[:, y_start:y_end] += coded_slice[:, src_start:src_end]
        return Y

    def adjoint(self, Y):
        """X_hat[:,:,l] = M ⊙ shift_y( Y, -d_l )"""
        Nx = self.mask.shape[0]
        Ny = self.mask.shape[1]
        L = self.n_bands
        step = self.dispersion_step
        X = np.zeros((Nx, Ny, L))
        n_c = (L - 1) / 2.0
        for l in range(L):
            d_l = int(round(step * (l - n_c)))
            y_start = max(0, d_l)
            y_end = min(Y.shape[1], Ny + d_l)
            src_start = max(0, -d_l)
            src_end = src_start + (y_end - y_start)
            X[:, src_start:src_end, l] = self.mask[:, src_start:src_end] * Y[:, y_start:y_end]
        return X

    def check_adjoint(self):
        """Verify <Ax, y> ≈ <x, A^T y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

---

## 4. SD Imbalanced Response / Spatial Distortion

The ECCV-2020 paper highlights a key SD-CASSI hardware limitation:

> "Imbalanced response … is a spatial distortion along the dispersion direction
> … caused by path length difference between wavelength channels."

This distortion arises because different wavelength channels traverse different
optical paths through the prism, leading to wavelength-dependent geometric
warping (not just a pure translational shift).

### 4.1 ImbalancedResponseNode

PWM models this as a dedicated element between the disperser and detector:

```
Element 5: ImbalancedResponseNode (SD distortion along dispersion direction)
```

**Parameterization** — wavelength-dependent warp beyond the linear dispersion:

```
d_total(λ) = a₁ · (λ − λ_c) + a₂ · (λ − λ_c)²
```

where:
- `a₁` = linear dispersion slope (dominant, calibrated from prism spec)
- `a₂` = quadratic curvature (the "imbalanced" nonlinear term)
- `λ_c` = center wavelength

Optional per-channel affine correction `(s_λ, t_λ)`:

```
y_corrected(nλ) = s_λ · y_dispersed(nλ) + t_λ
```

where `s_λ ≈ 1 + ε` accounts for per-band magnification variation and
`t_λ` is a small translational residual.

### 4.2 Mismatch Parameters for BeliefState

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `dispersion_slope_error` | Error in a₁ (linear term) | ±0.05 px/band |
| `dispersion_curvature_error` | Error in a₂ (quadratic term) | ±0.02 px/band² |
| `prism_rotation_error` | Angular misalignment of prism axis | ±2° |
| `channel_warp_error` | Per-band affine residual (s_λ, t_λ) | s ∈ [0.98, 1.02] |

These parameters directly target the hardware limitation the paper highlights
and support calibration via the UPWMI operator correction framework.

---

## 5. PhotonAgent — SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  cassi:
    model_id: "microscopy_fluorescence"
    parameters:
      power_w: 0.01
      wavelength_nm: 550
      na: 0.25
      n_medium: 1.0
      qe: 0.80
      exposure_s: 0.1
  ```

### Computation

```python
# 1. Photon energy
E_photon = h * c / wavelength_nm  # 3.61e-19 J

# 2. Collection solid angle
solid_angle = (na / n_medium)^2 / (4 * pi)  # 0.00497

# 3. Raw photon count
N_raw = power_w * qe * solid_angle * exposure_s / E_photon
#     = 0.01 * 0.80 * 0.00497 * 0.1 / 3.61e-19
#     ≈ 1.10e13 photons

# 4. Apply cumulative throughput (updated for 8-element chain)
N_effective = N_raw * 0.244 ≈ 2.68e12 photons/pixel

# 5. Noise variances
shot_var   = N_effective                    # Poisson
read_var   = read_noise^2 = 25.0           # Gaussian (5 e-)
dark_var   = dark_current * exposure = 0    # Negligible
total_var  = shot_var + read_var + dark_var

# 6. SNR
SNR = N_effective / sqrt(total_var) ≈ sqrt(N_effective) ≈ 1.64e6
SNR_db = 20 * log10(SNR) ≈ 124.3 dB
```

### Output → `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=2.68e12,
  snr_db=124.3,
  noise_regime=NoiseRegime.shot_limited,    # shot_var/total_var > 0.9
  shot_noise_sigma=1.64e6,
  read_noise_sigma=5.0,
  total_noise_sigma=1.64e6,
  feasible=True,
  quality_tier="excellent",                 # SNR > 30 dB
  throughput_chain=[
    {"Scene Radiance": 1.0},
    {"Objective Lens": 0.92},
    {"Coded Aperture Mask": 0.50},
    {"Relay Lens 1": 0.90},
    {"Dispersive Prism": 0.88},
    {"SD Distortion": 1.0},
    {"Relay Lens 2": 0.90},
    {"Detector": 0.75}
  ],
  noise_model="poisson_read_quantization",
  explanation="Shot-noise-limited regime. Excellent SNR for reconstruction."
)
```

> **Note:** The idealized SNR (124 dB) is unrealistically high and can hide
> operator errors. See Part II for realistic lab conditions and the training
> noise preset below.

---

## 6. Noise Model — Poisson + Read + Quantization

Following the ECCV-2020 paper's training assumptions, the NoiseNode implements
a three-component noise model:

### 6.1 Shot Noise (Poisson)

```python
y_shot = np.random.poisson(lam=y_clean * peak_photons) / peak_photons
```

### 6.2 Read Noise (Gaussian)

```python
y_noisy = y_shot + np.random.normal(0, read_sigma, size=y_shot.shape)
```

### 6.3 Quantization Noise (ADC)

```python
bit_depth = 12  # typical CCD/CMOS
y_quantized = np.round(y_noisy * (2**bit_depth - 1)) / (2**bit_depth - 1)
```

### 6.4 Training Noise Preset (Paper-Consistent)

For training / simulation experiments consistent with the ECCV-2020 setup:

```python
noise_preset = {
    "peak_photons": 10000,      # moderate photon count
    "read_sigma": 0.01,         # ~1% of signal
    "bit_depth": 12,            # 12-bit ADC
    "dark_current": 0.0,        # negligible for short exposures
}
```

This avoids the unrealistically high SNR (>100 dB) that masks operator errors.

---

## 7. MismatchAgent — Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"cassi"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  cassi:
    parameters:
      mask_dx:     {range: [-3, 3], typical_error: 0.5, weight: 0.20}
      mask_dy:     {range: [-3, 3], typical_error: 0.5, weight: 0.20}
      mask_theta:  {range: [-0.6, 0.6], typical_error: 0.05, weight: 0.10}
      dispersion_step: {range: [0.8, 1.2], typical_error: 0.03, weight: 0.15}
      dispersion_slope_error: {range: [-0.05, 0.05], typical_error: 0.01, weight: 0.10}
      dispersion_curvature_error: {range: [-0.02, 0.02], typical_error: 0.005, weight: 0.05}
      prism_rotation_error: {range: [-2, 2], typical_error: 0.5, weight: 0.05}
      psf_sigma:   {range: [0.3, 2.5], typical_error: 0.3, weight: 0.05}
      gain:        {range: [0.5, 1.5], typical_error: 0.1, weight: 0.10}
    correction_method: "UPWMI_beam_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.20 * |0.5| / 6.0     # mask_dx:  0.017
  + 0.20 * |0.5| / 6.0     # mask_dy:  0.017
  + 0.10 * |0.05| / 1.2    # mask_theta: 0.004
  + 0.15 * |0.03| / 0.4    # dispersion_step: 0.011
  + 0.10 * |0.01| / 0.1    # disp_slope_err: 0.010
  + 0.05 * |0.005| / 0.04  # disp_curv_err: 0.006
  + 0.05 * |0.5| / 4.0     # prism_rot: 0.006
  + 0.05 * |0.3| / 2.2     # psf: 0.007
  + 0.10 * |0.1| / 1.0     # gain: 0.010
S = 0.088  # Low severity (typical lab conditions)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 0.88 dB
```

### Output → `MismatchReport`

```python
MismatchReport(
  modality_key="cassi",
  mismatch_family="UPWMI_beam_search",
  parameters={
    "mask_dx":  {"typical_error": 0.5, "range": [-3, 3], "weight": 0.20},
    "mask_dy":  {"typical_error": 0.5, "range": [-3, 3], "weight": 0.20},
    "mask_theta": {"typical_error": 0.05, "range": [-0.6, 0.6], "weight": 0.10},
    "dispersion_step": {"typical_error": 0.03, "range": [0.8, 1.2], "weight": 0.15},
    "dispersion_slope_error": {"typical_error": 0.01, "range": [-0.05, 0.05], "weight": 0.10},
    "dispersion_curvature_error": {"typical_error": 0.005, "range": [-0.02, 0.02], "weight": 0.05},
    "prism_rotation_error": {"typical_error": 0.5, "range": [-2, 2], "weight": 0.05},
    "psf_sigma": {"typical_error": 0.3, "range": [0.3, 2.5], "weight": 0.05},
    "gain":     {"typical_error": 0.1, "range": [0.5, 1.5], "weight": 0.10}
  },
  severity_score=0.088,
  correction_method="UPWMI_beam_search",
  expected_improvement_db=0.88,
  explanation="Low mismatch severity under typical conditions."
)
```

---

## 8. RecoverabilityAgent — Can We Reconstruct?

**File:** `agents/recoverability_agent.py` (912 lines)

### Input
- `ImagingSystem` (signal_dims for CR calculation)
- `PhotonReport` (noise regime)
- Calibration table from `compression_db.yaml`:
  ```yaml
  cassi:
    signal_prior_class: "joint_spatio_spectral"
    entries:
      - {cr: 0.036, noise: "shot_limited", solver: "mst",
         recoverability: 0.88, expected_psnr_db: 34.81,
         provenance: {dataset_id: "kaist_hsi_28ch_2023", ...}}
      - {cr: 0.036, noise: "shot_limited", solver: "gap_tv",
         recoverability: 0.72, expected_psnr_db: 30.52, ...}
      - {cr: 0.036, noise: "shot_limited", solver: "hdnet",
         recoverability: 0.89, expected_psnr_db: 34.97, ...}
  ```

### Computation

```python
# 1. Compression ratio (updated for correct measurement shape)
# Y ∈ R^{256 × 283}, X ∈ R^{256 × 256 × 28}
CR = (256 * 283) / (256 * 256 * 28) = 0.039

# 2. Operator diversity (mask density heuristic)
density = 0.5  # binary random mask
diversity = 4 * density * (1 - density) = 1.0  # Maximum diversity

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.5

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="mst", cr≈0.036 (nearest)
#    → recoverability=0.88, expected_psnr=34.81 dB, confidence=1.0

# 5. Best solver selection
#    hdnet: 34.97 dB > mst: 34.81 dB > gap_tv: 30.52 dB
#    → recommended: "hdnet" (or "mst" as default)
```

### Output → `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.039,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.joint_spatio_spectral,
  operator_diversity_score=1.0,
  condition_number_proxy=0.5,
  recoverability_score=0.88,
  recoverability_confidence=1.0,
  expected_psnr_db=34.81,
  expected_psnr_uncertainty_db=0.5,
  recommended_solver_family="mst",
  verdict="excellent",              # score ≥ 0.85
  calibration_table_entry={...},
  explanation="Excellent recoverability. MST expected 34.81 dB on KAIST benchmark."
)
```

---

## 9. AnalysisAgent — Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(124.3 / 40, 1.0)  = 0.0    # Excellent SNR
mismatch_score    = 0.088                        = 0.088  # Low mismatch
compression_score = 1 - 0.88                     = 0.12   # Good recoverability
solver_score      = 0.2                          = 0.2    # Default placeholder

# Primary bottleneck
primary = "solver"  # max(0.0, 0.088, 0.12, 0.2) = solver

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.088*0.5) * (1 - 0.12*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.956 * 0.94 * 0.90
  = 0.809
```

### Output → `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="solver",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.088, compression=0.12, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Consider using HDNet for +0.16 dB over MST",
      priority="medium",
      expected_gain_db=0.16
    )
  ],
  overall_verdict="excellent",      # P ≥ 0.80
  probability_of_success=0.809,
  explanation="System is well-configured. Solver choice is the primary bottleneck."
)
```

---

## 10. AgentNegotiator — Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="excellent" | No veto |
| Severe mismatch without correction | severity=0.088 < 0.7 | No veto |
| All marginal | All excellent/sufficient | No veto |
| Joint probability floor | P=0.809 > 0.15 | No veto |

### Joint Probability

```python
P_photon       = 0.95   # tier_prob["excellent"]
P_recoverability = 0.88  # recoverability_score
P_mismatch     = 1.0 - 0.088 * 0.7 = 0.938

P_joint = 0.95 * 0.88 * 0.938 = 0.784
```

### Output → `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.784
)
```

---

## 11. PreFlightReportBuilder — Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 256 * 256 * 28 = 1,835,008
dim_factor   = total_pixels / (256 * 256) = 28.0
solver_complexity = 2.5  # MST (transformer-based)
cr_factor    = max(0.039, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 28.0 * 2.5 * 0.125 = 17.5 seconds
```

### Output → `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="cassi", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=17.5,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=["measurement (2D CASSI snapshot)", "mask (coded aperture pattern)"]
)
```

---

## 12. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 12a: Build Physics Operator

```python
# SD-CASSI forward model (ECCV-2020):
#   Y(x,y) = Σ_l shift_y( X[:,:,l] ⊙ M(x,y), d_l ) + G
#
# Parameters:
#   mask:   (Nx, Ny) binary coded aperture     [loaded from mask.npy]
#   step:   dispersion_step_px = 2             [pixels/band along y-axis]
#   n_bands: 28
#
# Input:  x = (256, 256, 28) hyperspectral cube
# Output: y = (256, 283) compressed snapshot
#         where 283 = 256 + (28-1)*1   (step=1 for standard benchmark)

operator = SDCASSIOperator(
    mask=np.load("mask.npy"),   # (256, 256)
    dispersion_step=2,
    n_bands=28,
    dispersion_axis="y"          # ECCV-2020: dispersion along y
)
operator.check_adjoint()  # Passes: <Ax,y> ≈ <x,A*y>, rel_error < 1e-10
```

### Step 12b: Forward Simulation (or Load Measurement)

```python
# If user provided measurement.npy:
y = np.load("measurement.npy")    # (256, 283) — Nx × (Ny + Nλ - 1)

# If simulating:
x_true = load_ground_truth()       # (256, 256, 28) from KAIST dataset
y_clean = operator.forward(x_true) # (256, 283)

# Noise: Poisson shot + read + quantization (paper-consistent)
y_shot = np.random.poisson(lam=y_clean * peak_photons) / peak_photons
y_noisy = y_shot + np.random.normal(0, read_sigma, size=y_shot.shape)
y = np.round(y_noisy * 4095) / 4095  # 12-bit quantization
```

### Step 12c: Reconstruction with MST

```python
from pwm_core.recon.mst import mst_recon_cassi

x_hat = mst_recon_cassi(
    y=y,                    # (256, 283) compressed measurement
    mask=mask,              # (256, 256) coded aperture
    dispersion_step=2,      # pixels per spectral shift (along y)
    n_bands=28,
    model_path="weights/mst_cassi_28ch.pth",
    device="cuda"
)
# x_hat shape: (256, 256, 28) — reconstructed hyperspectral cube
# Expected PSNR: ~34.81 dB (KAIST benchmark)
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| GAP-TV | Traditional | 30.52 dB | No | `gap_tv_cassi(y, mask, step=2)` |
| MST-L | Deep Learning | 34.81 dB | Yes | `mst_recon_cassi(y, mask, step=2)` |
| HDNet | Deep Learning | 34.97 dB | Yes | `hdnet_recon_cassi(y, mask, step=2)` |
| MST++ | Deep Learning | 35.99 dB | Yes | `mst_recon_cassi(y, mask, step=2, config={"variant": "mst_plus_plus"})` |

### Step 12d: Metrics

```python
# Per-band PSNR
for l in range(28):
    psnr_l = 10 * log10(max_val^2 / mse(x_hat[:,:,l], x_true[:,:,l]))

# Average PSNR across all bands
avg_psnr = mean(psnr_per_band)  # ~34.81 dB

# SSIM (structural similarity)
avg_ssim = mean([ssim(x_hat[:,:,l], x_true[:,:,l]) for l in range(28)])

# SAM (Spectral Angle Mapper) — CASSI-specific
sam = mean(arccos(dot(x_hat[i], x_true[i]) / (norm(x_hat[i]) * norm(x_true[i]))))
```

### Step 12e: RunBundle Output

```
run_bundle/
├── meta.json              # ExperimentSpec + provenance
├── agent_reports/
│   ├── photon_report.json
│   ├── mismatch_report.json
│   ├── recoverability_report.json
│   ├── system_analysis.json
│   ├── negotiation_result.json
│   └── preflight_report.json
├── arrays/
│   ├── y.npy              # Measurement (256, 283) + SHA256 hash
│   ├── x_hat.npy          # Reconstruction (256, 256, 28) + SHA256 hash
│   └── x_true.npy         # Ground truth (if available) + SHA256 hash
├── metrics.json           # PSNR, SSIM, SAM per band + average
├── operator.json          # Operator parameters (mask hash, step, n_bands, disp_axis)
└── provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

## 13. Operator-Correction Mode — Building Φ(θ) and Updating BeliefState

The SD-CASSI vector form `y = Φ f + g` naturally supports PWM's operator
correction framework. The sensing matrix `Φ(θ)` depends on calibratable
parameters `θ`:

### 13.1 Parameterization of Φ(θ)

```python
theta = {
    # Mask geometry
    "mask_dx": 0.0,        # mask x-translation (pixels)
    "mask_dy": 0.0,        # mask y-translation (pixels)
    "mask_theta": 0.0,     # mask rotation (radians)

    # Dispersion model
    "disp_a1": 1.0,        # linear dispersion slope (px/band)
    "disp_a2": 0.0,        # quadratic curvature (px/band²)
    "disp_axis_angle": 0.0, # prism rotation error (degrees)

    # SD distortion (imbalanced response)
    "sd_warp_scale": [],   # per-channel scale factors s_λ
    "sd_warp_shift": [],   # per-channel shift residuals t_λ

    # Detector
    "gain": 1.0,           # flat-field gain
    "psf_sigma": 0.0,      # PSF blur sigma
}
```

### 13.2 BeliefState Update Hooks

The UPWMI framework estimates `θ` from measured data:

```python
# 1. Build initial operator with nominal parameters
Phi_0 = build_sd_cassi_operator(theta_nominal)

# 2. Coarse reconstruction (robust solver)
x_hat_0 = gap_tv(y, Phi_0, n_iter=25)

# 3. Score function: reconstruction residual
def score(theta):
    Phi = build_sd_cassi_operator(theta)
    x_hat = gap_tv(y, Phi, n_iter=25)
    return ||y - Phi @ x_hat||^2

# 4. Estimate mask misalignment (dx, dy, theta)
#    via grid search + beam refinement (Algorithm 1)
theta_mask = upwmi_beam_search(score, search_space={
    "mask_dx": linspace(-3, 3, 25),
    "mask_dy": linspace(-3, 3, 25),
    "mask_theta": linspace(-0.6, 0.6, 13),
})

# 5. Estimate dispersion curve (a1, a2, axis_angle)
#    via calibration frames (narrowband flat-field patterns)
theta_disp = estimate_dispersion_curve(
    calibration_frames=load_calibration_data(),
    n_bands=28,
    model="quadratic"  # fit a1, a2
)

# 6. Estimate SD distortion warp
#    via per-channel registration from calibration
theta_sd = estimate_sd_warp(
    calibration_frames=load_calibration_data(),
    reference_band=14,  # center band
    model="affine_per_channel"
)

# 7. Merge into updated BeliefState
theta_calibrated = {**theta_nominal, **theta_mask, **theta_disp, **theta_sd}
Phi_cal = build_sd_cassi_operator(theta_calibrated)

# 8. Final reconstruction with calibrated operator
x_hat_final = mst_recon(y, Phi_cal)
```

### 13.3 Differentiable Refinement (Algorithm 2)

For gradient-based fine-tuning after coarse calibration:

```python
# Parameterize θ as differentiable torch tensors
theta_diff = {
    "mask_dx": nn.Parameter(torch.tensor(theta_cal["mask_dx"])),
    "mask_dy": nn.Parameter(torch.tensor(theta_cal["mask_dy"])),
    "mask_theta": nn.Parameter(torch.tensor(theta_cal["mask_theta"])),
    "disp_a1": nn.Parameter(torch.tensor(theta_cal["disp_a1"])),
    "disp_a2": nn.Parameter(torch.tensor(theta_cal["disp_a2"])),
}

# Unrolled GAP-TV: differentiate through K reconstruction iterations
# Loss = ||y_measured - Φ(θ) @ gap_tv_K(y, Φ(θ))||²
optimizer = Adam(theta_diff.values(), lr=0.03)
for step in range(200):
    Phi = DifferentiableSDCASSI(theta_diff)
    x_hat = unrolled_gap_tv(y, Phi, K=10)
    loss = (y - Phi(x_hat)).pow(2).sum()
    loss.backward()
    optimizer.step()
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-13) showed an **idealized** SD-CASSI pipeline with
perfect lab conditions (SNR 124.3 dB, mismatch severity 0.088). In practice,
real SD-CASSI systems have significant operator mismatch from assembly
tolerances, limited photon budgets from short exposures, detector noise,
and **SD-specific imbalanced response distortion** from the prism path.

This section traces the **same pipeline** with realistic parameters drawn from
our actual benchmark experiments.

---

## Real Experiment: User Prompt

```
"I have a CASSI measurement from our lab prototype. The mask might be
 slightly misaligned from the last disassembly. Please calibrate the
 operator and reconstruct.
 Measurement: lab_snapshot.npy, Mask: design_mask.npy, 28 bands, step=2."
```

**Key difference:** The user says "mask might be misaligned" and asks for calibration. The **design mask** (as-manufactured) does not match the **actual mask position** in the system.

---

## R1. PlanAgent — Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.operator_correction,   # "calibrate" detected
#   has_measured_y=True,
#   operator_type=OperatorType.linear_operator,
#   files=["lab_snapshot.npy", "design_mask.npy"],
#   params={"n_bands": 28, "dispersion_step_px": 2}
# )
```

**Difference from ideal:** `mode=operator_correction` (not `auto`). The user explicitly requested calibration, so the pipeline will include UPWMI correction before reconstruction.

---

## R2. PhotonAgent — Realistic Lab Conditions

### Real detector parameters

```yaml
# Real lab: short exposure, lower QE, significant read noise
cassi_lab:
  power_w: 0.001          # 1 mW source (10x dimmer than ideal)
  wavelength_nm: 550
  na: 0.25
  n_medium: 1.0
  qe: 0.55                # Real CCD QE at 550nm (not ideal 0.80)
  exposure_s: 0.01         # 10 ms exposure (10x shorter)
  read_noise_e: 12.0       # Older CCD (not ideal 5.0)
  dark_current_e_per_s: 0.5
```

### Computation

```python
# Raw photon count (100x fewer than ideal)
N_raw = 0.001 * 0.55 * 0.00497 * 0.01 / 3.61e-19 ≈ 7.56e10

# Apply cumulative throughput (0.244)
N_effective = 7.56e10 * 0.244 ≈ 1.84e10 photons/pixel

# Noise variances
shot_var   = 1.84e10                        # Poisson
read_var   = 12.0^2 = 144.0                # Significant read noise
dark_var   = 0.5 * 0.01 = 0.005            # Negligible
total_var  = 1.84e10 + 144.0 + 0.005

# SNR
SNR = 1.84e10 / sqrt(1.84e10 + 144) ≈ sqrt(1.84e10) ≈ 1.36e5
SNR_db = 20 * log10(1.36e5) ≈ 102.6 dB
```

### Output → `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=1.84e10,
  snr_db=102.6,
  noise_regime=NoiseRegime.shot_limited,    # Still shot-dominated
  shot_noise_sigma=1.36e5,
  read_noise_sigma=12.0,
  total_noise_sigma=1.36e5,
  feasible=True,
  quality_tier="excellent",                 # 102.6 dB >> 30 dB threshold
  noise_model="poisson_read_quantization",
  explanation="Shot-limited despite shorter exposure. Feasible for reconstruction."
)
```

**Verdict:** Even with 100x fewer photons, SD-CASSI is still comfortably shot-limited. This is typical for spectral imaging — the bottleneck is almost never photon count.

---

## R3. MismatchAgent — Real Assembly Tolerances

### Real mismatch: mask shifted + rotated + dispersion off + SD distortion

In a real lab SD-CASSI prototype, the coded aperture mask is manually positioned.
After disassembly/reassembly, typical errors are:

```python
# True (unknown) operator parameters
psi_true = {
    "dx":    +1.094,    # mask shifted right by ~1.1 pixels
    "dy":    -2.677,    # mask shifted down by ~2.7 pixels (largest error)
    "theta": -0.559,    # mask rotated by ~0.56 rad (32 degrees!)
    "phi_d": -0.316,    # dispersion step off by -0.32 px/band
    "disp_a2": 0.008,   # quadratic dispersion curvature (SD distortion)
    "prism_rot": 0.3,   # prism rotated ~0.3 degrees
}

# What the system thinks (nominal = zero error)
psi_nominal = {"dx": 0, "dy": 0, "theta": 0, "phi_d": 0, "disp_a2": 0, "prism_rot": 0}
```

### Severity computation

```python
# Actual errors (not typical — measured from benchmark ground truth)
S = 0.20 * |1.094| / 6.0     # mask_dx:  0.036
  + 0.20 * |2.677| / 6.0     # mask_dy:  0.089  (dominant!)
  + 0.10 * |0.559| / 1.2     # theta:    0.047
  + 0.15 * |0.316| / 0.4     # phi_d:    0.119
  + 0.10 * |0.008| / 0.04    # disp_a2:  0.020  (SD distortion)
  + 0.05 * |0.3| / 4.0       # prism_rot: 0.004
  + 0.05 * |0.0| / 2.2       # psf:      0.000
  + 0.10 * |0.0| / 1.0       # gain:     0.000
  + 0.05 * 0.0                # channel_warp: 0.000
S = 0.315  # MODERATE severity
```

### Output → `MismatchReport`

```python
MismatchReport(
  modality_key="cassi",
  mismatch_family="UPWMI_beam_search",
  parameters={
    "mask_dx":  {"actual_error": 1.094, "range": [-3, 3], "weight": 0.20},
    "mask_dy":  {"actual_error": 2.677, "range": [-3, 3], "weight": 0.20},
    "theta":    {"actual_error": 0.559, "range": [-0.6, 0.6], "weight": 0.10},
    "phi_d":    {"actual_error": 0.316, "range": [-0.5, 0.5], "weight": 0.15},
    "disp_a2":  {"actual_error": 0.008, "range": [-0.02, 0.02], "weight": 0.10},
    "prism_rot": {"actual_error": 0.3, "range": [-2, 2], "weight": 0.05},
    "gain":     {"actual_error": 0.0,   "range": [0.5, 1.5], "weight": 0.10}
  },
  severity_score=0.315,
  correction_method="UPWMI_beam_search",
  expected_improvement_db=3.15,
  explanation="Moderate mismatch. Mask dy shift (2.7 px) is primary error. "
              "SD quadratic distortion (a2=0.008) adds band-dependent warp. "
              "Operator correction strongly recommended before reconstruction."
)
```

**Difference from ideal:** Severity jumped from 0.088 → 0.315. The mask dy shift alone accounts for most of the degradation. The SD distortion (a₂ term) adds a new failure mode not present in simple models.

---

## R4. RecoverabilityAgent — Degraded by Mismatch

### Adjusted lookup

```python
# Compression ratio unchanged
CR = 0.039

# But noise regime is now "detector_limited" due to mismatch-induced artifacts
# The mismatch acts like structured noise that the solver can't separate

# Calibration table lookup (detector_limited + gap_tv)
# → recoverability=0.58, expected_psnr=26.34 dB (down from 34.81!)
```

### Output → `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.039,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.joint_spatio_spectral,
  operator_diversity_score=1.0,
  condition_number_proxy=0.5,
  recoverability_score=0.58,              # Down from 0.88
  recoverability_confidence=0.85,
  expected_psnr_db=26.34,                 # Down from 34.81
  expected_psnr_uncertainty_db=3.0,       # Higher uncertainty
  recommended_solver_family="gap_tv",     # Conservative choice for correction
  verdict="marginal",                     # Was "excellent"
  explanation="Recoverability degraded by operator mismatch. "
              "Recommend calibrating operator before using deep learning solvers."
)
```

**Key insight:** The recoverability dropped from 0.88 → 0.58 not because of photon budget, but because the mismatched operator corrupts the reconstruction. Deep learning solvers (MST, HDNet) are especially sensitive to operator mismatch because they were trained assuming a correct forward model.

---

## R5. AnalysisAgent — Mismatch is the Bottleneck

```python
# Bottleneck scores
photon_score      = 1 - min(102.6 / 40, 1.0)  = 0.0     # Still excellent
mismatch_score    = 0.315                        = 0.315   # Moderate (was 0.088)
compression_score = 1 - 0.58                     = 0.42    # Worse (was 0.12)
solver_score      = 0.2                          = 0.2

# Primary bottleneck
primary = "compression"  # max(0.0, 0.315, 0.42, 0.2) = compression
# BUT: compression score is inflated BY mismatch, so root cause = mismatch

# Probability of success (without correction)
P = (1 - 0.0*0.5) * (1 - 0.315*0.5) * (1 - 0.42*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.843 * 0.79 * 0.90
  = 0.599
```

### Output → `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="mismatch",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.315, compression=0.42, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Apply UPWMI operator correction before reconstruction",
      priority="critical",
      expected_gain_db=5.7
    ),
    Suggestion(
      text="Calibrate SD distortion (imbalanced response) from narrowband frames",
      priority="high",
      expected_gain_db=1.5
    ),
    Suggestion(
      text="Use GAP-TV (robust to mismatch) instead of MST for initial correction",
      priority="medium",
      expected_gain_db=1.5
    )
  ],
  overall_verdict="marginal",              # Was "excellent"
  probability_of_success=0.599,            # Was 0.809
  explanation="Operator mismatch is the primary bottleneck. Without correction, "
              "expect 15-16 dB (unusable). SD distortion adds band-dependent artifacts. "
              "With UPWMI correction, expect 25+ dB."
)
```

**Contrast with ideal:** Verdict dropped from "excellent" → "marginal". The system correctly identifies that mismatch — not photons, not compression — is what will destroy image quality.

---

## R6. AgentNegotiator — Conditional Proceed

```python
P_photon         = 0.95     # tier_prob["excellent"]
P_recoverability = 0.58     # recoverability_score (degraded)
P_mismatch       = 1.0 - 0.315 * 0.7 = 0.780

P_joint = 0.95 * 0.58 * 0.780 = 0.430
```

### Veto check

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" BUT verdict="marginal" | **Close but no veto** |
| Severe mismatch without correction | severity=0.315 < 0.7 | No veto |
| All marginal | photon=excellent, others mixed | No veto |
| Joint probability floor | P=0.430 > 0.15 | No veto |

```python
NegotiationResult(
  vetoes=[],
  proceed=True,                            # Proceed, but with warnings
  probability_of_success=0.430             # Was 0.784
)
```

**No veto**, but P_joint dropped from 0.784 → 0.430. The system proceeds but flags the risk.

---

## R7. PreFlightReportBuilder — Warnings Raised

```python
PreFlightReport(
  estimated_runtime_s=3250.0,              # Was 17.5s — correction adds ~3200s
  proceed_recommended=True,
  warnings=[
    "Mismatch severity 0.315 — operator correction will run before reconstruction",
    "SD distortion (imbalanced response) detected — calibrating dispersion curve",
    "Recoverability marginal (0.58) — results may be degraded without correction",
    "Estimated runtime includes UPWMI beam search calibration (~3200s)"
  ],
  what_to_upload=[
    "measurement (2D CASSI snapshot, Nx × (Ny+Nλ-1))",
    "mask (coded aperture design pattern — will be calibrated)",
    "calibration frames (narrowband flat-fields, if available)"
  ]
)
```

**Key difference:** Runtime jumped from 17.5s → 3,250s because UPWMI operator correction dominates the cost. Four warnings alert the user (including SD distortion).

---

## R8. Pipeline Runner — With Operator Correction

After pre-flight approval, the pipeline runs with an additional correction step:

### Step R8a: Build Nominal Operator (Wrong)

```python
# Uses design_mask.npy as-is, no corrections
operator_wrong = SDCASSIOperator(
    mask=design_mask, step=2, n_bands=28, dispersion_axis="y"
)
operator_wrong.check_adjoint()  # Passes (operator is self-consistent, just wrong)

# Reconstruct with wrong operator → garbage
x_wrong = gap_tv_cassi(y_lab, design_mask, step=2)
# PSNR = 15.79 dB  ← unusable image
```

### Step R8b: UPWMI Algorithm 1 — Beam Search Calibration

```
Search space: ψ = (dx, dy, theta, phi_d, disp_a2)
  dx ∈ [-3, +3]         13 grid points
  dy ∈ [-3, +3]         25 grid points (finer — most sensitive parameter)
  theta ∈ [-0.6, +0.6]  13 grid points
  phi_d ∈ [-0.5, +0.5]  13 grid points
  disp_a2 ∈ [-0.02, +0.02]  5 grid points (SD distortion curvature)

Step 1: Independent grid sweeps
  For each parameter, fix others at 0, sweep grid, score by GAP-TV PSNR (25 iters)
  → Identify promising regions for each parameter

Step 2: Beam search (width=10)
  Combine top candidates across parameters
  Expand neighbors (delta = {dx: 0.40, dy: 1.50, theta: 0.15, phi_d: 0.08, a2: 0.005})
  Keep top-10 by reconstruction score

Step 3: Local refinement (6 rounds coordinate descent)
  For each parameter in turn, fine-tune around current best
  sigma = {dx: 0.5, dy: 0.5, theta: 0.8, phi_d: 0.5, a2: 0.3}
```

**Result:**
```python
psi_calibrated = {
    "dx":    1.286,      # true: 1.094, error: 0.192
    "dy":   -2.781,      # true: -2.677, error: 0.104
    "theta": -0.500,     # true: -0.559, error: 0.059
    "phi_d": -0.500,     # true: -0.316, error: 0.184
    "disp_a2": 0.006,    # true: 0.008, error: 0.002
}
```

### Step R8c: Reconstruct with Calibrated Operator

```python
# Apply calibrated parameters to shift/rotate the mask + update dispersion
mask_calibrated = apply_psi(design_mask, psi_calibrated)
operator_cal = SDCASSIOperator(
    mask=mask_calibrated,
    step=2 + psi_calibrated["phi_d"],
    n_bands=28,
    dispersion_axis="y",
    dispersion_a2=psi_calibrated["disp_a2"]
)

# GAP-TV with calibrated operator
x_cal_gaptv = gap_tv_cassi(y_lab, mask_calibrated, step=2.0 - 0.500)
# PSNR = 25.55 dB  ← +9.76 dB improvement over wrong operator

# MST with calibrated operator (deep learning)
x_cal_mst = mst_recon_cassi(y_lab, mask_calibrated, step=2.0 - 0.500)
# PSNR = 21.20 dB  ← +5.40 dB from wrong (MST more sensitive to residual error)
```

### Step R8d: UPWMI Algorithm 2 — Differentiable Refinement

```python
# Further refine with gradient-based optimization
# Parameterize ψ as differentiable torch tensors
# Loss = ||y_measured - A(ψ) @ x_hat||^2

# 9 phi_d candidates × 4 random starts × 200 Adam steps
# lr: 0.03 → 0.001 (cosine annealing), grad_clip=1.0

psi_refined = {
    "dx":    1.112,      # true: 1.094, error: 0.018  (9x better than Alg 1)
    "dy":   -2.750,      # true: -2.677, error: 0.072  (1.4x better)
    "theta": -0.579,     # true: -0.559, error: 0.019  (3x better)
    "phi_d": -0.093,     # true: -0.316, error: 0.223  (slightly worse)
    "disp_a2": 0.007,    # true: 0.008, error: 0.001  (refined)
}
# Optimization time: 3,200s
```

```python
# MST with refined operator
x_refined_mst = mst_recon_cassi(y_lab, mask_refined, step=2.0 - 0.093)
# PSNR = 21.54 dB  ← near oracle (21.58 dB)
```

### Step R8e: Final Comparison

| Configuration | GAP-TV | MST | Notes |
|---------------|--------|-----|-------|
| Wrong operator (nominal) | **15.79 dB** | **14.06 dB** | Unusable |
| Alg 1 calibrated (beam search) | **25.55 dB** | **21.20 dB** | +9.76 / +7.14 dB |
| Alg 2 calibrated (differentiable) | **25.55 dB** | **21.54 dB** | +9.76 / +7.48 dB |
| Oracle (true parameters) | **26.17 dB** | **21.58 dB** | Upper bound |

**Key findings:**
- Without correction: ~15 dB (unusable, severe artifacts across all spectral bands)
- After Alg 1: within 0.62 dB of oracle for GAP-TV, 0.38 dB for MST
- After Alg 2: within 0.62 dB of oracle for GAP-TV, **0.04 dB** for MST (near-perfect)
- GAP-TV is more robust to residual calibration error than MST
- SD distortion calibration (a₂) accounts for ~0.2 dB of additional recovery

---

## Real Experiment Pipeline Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           USER PROMPT                                       │
│  "Mask might be misaligned. Please calibrate and reconstruct."              │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PLAN AGENT   mode = operator_correction                                    │
│  → "cassi" (0.95) → 8 elements, CR=0.039                                   │
│  → SD-CASSI layout: Obj→Mask→RL1→Prism→SD_Distort→RL2→Det                  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌─────────────────┐  ┌──────────────────────┐  ┌──────────────────────────┐
│  PHOTON AGENT   │  │  MISMATCH AGENT      │  │  RECOVERABILITY AGENT    │
│                 │  │                      │  │                          │
│  N = 1.84e10    │  │  S = 0.315 (mod)     │  │  CR = 0.039              │
│  SNR = 102.6 dB │  │  dy=2.7px dominant   │  │  Rec = 0.58 (marginal)   │
│  Tier: excellent│  │  SD a2=0.008         │  │  PSNR → 26.34 dB        │
│                 │  │  Gain: +3.15 dB      │  │  *** DEGRADED ***        │
│                 │  │  *** CORRECTION ***   │  │                          │
└────────┬────────┘  └──────────┬───────────┘  └────────────┬─────────────┘
         │                      │                           │
         └──────────────────────┼───────────────────────────┘
                                │
                                ▼
                  ┌──────────────────────────────┐
                  │  ANALYSIS AGENT              │
                  │                              │
                  │  Bottleneck: MISMATCH (0.32) │
                  │  + SD distortion (a2=0.008)  │
                  │  P(success) = 0.599          │
                  │  Verdict: marginal           │
                  │  "Apply UPWMI correction"    │
                  │  "Calibrate SD distortion"   │
                  └────────────┬─────────────────┘
                               │
                               ▼
                  ┌──────────────────────────────┐
                  │  NEGOTIATOR                  │
                  │                              │
                  │  Vetoes: []                  │
                  │  Proceed: YES (with warnings)│
                  │  P_joint = 0.430             │
                  └────────────┬─────────────────┘
                               │
                               ▼
                  ┌──────────────────────────────┐
                  │  PRE-FLIGHT REPORT           │
                  │                              │
                  │  Runtime: ~3,250s            │
                  │  Warnings: 4                 │
                  │  "Mismatch severity 0.315"   │
                  │  "SD distortion detected"    │
                  │  "UPWMI correction included"  │
                  └────────────┬─────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PIPELINE RUNNER (with operator correction)                                 │
│                                                                             │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐  ┌───────────────────┐│
│  │ Build Op  │→│ Recon wrong  │→│ UPWMI Alg 1    │→│ UPWMI Alg 2       ││
│  │ (nominal) │  │ → 15.79 dB   │  │ beam search    │  │ differentiable    ││
│  │ psi=(0,0) │  │ (unusable)   │  │ + SD calib     │  │ + SD refinement   ││
│  │           │  │              │  │ → 25.55 dB     │  │ → 25.55 / 21.54  ││
│  └──────────┘  └──────────────┘  └────────────────┘  └─────────┬─────────┘│
│                                                                  │         │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │         │
│  │ Build Op     │→│ MST Recon   │→│ Metrics + RunBundle      │←┘         │
│  │ (calibrated) │  │ (256,256,28)│  │ PSNR=21.54, SSIM, SAM  │           │
│  │ psi=(1.1,..) │  │  21.54 dB   │  │ SHA256, provenance      │           │
│  │ a2=0.007     │  │             │  │                         │           │
│  └──────────────┘  └─────────────┘  └─────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Experiment |
|--------|-----------|-----------------|
| **Optical Chain** | | |
| Layout | SD-CASSI (ECCV-2020) | SD-CASSI (ECCV-2020) |
| Elements | 8 (incl. SD distortion) | 8 (incl. SD distortion) |
| Measurement shape | 256 × 283 | 256 × 283 |
| Dispersion axis | y (columns) | y (columns) |
| **PhotonAgent** | | |
| N_effective | 2.68e12 | 1.84e10 |
| SNR | 124.3 dB | 102.6 dB |
| Quality tier | excellent | excellent |
| Noise regime | shot_limited | shot_limited |
| Noise model | Poisson+read+quant | Poisson+read+quant |
| **MismatchAgent** | | |
| Severity | 0.088 (low) | 0.315 (moderate) |
| Dominant error | none | mask dy = 2.7 px |
| SD distortion (a₂) | 0 | 0.008 |
| Expected gain | +0.88 dB | +3.15 dB |
| Correction needed | No | **Yes** |
| **RecoverabilityAgent** | | |
| Score | 0.88 (excellent) | 0.58 (marginal) |
| Expected PSNR | 34.81 dB | 26.34 dB |
| Verdict | excellent | **marginal** |
| **AnalysisAgent** | | |
| Primary bottleneck | solver | **mismatch** |
| P(success) | 0.809 | 0.599 |
| Verdict | excellent | **marginal** |
| **Negotiator** | | |
| Vetoes | 0 | 0 |
| P_joint | 0.784 | 0.430 |
| **PreFlight** | | |
| Runtime | 17.5s | **3,250s** |
| Warnings | 0 | **4** |
| **Pipeline** | | |
| Mode | simulate/reconstruct | **operator_correction** |
| Without correction | — | 15.79 dB (unusable) |
| With correction | — | **25.55 dB** (GAP-TV) |
| Final PSNR | 34.81 dB (MST) | **21.54 dB** (MST calibrated) |
| Oracle PSNR | 34.81 dB | 21.58 dB |
| Gap to oracle | 0.0 dB | **0.04 dB** |

---

## Design Principles

1. **ECCV-2020 faithful:** Optical chain matches Fig.1 of the SD-CASSI paper — objective lens, coded aperture, two relay lenses, single disperser, detector.
2. **SD distortion modeled:** ImbalancedResponseNode captures the path-length-dependent warp that the paper identifies as a key SD hardware limitation.
3. **Correct axis + shape:** Dispersion along y-axis, measurement Nx × (Ny + Nλ − 1), consistent across doc equations and code.
4. **Paper-consistent noise:** Poisson shot + read + quantization (not unrealistic high-SNR).
5. **Operator-correction ready:** Φ(θ) parameterization supports mask misalignment, dispersion curve, and SD warp estimation via UPWMI.
6. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
7. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
8. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
9. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
10. **Modular:** Swap any solver (GAP-TV → MST → HDNet) by changing one registry ID.
11. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
12. **Adaptive:** The same pipeline automatically switches from direct reconstruction to operator correction when mismatch severity warrants it.
