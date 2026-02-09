# CASSI Working Process

## End-to-End Pipeline for Coded Aperture Snapshot Spectral Imaging

This document traces a complete CASSI experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

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
#   display_name="Coded Aperture Snapshot Spectral Imaging (CASSI)",
#   signal_dims={"x": [256, 256, 28], "y": [283, 256]},
#   forward_model_type=ForwardModelType.linear_operator,
#   elements=[...6 elements...],
#   default_solver="mst"
# )
```

**CASSI Element Chain (6 elements):**

```
Source (halogen) ──► Coded Aperture Mask ──► Relay Lens ──► Dispersive Prism ──► Imaging Lens ──► CCD Detector
  throughput=1.0      throughput=0.50       throughput=0.90   throughput=0.88     throughput=0.90   throughput=0.75
  noise: none         noise: fixed_pattern   noise: aberration  noise: alignment    noise: aberration  noise: shot+read+quant
                            + alignment
```

**Cumulative throughput:** `0.50 × 0.90 × 0.88 × 0.90 × 0.75 = 0.267`

---

## 3. PhotonAgent — SNR & Noise Analysis

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

# 4. Apply cumulative throughput
N_effective = N_raw * 0.267 ≈ 2.94e12 photons/pixel

# 5. Noise variances
shot_var   = N_effective                    # Poisson
read_var   = read_noise^2 = 25.0           # Gaussian (5 e-)
dark_var   = dark_current * exposure = 0    # Negligible
total_var  = shot_var + read_var + dark_var

# 6. SNR
SNR = N_effective / sqrt(total_var) ≈ sqrt(N_effective) ≈ 1.71e6
SNR_db = 20 * log10(SNR) ≈ 124.7 dB
```

### Output → `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=2.94e12,
  snr_db=124.7,
  noise_regime=NoiseRegime.shot_limited,    # shot_var/total_var > 0.9
  shot_noise_sigma=1.71e6,
  read_noise_sigma=5.0,
  total_noise_sigma=1.71e6,
  feasible=True,
  quality_tier="excellent",                 # SNR > 30 dB
  throughput_chain=[
    {"Broadband Source": 1.0},
    {"Coded Aperture Mask": 0.50},
    {"Relay Lens": 0.90},
    {"Dispersive Prism": 0.88},
    {"Imaging Lens": 0.90},
    {"CCD Detector": 0.75}
  ],
  noise_model="poisson",
  explanation="Shot-noise-limited regime. Excellent SNR for reconstruction."
)
```

---

## 4. MismatchAgent — Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"cassi"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  cassi:
    parameters:
      mask_dx:     {range: [-3, 3], typical_error: 0.5, weight: 0.30}
      mask_dy:     {range: [-3, 3], typical_error: 0.5, weight: 0.30}
      dispersion_step: {range: [0.8, 1.2], typical_error: 0.03, weight: 0.20}
      psf_sigma:   {range: [0.3, 2.5], typical_error: 0.3, weight: 0.10}
      gain:        {range: [0.5, 1.5], typical_error: 0.1, weight: 0.10}
    correction_method: "UPWMI_beam_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.30 * |0.5| / 6.0    # mask_dx: 0.025
  + 0.30 * |0.5| / 6.0    # mask_dy: 0.025
  + 0.20 * |0.03| / 0.4   # dispersion: 0.015
  + 0.10 * |0.3| / 2.2    # psf: 0.014
  + 0.10 * |0.1| / 1.0    # gain: 0.010
S = 0.089  # Low severity (typical lab conditions)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 0.89 dB
```

### Output → `MismatchReport`

```python
MismatchReport(
  modality_key="cassi",
  mismatch_family="UPWMI_beam_search",
  parameters={
    "mask_dx":  {"typical_error": 0.5, "range": [-3, 3], "weight": 0.30},
    "mask_dy":  {"typical_error": 0.5, "range": [-3, 3], "weight": 0.30},
    "dispersion_step": {"typical_error": 0.03, "range": [0.8, 1.2], "weight": 0.20},
    "psf_sigma": {"typical_error": 0.3, "range": [0.3, 2.5], "weight": 0.10},
    "gain":     {"typical_error": 0.1, "range": [0.5, 1.5], "weight": 0.10}
  },
  severity_score=0.089,
  correction_method="UPWMI_beam_search",
  expected_improvement_db=0.89,
  explanation="Low mismatch severity under typical conditions."
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
# 1. Compression ratio
CR = prod(y_shape) / prod(x_shape) = (283 * 256) / (256 * 256 * 28) = 0.036

# 2. Operator diversity (mask density heuristic)
density = 0.5  # binary random mask
diversity = 4 * density * (1 - density) = 1.0  # Maximum diversity

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.5

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="mst", cr=0.036 (exact match)
#    → recoverability=0.88, expected_psnr=34.81 dB, confidence=1.0

# 5. Best solver selection
#    hdnet: 34.97 dB > mst: 34.81 dB > gap_tv: 30.52 dB
#    → recommended: "hdnet" (or "mst" as default)
```

### Output → `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.036,
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

## 6. AnalysisAgent — Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(124.7 / 40, 1.0)  = 0.0    # Excellent SNR
mismatch_score    = 0.089                        = 0.089  # Low mismatch
compression_score = 1 - 0.88                     = 0.12   # Good recoverability
solver_score      = 0.2                          = 0.2    # Default placeholder

# Primary bottleneck
primary = "solver"  # max(0.0, 0.089, 0.12, 0.2) = solver

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.089*0.5) * (1 - 0.12*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.956 * 0.94 * 0.90
  = 0.808
```

### Output → `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="solver",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.089, compression=0.12, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Consider using HDNet for +0.16 dB over MST",
      priority="medium",
      expected_gain_db=0.16
    )
  ],
  overall_verdict="excellent",      # P ≥ 0.80
  probability_of_success=0.808,
  explanation="System is well-configured. Solver choice is the primary bottleneck."
)
```

---

## 7. AgentNegotiator — Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="excellent" | No veto |
| Severe mismatch without correction | severity=0.089 < 0.7 | No veto |
| All marginal | All excellent/sufficient | No veto |
| Joint probability floor | P=0.808 > 0.15 | No veto |

### Joint Probability

```python
P_photon       = 0.95   # tier_prob["excellent"]
P_recoverability = 0.88  # recoverability_score
P_mismatch     = 1.0 - 0.089 * 0.7 = 0.938

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

## 8. PreFlightReportBuilder — Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 256 * 256 * 28 = 1,835,008
dim_factor   = total_pixels / (256 * 256) = 28.0
solver_complexity = 2.5  # MST (transformer-based)
cr_factor    = max(0.036, 1.0) / 8.0 = 0.125

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

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# CASSI forward model: y(x,y) = Σ_l M(x,y) · X(x, y - s(l), l)
#
# Parameters:
#   mask:   (H, W) binary coded aperture     [loaded from mask.npy]
#   step:   dispersion_step_px = 2           [pixels/band]
#   n_bands: 28
#
# Input:  x = (256, 256, 28) hyperspectral cube
# Output: y = (283, 256) compressed snapshot
#         where 283 = 256 + (28-1)*2

class CASSIOperator(PhysicsOperator):
    def forward(self, x):
        """y(r,c) = Σ_l mask(r,c) * cube(r, c - l*step, l)"""
        y = np.zeros((H + (L-1)*step, W))
        for l in range(L):
            shifted_mask = shift(mask, l * step, axis=0)
            y[l*step : l*step+H, :] += shifted_mask * x[:, :, l]
        return y

    def adjoint(self, y):
        """x_hat(r,c,l) = mask(r,c) * y(r + l*step, c)"""
        x = np.zeros((H, W, L))
        for l in range(L):
            x[:, :, l] = mask * y[l*step : l*step+H, :]
        return x

    def check_adjoint(self):
        """Verify <Ax, y> ≈ <x, A*y> for random x, y"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-10)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided measurement.npy:
y = np.load("measurement.npy")    # (283, 256)

# If simulating:
x_true = load_ground_truth()       # (256, 256, 28) from KAIST dataset
y = operator.forward(x_true)       # (283, 256)
y += np.random.poisson(y)          # Shot noise (Poisson)
```

### Step 9c: Reconstruction with MST

```python
from pwm_core.recon.mst import mst_recon_cassi

x_hat = mst_recon_cassi(
    y=y,                    # (283, 256) compressed measurement
    mask=mask,              # (256, 256) coded aperture
    dispersion_step=2,      # pixels per spectral shift
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

### Step 9d: Metrics

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

### Step 9e: RunBundle Output

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
│   ├── y.npy              # Measurement (283, 256) + SHA256 hash
│   ├── x_hat.npy          # Reconstruction (256, 256, 28) + SHA256 hash
│   └── x_true.npy         # Ground truth (if available) + SHA256 hash
├── metrics.json           # PSNR, SSIM, SAM per band + average
├── operator.json          # Operator parameters (mask hash, step, n_bands)
└── provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an **idealized** CASSI pipeline with perfect lab conditions (SNR 124.7 dB, mismatch severity 0.089). In practice, real CASSI systems have significant operator mismatch from assembly tolerances, limited photon budgets from short exposures, and detector noise.

This section traces the **same pipeline** with realistic parameters drawn from our actual benchmark experiments.

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

# Apply cumulative throughput (0.267)
N_effective = 7.56e10 * 0.267 ≈ 2.02e10 photons/pixel

# Noise variances
shot_var   = 2.02e10                        # Poisson
read_var   = 12.0^2 = 144.0                # Significant read noise
dark_var   = 0.5 * 0.01 = 0.005            # Negligible
total_var  = 2.02e10 + 144.0 + 0.005

# SNR
SNR = 2.02e10 / sqrt(2.02e10 + 144) ≈ sqrt(2.02e10) ≈ 1.42e5
SNR_db = 20 * log10(1.42e5) ≈ 103.0 dB
```

### Output → `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=2.02e10,
  snr_db=103.0,
  noise_regime=NoiseRegime.shot_limited,    # Still shot-dominated
  shot_noise_sigma=1.42e5,
  read_noise_sigma=12.0,
  total_noise_sigma=1.42e5,
  feasible=True,
  quality_tier="excellent",                 # 103 dB >> 30 dB threshold
  noise_model="poisson",
  explanation="Shot-limited despite shorter exposure. Feasible for reconstruction."
)
```

**Verdict:** Even with 100x fewer photons, CASSI is still comfortably shot-limited. This is typical for spectral imaging — the bottleneck is almost never photon count.

---

## R3. MismatchAgent — Real Assembly Tolerances

### Real mismatch: mask shifted + rotated + dispersion off

In a real lab CASSI prototype, the coded aperture mask is manually positioned. After disassembly/reassembly, typical errors are:

```python
# True (unknown) operator parameters
psi_true = {
    "dx":    +1.094,    # mask shifted right by ~1.1 pixels
    "dy":    -2.677,    # mask shifted down by ~2.7 pixels (largest error)
    "theta": -0.559,    # mask rotated by ~0.56 rad (32 degrees!)
    "phi_d": -0.316,    # dispersion step off by -0.32 px/band
}

# What the system thinks (nominal = zero error)
psi_nominal = {"dx": 0, "dy": 0, "theta": 0, "phi_d": 0}
```

### Severity computation

```python
# Actual errors (not typical — measured from benchmark ground truth)
S = 0.30 * |1.094| / 6.0     # mask_dx:  0.0547
  + 0.30 * |2.677| / 6.0     # mask_dy:  0.1339  (dominant!)
  + 0.20 * |0.559| / 1.2     # theta:    0.0932  (dispersion range proxy)
  + 0.10 * |0.316| / 0.4     # phi_d:    0.0790
  + 0.10 * |0.0|   / 1.0     # gain:     0.0000
S = 0.361  # MODERATE severity

# Expected improvement from correction
improvement_db = clip(10 * 0.361, 0, 20) = 3.61 dB
```

### Output → `MismatchReport`

```python
MismatchReport(
  modality_key="cassi",
  mismatch_family="UPWMI_beam_search",
  parameters={
    "mask_dx":  {"actual_error": 1.094, "range": [-3, 3], "weight": 0.30},
    "mask_dy":  {"actual_error": 2.677, "range": [-3, 3], "weight": 0.30},
    "theta":    {"actual_error": 0.559, "range": [-0.6, 0.6], "weight": 0.20},
    "phi_d":    {"actual_error": 0.316, "range": [-0.5, 0.5], "weight": 0.10},
    "gain":     {"actual_error": 0.0,   "range": [0.5, 1.5], "weight": 0.10}
  },
  severity_score=0.361,
  correction_method="UPWMI_beam_search",
  expected_improvement_db=3.61,
  explanation="Moderate mismatch. Mask dy shift (2.7 px) is primary error. "
              "Operator correction strongly recommended before reconstruction."
)
```

**Difference from ideal:** Severity jumped from 0.089 → 0.361. The mask dy shift alone accounts for most of the degradation. Without correction, reconstruction quality will be severely impacted.

---

## R4. RecoverabilityAgent — Degraded by Mismatch

### Adjusted lookup

```python
# Compression ratio unchanged
CR = 0.036

# But noise regime is now "detector_limited" due to mismatch-induced artifacts
# The mismatch acts like structured noise that the solver can't separate

# Calibration table lookup (detector_limited + gap_tv)
# → recoverability=0.58, expected_psnr=26.34 dB (down from 34.81!)
```

### Output → `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.036,
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
photon_score      = 1 - min(103.0 / 40, 1.0)  = 0.0     # Still excellent
mismatch_score    = 0.361                        = 0.361   # Moderate (was 0.089)
compression_score = 1 - 0.58                     = 0.42    # Worse (was 0.12)
solver_score      = 0.2                          = 0.2

# Primary bottleneck
primary = "compression"  # max(0.0, 0.361, 0.42, 0.2) = compression
# BUT: compression score is inflated BY mismatch, so root cause = mismatch

# Probability of success (without correction)
P = (1 - 0.0*0.5) * (1 - 0.361*0.5) * (1 - 0.42*0.5) * (1 - 0.2*0.5)
  = 1.0 * 0.820 * 0.79 * 0.90
  = 0.583
```

### Output → `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="mismatch",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.361, compression=0.42, solver=0.2
  ),
  suggestions=[
    Suggestion(
      text="Apply UPWMI operator correction before reconstruction",
      priority="critical",
      expected_gain_db=5.7
    ),
    Suggestion(
      text="Recalibrate the forward operator from measured data",
      priority="high",
      expected_gain_db=3.0
    ),
    Suggestion(
      text="Use GAP-TV (robust to mismatch) instead of MST for initial correction",
      priority="medium",
      expected_gain_db=1.5
    )
  ],
  overall_verdict="marginal",              # Was "excellent"
  probability_of_success=0.583,            # Was 0.808
  explanation="Operator mismatch is the primary bottleneck. Without correction, "
              "expect 15-16 dB (unusable). With UPWMI correction, expect 25+ dB."
)
```

**Contrast with ideal:** Verdict dropped from "excellent" → "marginal". The system correctly identifies that mismatch — not photons, not compression — is what will destroy image quality.

---

## R6. AgentNegotiator — Conditional Proceed

```python
P_photon         = 0.95     # tier_prob["excellent"]
P_recoverability = 0.58     # recoverability_score (degraded)
P_mismatch       = 1.0 - 0.361 * 0.7 = 0.747

P_joint = 0.95 * 0.58 * 0.747 = 0.411
```

### Veto check

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" BUT verdict="marginal" | **Close but no veto** |
| Severe mismatch without correction | severity=0.361 < 0.7 | No veto |
| All marginal | photon=excellent, others mixed | No veto |
| Joint probability floor | P=0.411 > 0.15 | No veto |

```python
NegotiationResult(
  vetoes=[],
  proceed=True,                            # Proceed, but with warnings
  probability_of_success=0.411             # Was 0.784
)
```

**No veto**, but P_joint dropped from 0.784 → 0.411. The system proceeds but flags the risk.

---

## R7. PreFlightReportBuilder — Warnings Raised

```python
PreFlightReport(
  estimated_runtime_s=3250.0,              # Was 17.5s — correction adds ~3200s
  proceed_recommended=True,
  warnings=[
    "Mismatch severity 0.361 — operator correction will run before reconstruction",
    "Recoverability marginal (0.58) — results may be degraded without correction",
    "Estimated runtime includes UPWMI beam search calibration (~3200s)"
  ],
  what_to_upload=[
    "measurement (2D CASSI snapshot)",
    "mask (coded aperture design pattern — will be calibrated)"
  ]
)
```

**Key difference:** Runtime jumped from 17.5s → 3,250s because UPWMI operator correction dominates the cost. Three warnings alert the user.

---

## R8. Pipeline Runner — With Operator Correction

After pre-flight approval, the pipeline runs with an additional correction step:

### Step R8a: Build Nominal Operator (Wrong)

```python
# Uses design_mask.npy as-is, no corrections
operator_wrong = CASSIOperator(mask=design_mask, step=2, n_bands=28)
operator_wrong.check_adjoint()  # Passes (operator is self-consistent, just wrong)

# Reconstruct with wrong operator → garbage
x_wrong = gap_tv_cassi(y_lab, design_mask, step=2)
# PSNR = 15.79 dB  ← unusable image
```

### Step R8b: UPWMI Algorithm 1 — Beam Search Calibration

```
Search space: ψ = (dx, dy, theta, phi_d)
  dx ∈ [-3, +3]     13 grid points
  dy ∈ [-3, +3]     25 grid points (finer — most sensitive parameter)
  theta ∈ [-0.6, +0.6]  13 grid points
  phi_d ∈ [-0.5, +0.5]  13 grid points

Step 1: Independent grid sweeps
  For each parameter, fix others at 0, sweep grid, score by GAP-TV PSNR (25 iters)
  → Identify promising regions for each parameter

Step 2: Beam search (width=10)
  Combine top candidates across parameters
  Expand neighbors (delta = {dx: 0.40, dy: 1.50, theta: 0.15, phi_d: 0.08})
  Keep top-10 by reconstruction score

Step 3: Local refinement (6 rounds coordinate descent)
  For each parameter in turn, fine-tune around current best
  sigma = {dx: 0.5, dy: 0.5, theta: 0.8, phi_d: 0.5}
```

**Result:**
```python
psi_calibrated = {
    "dx":    1.286,      # true: 1.094, error: 0.192
    "dy":   -2.781,      # true: -2.677, error: 0.104
    "theta": -0.500,     # true: -0.559, error: 0.059
    "phi_d": -0.500,     # true: -0.316, error: 0.184
}
```

### Step R8c: Reconstruct with Calibrated Operator

```python
# Apply calibrated parameters to shift/rotate the mask
mask_calibrated = apply_psi(design_mask, psi_calibrated)
operator_cal = CASSIOperator(mask=mask_calibrated, step=2+phi_d, n_bands=28)

# GAP-TV with calibrated operator
x_cal_gaptv = gap_tv_cassi(y_lab, mask_calibrated, step=2.0-0.500)
# PSNR = 25.55 dB  ← +9.76 dB improvement over wrong operator

# MST with calibrated operator (deep learning)
x_cal_mst = mst_recon_cassi(y_lab, mask_calibrated, step=2.0-0.500)
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
}
# Optimization time: 3,200s
```

```python
# MST with refined operator
x_refined_mst = mst_recon_cassi(y_lab, mask_refined, step=2.0-0.093)
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
│  → "cassi" (0.95) → 6 elements, CR=0.036                                   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌─────────────────┐  ┌──────────────────────┐  ┌──────────────────────────┐
│  PHOTON AGENT   │  │  MISMATCH AGENT      │  │  RECOVERABILITY AGENT    │
│                 │  │                      │  │                          │
│  N = 2.02e10    │  │  S = 0.361 (mod)     │  │  CR = 0.036              │
│  SNR = 103.0 dB │  │  dy=2.7px dominant   │  │  Rec = 0.58 (marginal)   │
│  Tier: excellent│  │  Gain: +3.61 dB      │  │  PSNR → 26.34 dB        │
│                 │  │  *** CORRECTION ***   │  │  *** DEGRADED ***        │
└────────┬────────┘  └──────────┬───────────┘  └────────────┬─────────────┘
         │                      │                           │
         └──────────────────────┼───────────────────────────┘
                                │
                                ▼
                  ┌──────────────────────────────┐
                  │  ANALYSIS AGENT              │
                  │                              │
                  │  Bottleneck: MISMATCH (0.36) │
                  │  P(success) = 0.583          │
                  │  Verdict: marginal           │
                  │  "Apply UPWMI correction"    │
                  └────────────┬─────────────────┘
                               │
                               ▼
                  ┌──────────────────────────────┐
                  │  NEGOTIATOR                  │
                  │                              │
                  │  Vetoes: []                  │
                  │  Proceed: YES (with warnings)│
                  │  P_joint = 0.411             │
                  └────────────┬─────────────────┘
                               │
                               ▼
                  ┌──────────────────────────────┐
                  │  PRE-FLIGHT REPORT           │
                  │                              │
                  │  Runtime: ~3,250s            │
                  │  Warnings: 3                 │
                  │  "Mismatch severity 0.361"   │
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
│  │ psi=(0,0) │  │ (unusable)   │  │ → 25.55 dB     │  │ → 25.55 / 21.54  ││
│  └──────────┘  └──────────────┘  └────────────────┘  └─────────┬─────────┘│
│                                                                  │         │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │         │
│  │ Build Op     │→│ MST Recon   │→│ Metrics + RunBundle      │←┘         │
│  │ (calibrated) │  │ (256,256,28)│  │ PSNR=21.54, SSIM, SAM  │           │
│  │ psi=(1.1,..) │  │  21.54 dB   │  │ SHA256, provenance      │           │
│  └──────────────┘  └─────────────┘  └─────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Experiment |
|--------|-----------|-----------------|
| **Photon Agent** | | |
| N_effective | 2.94e12 | 2.02e10 |
| SNR | 124.7 dB | 103.0 dB |
| Quality tier | excellent | excellent |
| Noise regime | shot_limited | shot_limited |
| **Mismatch Agent** | | |
| Severity | 0.089 (low) | 0.361 (moderate) |
| Dominant error | none | mask dy = 2.7 px |
| Expected gain | +0.89 dB | +3.61 dB |
| Correction needed | No | **Yes** |
| **Recoverability Agent** | | |
| Score | 0.88 (excellent) | 0.58 (marginal) |
| Expected PSNR | 34.81 dB | 26.34 dB |
| Verdict | excellent | **marginal** |
| **Analysis Agent** | | |
| Primary bottleneck | solver | **mismatch** |
| P(success) | 0.808 | 0.583 |
| Verdict | excellent | **marginal** |
| **Negotiator** | | |
| Vetoes | 0 | 0 |
| P_joint | 0.784 | 0.411 |
| **PreFlight** | | |
| Runtime | 17.5s | **3,250s** |
| Warnings | 0 | **3** |
| **Pipeline** | | |
| Mode | simulate/reconstruct | **operator_correction** |
| Without correction | — | 15.79 dB (unusable) |
| With correction | — | **25.55 dB** (GAP-TV) |
| Final PSNR | 34.81 dB (MST) | **21.54 dB** (MST calibrated) |
| Oracle PSNR | 34.81 dB | 21.58 dB |
| Gap to oracle | 0.0 dB | **0.04 dB** |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (GAP-TV → MST → HDNet) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Adaptive:** The same pipeline automatically switches from direct reconstruction to operator correction when mismatch severity warrants it.
