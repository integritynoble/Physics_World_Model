# Ptychography Working Process

## End-to-End Pipeline for Ptychographic Imaging

This document traces a complete ptychography experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct the amplitude and phase of a sample from overlapping
 diffraction patterns. Patterns: diffraction.npy, Positions: positions.npy,
 wavelength=0.155 nm (8 keV X-ray), 16 scan positions, 60% overlap."
```

---

## 2. PlanAgent --- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "diffraction.npy" detected
#   operator_type=OperatorType.nonlinear_operator,
#   files=["diffraction.npy", "positions.npy"],
#   params={"wavelength_nm": 0.155, "n_positions": 16, "overlap": 0.6}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> ptychography entry
ptychography:
  keywords: [ptychography, phase_retrieval, scanning, coherent, CDI, ePIE]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="ptychography",
#   confidence=0.93,
#   reasoning="Matched keywords: ptychography, diffraction, phase_retrieval"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the ptychography registry entry:

```python
system = plan_agent.build_imaging_system("ptychography")
# ImagingSystem(
#   modality_key="ptychography",
#   display_name="Ptychographic Imaging",
#   signal_dims={"x": [256, 256], "y": [16, 128, 128]},
#   forward_model_type=ForwardModelType.nonlinear_operator,
#   elements=[...5 elements...],
#   default_solver="epie"
# )
```

**Ptychography Element Chain (5 elements):**

```
Coherent X-ray Source ---> Zone Plate (Probe Former) ---> Scanning Stage ---> Sample ---> Pixel Array Detector
  throughput=1.0            throughput=0.15               throughput=1.0      throughput=0.80  throughput=0.90
  noise: none               noise: aberration              noise: alignment    noise: none      noise: shot+read
  E=8 keV                   outermost_zone=50 nm          n_pos=16            thin_sample      photon_counting
  lambda=0.155 nm           focal_length=25 mm            overlap=60%         max_phase=1.5    128x128 px
```

**Cumulative throughput:** `1.0 x 0.15 x 1.0 x 0.80 x 0.90 = 0.108`

**Forward model:** `I_j = |FFT(P(r) * O(r - r_j))|^2, j = 1..N_pos`

This is a **nonlinear** operator because the measurement is the squared modulus of the Fourier transform, destroying phase information.

---

## 3. PhotonAgent --- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  ptychography:
    model_id: "generic_detector"
    parameters:
      source_photons: 1.0e+08
      qe: 0.90
      exposure_s: 0.1
  ```

### Computation

```python
# 1. Source photon count (synchrotron beamline)
N_source = 1.0e8 photons/exposure

# 2. Apply cumulative throughput
N_effective = N_source * 0.108 = 1.08e7 photons/position

# 3. Photons per detector pixel
# 128x128 detector = 16,384 pixels
# Diffraction pattern distributes photons across detector
N_per_pixel = N_effective / (128 * 128) = 659 photons/pixel

# 4. Noise variances (photon-counting detector)
shot_var   = N_per_pixel = 659         # Poisson
read_var   = 0.0                        # Photon-counting: zero read noise
dark_var   = 0.0                        # Negligible at synchrotron
total_var  = shot_var + read_var

# 5. SNR per pixel
SNR = N_per_pixel / sqrt(total_var) = sqrt(659) = 25.7
SNR_db = 20 * log10(25.7) = 28.2 dB

# 6. Effective SNR with overlap redundancy
# 16 positions with 60% overlap -> each sample point illuminated ~4 times
# Redundancy gain: sqrt(4) = 2x
SNR_eff = SNR * 2.0 = 51.4
SNR_eff_db = 20 * log10(51.4) = 34.2 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=659,
  snr_db=34.2,
  noise_regime=NoiseRegime.shot_limited,   # shot_var/total_var > 0.9
  shot_noise_sigma=25.7,
  read_noise_sigma=0.0,
  total_noise_sigma=25.7,
  feasible=True,
  quality_tier="good",                      # 30 < SNR < 40 dB
  throughput_chain=[
    {"Coherent X-ray Source": 1.0},
    {"Zone Plate": 0.15},
    {"Scanning Stage": 1.0},
    {"Sample": 0.80},
    {"Pixel Array Detector": 0.90}
  ],
  noise_model="poisson",
  explanation="Shot-limited regime. Zone plate throughput (15%) is the main photon "
              "bottleneck. Scanning overlap provides 2x redundancy gain. Adequate for ePIE."
)
```

---

## 4. MismatchAgent --- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"ptychography"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  ptychography:
    parameters:
      probe_position_error_x:
        range: [-5.0, 5.0]
        typical_error: 1.0
        unit: "pixels"
        description: "Horizontal scan position error from stage hysteresis"
      probe_position_error_y:
        range: [-5.0, 5.0]
        typical_error: 1.0
        unit: "pixels"
        description: "Vertical scan position error from stage hysteresis"
      probe_defocus:
        range: [-2.0, 2.0]
        typical_error: 0.5
        unit: "um"
        description: "Probe defocus from incorrect sample-condenser distance"
    severity_weights:
      probe_position_error_x: 0.35
      probe_position_error_y: 0.35
      probe_defocus: 0.30
    correction_method: "gradient_descent"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.35 * |1.0| / 10.0    # position_x: 0.035
  + 0.35 * |1.0| / 10.0    # position_y: 0.035
  + 0.30 * |0.5| / 4.0     # defocus:    0.0375
S = 0.108  # Low severity (typical beamline conditions)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.08 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="ptychography",
  mismatch_family="gradient_descent",
  parameters={
    "probe_position_error_x": {"typical_error": 1.0, "range": [-5, 5], "weight": 0.35},
    "probe_position_error_y": {"typical_error": 1.0, "range": [-5, 5], "weight": 0.35},
    "probe_defocus":          {"typical_error": 0.5, "range": [-2, 2], "weight": 0.30}
  },
  severity_score=0.108,
  correction_method="gradient_descent",
  expected_improvement_db=1.08,
  explanation="Low mismatch severity under typical conditions. ePIE jointly "
              "refines probe and positions, partially self-correcting."
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
  ptychography:
    signal_prior_class: "low_rank"
    entries:
      - {cr: 0.50, noise: "shot_limited", solver: "epie",
         recoverability: 0.96, expected_psnr_db: 59.2,
         provenance: {dataset_id: "ptycho_siemens_star_2023", ...}}
      - {cr: 0.50, noise: "photon_starved", solver: "epie",
         recoverability: 0.78, expected_psnr_db: 42.5, ...}
      - {cr: 0.50, noise: "shot_limited", solver: "ptychonn",
         recoverability: 0.94, expected_psnr_db: 55.8, ...}
  ```

### Computation

```python
# 1. Compression ratio
# N_positions * detector_pixels vs object_pixels
# 16 * 128 * 128 = 262,144 measurements vs 256 * 256 = 65,536 object pixels
# But measurements are intensity (real) while object is complex (2 DOF/pixel)
CR = (16 * 128 * 128) / (2 * 256 * 256) = 262144 / 131072 = 2.0
# Oversampled! But phase retrieval needs >4x oversampling for reliable convergence
# Effective CR for calibration table: 0.50 (matching N_pos/N_max_pos)

# 2. Operator diversity (overlap-driven)
# 60% overlap -> highly redundant, excellent diversity
diversity = 4 * 0.6 * (1 - 0.6) = 0.96  # Near-maximum

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.510

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="epie", cr=0.50
#    -> recoverability=0.96, expected_psnr=59.2 dB, confidence=1.0

# 5. Best solver selection
#    epie: 59.2 dB > ptychonn: 55.8 dB
#    -> recommended: "epie" (also recovers probe function)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.50,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.low_rank,
  operator_diversity_score=0.96,
  condition_number_proxy=0.510,
  recoverability_score=0.96,
  recoverability_confidence=1.0,
  expected_psnr_db=59.2,
  expected_psnr_uncertainty_db=2.0,
  recommended_solver_family="epie",
  verdict="excellent",               # score >= 0.85
  calibration_table_entry={...},
  explanation="Excellent recoverability. 60% overlap provides strong redundancy. "
              "ePIE expected 59.2 dB on Siemens star benchmark (200 iterations)."
)
```

---

## 6. AnalysisAgent --- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(34.2 / 40, 1.0)   = 0.145   # Good but not excellent
mismatch_score    = 0.108                        = 0.108   # Low
compression_score = 1 - 0.96                     = 0.04    # Excellent recoverability
solver_score      = 0.1                          = 0.1     # ePIE well-characterized

# Primary bottleneck
primary = "photon"  # max(0.145, 0.108, 0.04, 0.1) = photon

# Probability of success
P = (1 - 0.145*0.5) * (1 - 0.108*0.5) * (1 - 0.04*0.5) * (1 - 0.1*0.5)
  = 0.928 * 0.946 * 0.98 * 0.95
  = 0.817
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="photon",
  bottleneck_scores=BottleneckScores(
    photon=0.145, mismatch=0.108, compression=0.04, solver=0.1
  ),
  suggestions=[
    Suggestion(
      text="Increase exposure time to boost per-pixel photon count (currently 659/px)",
      priority="medium",
      expected_gain_db=3.0
    ),
    Suggestion(
      text="PtychoNN provides near-real-time inference but 3.4 dB below ePIE",
      priority="low",
      expected_gain_db=-3.4
    )
  ],
  overall_verdict="excellent",       # P >= 0.80
  probability_of_success=0.817,
  explanation="System is well-configured. Photon budget is the marginal bottleneck; "
              "zone plate low throughput (15%) limits flux to sample."
)
```

---

## 7. AgentNegotiator --- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="good" AND verdict="excellent" | No veto |
| Severe mismatch without correction | severity=0.108 < 0.7 | No veto |
| All marginal | All good/excellent | No veto |
| Joint probability floor | P=0.817 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.85    # tier_prob["good"]
P_recoverability = 0.96    # recoverability_score
P_mismatch       = 1.0 - 0.108 * 0.7 = 0.924

P_joint = 0.85 * 0.96 * 0.924 = 0.754
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],               # No vetoes
  proceed=True,
  probability_of_success=0.754
)
```

---

## 8. PreFlightReportBuilder --- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 256 * 256 = 65,536
dim_factor   = total_pixels / (256 * 256) = 1.0
solver_complexity = 4.0  # ePIE iterative (200 iterations, 16 patterns each)
n_positions  = 16
iter_factor  = 200 / 100  # reference: 100 iterations

runtime_s = 2.0 * 1.0 * 4.0 * n_positions * iter_factor * 0.125
          = 2.0 * 1.0 * 4.0 * 16 * 2.0 * 0.125
          = 32.0 seconds
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="ptychography", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=32.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=[
    "diffraction patterns (intensity, [N_positions, Dp, Dp])",
    "scan positions (micrometers, [N_positions, 2])",
    "probe initial guess (optional, complex [Dp, Dp])"
  ]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Ptychography forward model: I_j = |FFT(P(r) * O(r - r_j))|^2
#
# Parameters:
#   probe:      (Dp, Dp) complex probe function
#   positions:  (N_pos, 2) scan positions in pixels
#   obj_shape:  (N, N) = (256, 256) object grid
#   wavelength: 0.155 nm
#
# Input:  O = (256, 256) complex object (amplitude * exp(i*phase))
# Output: I = (16, 128, 128) diffraction pattern intensities

class PtychographyOperator(PhysicsOperator):
    def forward(self, obj):
        """I_j = |FFT(P * O_j)|^2 for each scan position j"""
        I = np.zeros((n_pos, Dp, Dp), dtype=np.float32)
        for j in range(n_pos):
            py, px = positions[j]
            exit_wave = probe * obj[py:py+Dp, px:px+Dp]
            I[j] = np.abs(np.fft.fft2(exit_wave))**2
        return I

    def adjoint(self, I):
        """Gradient-based pseudo-adjoint for nonlinear model"""
        # Not a true adjoint (nonlinear operator)
        # Used only for initialization; ePIE handles inversion
        obj_est = np.zeros(obj_shape, dtype=np.complex64)
        weight = np.zeros(obj_shape, dtype=np.float32)
        for j in range(n_pos):
            py, px = positions[j]
            obj_est[py:py+Dp, px:px+Dp] += np.conj(probe)
            weight[py:py+Dp, px:px+Dp] += np.abs(probe)**2
        return obj_est / np.maximum(weight, 1e-8)

    def check_adjoint(self):
        """Nonlinear operator: adjoint check replaced by gradient check"""
        # Verifies d/dx ||F(x+eps*h) - F(x)||^2 matches <grad, h>
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-4)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided diffraction.npy:
I_measured = np.load("diffraction.npy")  # (16, 128, 128) float
positions = np.load("positions.npy")     # (16, 2) float -> integer pixels

# If simulating:
obj_true = amplitude * np.exp(1j * phase)  # (256, 256) complex
probe = np.ones((16, 16), dtype=np.complex64)  # initial flat probe
I_measured = operator.forward(obj_true)
I_measured = np.random.poisson(I_measured).astype(np.float32)  # Shot noise
```

### Step 9c: Reconstruction

**Algorithm 1: ePIE (extended Ptychographic Iterative Engine)**

```python
from pwm_core.recon.ptychography_solver import epie

obj_hat, probe_hat = epie(
    diffraction_patterns=I_measured,  # (16, 128, 128)
    positions=positions,               # (16, 2)
    obj_shape=(256, 256),
    probe_init=probe,                  # (128, 128) complex
    iterations=200,
    alpha=1.0,                         # object update step
    beta=1.0,                          # probe update step
)
# obj_hat: (256, 256) complex -- recovered amplitude + phase
# probe_hat: (128, 128) complex -- refined probe function
# Expected PSNR (amplitude): ~59.2 dB (Siemens star benchmark)
```

**ePIE update rules per position j:**

```python
# 1. Form exit wave
psi_j = probe * obj[py:py+Dp, px:px+Dp]

# 2. Propagate to detector
Psi_j = FFT(psi_j)

# 3. Modulus constraint: replace amplitude with measured sqrt(I)
Psi_j_corrected = sqrt(I_j) * Psi_j / (|Psi_j| + eps)

# 4. Back-propagate
psi_j_corrected = IFFT(Psi_j_corrected)

# 5. Update object
delta = psi_j_corrected - psi_j
obj[py:py+Dp, px:px+Dp] += alpha * conj(probe) / max(|probe|^2) * delta

# 6. Update probe
probe += beta * conj(obj_patch) / max(|obj_patch|^2) * delta
```

**Algorithm 2: PtychoNN (deep learning, real-time)**

```python
from pwm_core.recon.ptychonn import ptychonn_infer

# PtychoNN maps diffraction patterns directly to amplitude + phase patches
amp_hat, phase_hat = ptychonn_infer(
    diffraction_patterns=I_measured,  # (16, 128, 128)
    positions=positions,
    obj_shape=(256, 256),
    model_path="weights/ptychonn_4.7M.pth",
    device="cuda"
)
# Expected PSNR (amplitude): ~55.8 dB
# ~100x faster than ePIE but slightly lower quality
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| ePIE | Iterative | 59.2 dB | No | `epie(I, pos, shape, iterations=200)` |
| PtychoNN | Deep Learning | 55.8 dB | Yes | `ptychonn_infer(I, pos, shape)` |
| ePIE (photon-starved) | Iterative | 42.5 dB | No | `epie(I, pos, shape, iterations=500)` |

### Step 9d: Metrics

```python
# Amplitude PSNR
amp_recon = np.abs(obj_hat)
amp_true = np.abs(obj_true)
psnr_amp = 10 * np.log10(amp_true.max()**2 / np.mean((amp_recon - amp_true)**2))

# Phase RMSE (radians) -- ptychography-specific
phase_recon = np.angle(obj_hat)
phase_true = np.angle(obj_true)
# Remove global phase ambiguity (constant offset)
phase_diff = phase_recon - phase_true
phase_diff -= np.mean(phase_diff)
phase_rmse = np.sqrt(np.mean(phase_diff**2))

# SSIM on amplitude
ssim_amp = structural_similarity(amp_recon, amp_true)

# Resolution (Fourier Ring Correlation -- FRC)
# Split data into two halves, reconstruct independently, compute FRC
frc = fourier_ring_correlation(obj_hat_half1, obj_hat_half2)
resolution_nm = wavelength_nm / (2 * NA_effective)  # Rayleigh limit
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
|   +-- I_measured.npy     # Diffraction patterns (16, 128, 128) + SHA256
|   +-- obj_hat.npy        # Reconstructed object (256, 256) complex + SHA256
|   +-- probe_hat.npy      # Recovered probe (128, 128) complex + SHA256
|   +-- obj_true.npy       # Ground truth (if available) + SHA256
+-- metrics.json           # PSNR (amp), phase RMSE, SSIM, FRC resolution
+-- operator.json          # Operator parameters (positions hash, wavelength, Dp)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (ePIE -> PtychoNN) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Ptychography-specific:** Nonlinear forward model, complex-valued reconstruction, joint probe-object recovery, phase ambiguity handling in metrics.
