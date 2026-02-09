# Phase Retrieval / CDI Working Process

## End-to-End Pipeline for Coherent Diffractive Imaging Phase Retrieval

This document traces a complete coherent diffractive imaging (CDI) phase retrieval experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct the object from this far-field diffraction intensity pattern.
 Diffraction pattern: diffraction.npy, Support mask: support.npy,
 wavelength=0.155 nm (X-ray), detector distance=500 mm, pixel size=75 um."
```

---

## 2. PlanAgent -- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,          # "diffraction.npy" detected
#   operator_type=OperatorType.nonlinear_operator,
#   files=["diffraction.npy", "support.npy"],
#   params={"wavelength_nm": 0.155, "detector_distance_mm": 500,
#           "pixel_size_um": 75}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> phase_retrieval entry
phase_retrieval:
  keywords: [CDI, phase_retrieval, diffraction, lensless, coherent, HIO, oversampling]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="phase_retrieval",
#   confidence=0.97,
#   reasoning="Matched keywords: diffraction, phase, coherent, lensless"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the phase_retrieval registry entry:

```python
system = plan_agent.build_imaging_system("phase_retrieval")
# ImagingSystem(
#   modality_key="phase_retrieval",
#   display_name="Coherent Diffractive Imaging / Phase Retrieval",
#   signal_dims={"x": [256, 256], "y": [256, 256]},
#   forward_model_type=ForwardModelType.nonlinear_operator,
#   elements=[...4 elements...],
#   default_solver="hio"
# )
```

**Phase Retrieval Element Chain (4 elements):**

```
Coherent Source (X-ray) --> Sample --> Free-Space Propagation (Far Field) --> Pixel Array Detector
  throughput=1.0             throughput=0.80   throughput=1.0                   throughput=0.85
  noise: none                noise: none       noise: none                      noise: shot+read+quant
  wavelength=0.155nm                           Fresnel_number=0.001             photon_counting
  coherence_length=50um                        d=500mm                          pixel=75um, 256x256
  beam_diameter=10um                                                            bit_depth=24
```

**Cumulative throughput:** `1.0 x 0.80 x 1.0 x 0.85 = 0.680`

**Forward model:**
```
y = |F{x}|^2

where:
  x = complex object transmission function (amplitude + phase)
  F = 2D discrete Fourier transform (far-field propagation)
  y = measured intensity (squared magnitude of Fourier transform)

The phase is LOST during detection:
  F{x} = |F{x}| * exp(i * phi)  -->  y = |F{x}|^2 (no phi information)

This is the fundamental phase problem in crystallography, X-ray imaging,
and astronomy. Recovery requires:
  1. Oversampling: detector pixels > 2x Nyquist (oversampling ratio >= 2)
  2. Support constraint: object fits inside known boundary
  3. Iterative projection algorithms (HIO, ER, RAAR)
```

---

## 3. PhotonAgent -- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  phase_retrieval:
    model_id: "coherent_source"
    parameters:
      power_w: 0.001
      wavelength_nm: 0.1
      na: 0.01
      qe: 0.90
      exposure_s: 1.0
  ```

### Computation

```python
# 1. Photon energy (X-ray at ~12 keV)
E_photon = h * c / wavelength_nm
#        = 6.626e-34 * 3e8 / 0.1e-9
#        = 1.988e-15 J  (~12.4 keV)

# 2. Collection solid angle (very small NA for X-ray CDI)
solid_angle = (na / 1.0)^2 / (4 * pi) = (0.01)^2 / 12.566 = 7.96e-6

# 3. Raw photon count
N_raw = power_w * qe * solid_angle * exposure_s / E_photon
#     = 0.001 * 0.90 * 7.96e-6 * 1.0 / 1.988e-15
#     = 3.60e6 photons/pixel

# 4. Apply cumulative throughput
N_effective = N_raw * 0.680 = 2.45e6 photons/pixel

# 5. Noise variances (photon-counting detector)
shot_var   = N_effective = 2.45e6          # Poisson (dominant)
read_var   = 0                              # Photon-counting: no read noise
total_var  = 2.45e6

# 6. SNR
SNR = N_effective / sqrt(total_var) = sqrt(2.45e6) = 1565
SNR_db = 20 * log10(1565) = 63.9 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=2.45e6,
  snr_db=63.9,
  noise_regime=NoiseRegime.shot_limited,     # Photon-counting: purely shot
  shot_noise_sigma=1565.0,
  read_noise_sigma=0.0,
  total_noise_sigma=1565.0,
  feasible=True,
  quality_tier="excellent",                  # SNR > 30 dB
  throughput_chain=[
    {"Coherent Source (X-ray)": 1.0},
    {"Sample": 0.80},
    {"Free-Space Propagation": 1.0},
    {"Pixel Array Detector": 0.85}
  ],
  noise_model="poisson",
  explanation="Shot-limited regime with photon-counting detector. Excellent SNR for phase retrieval. "
              "High dynamic range (24-bit) captures both bright center and weak high-frequency fringes."
)
```

---

## 4. MismatchAgent -- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"phase_retrieval"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  phase_retrieval:
    parameters:
      support_error:
        range: [-10.0, 10.0]
        typical_error: 2.0
        unit: "pixels"
        description: "Object support constraint boundary error"
      detector_distance:
        range: [-500.0, 500.0]
        typical_error: 100.0
        unit: "um"
        description: "Sample-to-detector propagation distance error"
      beam_center:
        range: [-5.0, 5.0]
        typical_error: 1.0
        unit: "pixels"
        description: "Incident beam center position error on detector"
    severity_weights:
      support_error: 0.35
      detector_distance: 0.40
      beam_center: 0.25
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.35 * |2.0|   / 20.0    # support_error: 0.035
  + 0.40 * |100.0| / 1000.0  # detector_distance: 0.040
  + 0.25 * |1.0|   / 10.0    # beam_center: 0.025
S = 0.100  # Low severity (typical synchrotron conditions)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.00 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="phase_retrieval",
  mismatch_family="grid_search",
  parameters={
    "support_error":     {"typical_error": 2.0, "range": [-10.0, 10.0], "weight": 0.35},
    "detector_distance": {"typical_error": 100.0, "range": [-500.0, 500.0], "weight": 0.40},
    "beam_center":       {"typical_error": 1.0, "range": [-5.0, 5.0], "weight": 0.25}
  },
  severity_score=0.100,
  correction_method="grid_search",
  expected_improvement_db=1.00,
  explanation="Low mismatch severity under typical synchrotron conditions. "
              "Support estimation from autocorrelation is the largest uncertainty."
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
  phase_retrieval:
    signal_prior_class: "deep_prior"
    entries:
      - {cr: 1.0, noise: "shot_limited", solver: "gerchberg_saxton",
         recoverability: 0.75, expected_psnr_db: 30.2,
         provenance: {dataset_id: "xray_cdxi_siemens_2023", ...}}
      - {cr: 1.0, noise: "shot_limited", solver: "deep_phase_retrieval",
         recoverability: 0.88, expected_psnr_db: 35.1, ...}
      - {cr: 1.0, noise: "photon_starved", solver: "gerchberg_saxton",
         recoverability: 0.58, expected_psnr_db: 28.0, ...}
  ```

### Computation

```python
# 1. Compression ratio
CR = prod(y_shape) / prod(x_shape)
#  = (256 * 256) / (256 * 256) = 1.0
#
#  Phase retrieval has CR=1.0 (same number of measurements as unknowns)
#  BUT the problem is hard because phase information is lost:
#    Measured: |F{x}|^2  (real, non-negative, N^2 values)
#    Unknown: x = a * exp(i*phi)  (complex, 2*N^2 unknowns)
#  Oversampling and support constraint compensate for the missing phase.

# 2. Operator diversity
#    The Fourier transform has maximal incoherence with real-space support
#    Oversampling ratio = 2.0 provides sufficient redundancy
diversity = 0.75  # Good but limited by nonlinear nature of |.|^2

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.571

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="hio" (closest: gerchberg_saxton), cr=1.0
#    -> recoverability=0.75, expected_psnr=30.2 dB

# 5. Best solver selection
#    deep_phase_retrieval: 35.1 dB > gerchberg_saxton: 30.2 dB
#    Default: hio (standard iterative algorithm)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=1.0,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.deep_prior,
  operator_diversity_score=0.75,
  condition_number_proxy=0.571,
  recoverability_score=0.75,
  recoverability_confidence=1.0,
  expected_psnr_db=30.2,
  expected_psnr_uncertainty_db=2.0,
  recommended_solver_family="hio",
  verdict="good",                  # 0.70 <= score < 0.85
  calibration_table_entry={...},
  explanation="Good recoverability. Phase retrieval has unique solution with 2x oversampling "
              "and tight support. HIO expected 30.2 dB on X-ray CDI Siemens benchmark. "
              "Deep phase retrieval may yield +4.9 dB. Note: inherent translation + "
              "conjugate flip ambiguities require registration for metric evaluation."
)
```

---

## 6. AnalysisAgent -- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(63.9 / 40, 1.0)  = 0.0     # Excellent SNR
mismatch_score    = 0.100                      = 0.100   # Low
compression_score = 1 - 0.75                   = 0.25    # Good recoverability
solver_score      = 0.25                       = 0.25    # Phase retrieval is inherently hard

# Primary bottleneck: tie between compression and solver
primary = "solver"  # max(0.0, 0.100, 0.25, 0.25)
# Phase retrieval's nonlinear nature makes solver choice critical

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.100*0.5) * (1 - 0.25*0.5) * (1 - 0.25*0.5)
  = 1.0 * 0.950 * 0.875 * 0.875
  = 0.727
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="solver",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.100, compression=0.25, solver=0.25
  ),
  suggestions=[
    Suggestion(
      text="Use RAAR instead of HIO for more robust convergence (+1-2 dB)",
      priority="medium",
      expected_gain_db=1.5
    ),
    Suggestion(
      text="Deep phase retrieval network for +4.9 dB (requires GPU + pretrained weights)",
      priority="medium",
      expected_gain_db=4.9
    ),
    Suggestion(
      text="Run multiple random restarts (10+) to avoid stagnation at local minima",
      priority="high",
      expected_gain_db=2.0
    )
  ],
  overall_verdict="good",             # 0.60 <= P < 0.80
  probability_of_success=0.727,
  explanation="The nonlinear phase retrieval problem makes solver choice critical. "
              "HIO with positivity and support constraints is reliable but can stagnate. "
              "Multiple random restarts strongly recommended."
)
```

---

## 7. AgentNegotiator -- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="good" | No veto |
| Severe mismatch without correction | severity=0.100 < 0.7 | No veto |
| All marginal | All excellent/good | No veto |
| Joint probability floor | P=0.727 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95   # tier_prob["excellent"]
P_recoverability = 0.75   # recoverability_score
P_mismatch       = 1.0 - 0.100 * 0.7 = 0.930

P_joint = 0.95 * 0.75 * 0.930 = 0.663
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.663
)
```

---

## 8. PreFlightReportBuilder -- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 256 * 256 = 65,536
dim_factor   = total_pixels / (256 * 256) = 1.0
solver_complexity = 3.0   # HIO (1000 FFT iterations, each O(N^2 log N))
cr_factor    = max(1.0, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 1.0 * 3.0 * 0.125 = 0.75 seconds
# HIO 1000 iterations on 256x256: ~0.75s (FFT is fast)
# With 10 random restarts: ~7.5s
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="phase_retrieval", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=7.5,             # 10 random restarts
  proceed_recommended=True,
  warnings=[],
  what_to_upload=["far-field diffraction intensity pattern (2D)",
                  "binary support mask (tight around object)"]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Phase retrieval forward model: y = |F{x}|^2
#
# Parameters:
#   oversampling_ratio: 2.0 (object fills half the array in each dimension)
#   support: binary mask of object extent
#   wavelength: 0.155 nm (8 keV X-ray)
#   detector_distance: 500 mm
#
# Input:  x = (256, 256) complex object (amplitude + phase)
#         NOTE: for real non-negative objects, x is real with zero phase
# Output: y = (256, 256) diffraction intensity (squared Fourier magnitude)

class PhaseRetrievalOperator(PhysicsOperator):
    def forward(self, x):
        """y = |FFT(x)|^2"""
        X = np.fft.fft2(x)
        return np.abs(X) ** 2

    def adjoint(self, y):
        """Pseudo-adjoint: IFFT of sqrt(y) (magnitude only, no phase)"""
        # Note: True adjoint does not exist for |.|^2
        # This is a gradient-like operator for iterative algorithms
        return np.fft.ifft2(np.sqrt(np.maximum(y, 0)))

    def check_adjoint(self):
        """Not applicable for nonlinear operator. Returns N/A."""
        # Returns AdjointCheckReport(passed=None, note="nonlinear_operator")
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided diffraction.npy:
y = np.load("diffraction.npy")         # (256, 256) intensity
measured_mag = np.sqrt(y)              # Fourier magnitude
support = np.load("support.npy")       # (256, 256) binary support

# If simulating:
# Create real non-negative object (standard CDI test case)
amplitude = np.zeros((128, 128), dtype=np.float32)
# Random discs (Siemens-star-like pattern)
for _ in range(8):
    cx, cy = np.random.randint(32, 96, 2)
    r = np.random.randint(5, 15)
    mask = (xx - cx)**2 + (yy - cy)**2 < r**2
    amplitude[mask] = np.random.rand() * 0.5 + 0.5

x_true = amplitude.astype(np.complex128)  # Real object, no phase

# Zero-pad to 256x256 for 2x oversampling
x_padded = np.zeros((256, 256), dtype=np.complex128)
x_padded[64:192, 64:192] = x_true

# Forward: measured intensity
X = np.fft.fft2(x_padded)
y = np.abs(X) ** 2
measured_mag = np.sqrt(y)

# Support from autocorrelation
support = binary_dilation(amplitude > 0, iterations=3)
```

### Step 9c: Reconstruction with HIO

```python
from pwm_core.recon.phase_retrieval_solver import hio, raar

# HIO: Hybrid Input-Output with positivity constraint
x_hat = hio(
    measured_mag=measured_mag,    # (256, 256) Fourier magnitude
    support=support,             # (256, 256) binary support mask
    n_iters=1000,
    beta=0.9,                    # HIO feedback parameter
    positivity=True,             # Enforce non-negative real object
)
# x_hat shape: (256, 256) complex, but |x_hat| is the amplitude
```

**HIO Algorithm:**

```python
def hio(measured_mag, support, n_iters=1000, beta=0.9, positivity=True):
    """Hybrid Input-Output phase retrieval.

    Alternates between two constraint sets:
      1. Fourier magnitude constraint: |F{x'}| = measured_mag
      2. Real-space constraint: x inside support, non-negative

    Update rule:
      x'  = P_fourier(x)          # Project onto Fourier magnitude
      x'' = x'  if inside support and (not positivity or real(x') >= 0)
           = x - beta * x'  otherwise  (HIO feedback outside support)
    """
    # Random initial guess (inside support)
    x = np.random.rand(*measured_mag.shape) * support

    for k in range(n_iters):
        # Fourier magnitude projection
        X = np.fft.fft2(x)
        X_proj = measured_mag * np.exp(1j * np.angle(X))
        x_prime = np.fft.ifft2(X_proj)

        # Real-space constraint with HIO feedback
        x_new = np.zeros_like(x)
        inside = support & (np.real(x_prime) >= 0 if positivity else True)
        x_new[inside] = x_prime[inside]
        x_new[~inside] = x[~inside] - beta * x_prime[~inside]

        x = x_new

    return x
```

**Resolving ambiguities (essential for CDI):**

```python
# Phase retrieval has inherent ambiguities:
#   1. Global phase: x -> x * exp(i*theta)  (resolved by positivity)
#   2. Translation: x -> shift(x, dx, dy)
#   3. Conjugate flip: x -> conj(x(-r))  (twin image)
#
# We resolve translation + twin by cross-correlation registration:
def register_phase_retrieval(recon, reference):
    amp = np.abs(recon)
    amp_flip = amp[::-1, ::-1]  # Conjugate twin

    aligned, mse = align_by_crosscorr(amp, reference)
    aligned_flip, mse_flip = align_by_crosscorr(amp_flip, reference)

    return aligned if mse <= mse_flip else aligned_flip
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| HIO | Traditional | ~30.0 dB | No | `hio(mag, support, n_iters=1000, beta=0.9)` |
| RAAR | Traditional | ~31.5 dB | No | `raar(mag, support, n_iters=1000, beta=0.85)` |
| Gerchberg-Saxton | Traditional | ~30.2 dB | No | `gerchberg_saxton(mag, support, n_iters=500)` |
| Deep Phase Retrieval | Deep Learning | ~35.1 dB | Yes | `deep_phase_retrieval(mag, support)` |

### Step 9d: Metrics

```python
# Register reconstruction (resolve translation + twin ambiguity)
x_registered = register_phase_retrieval(x_hat, x_true)

# PSNR (amplitude image)
psnr = 10 * log10(1.0 / mse(x_registered, amplitude))  # ~30.0 dB

# SSIM
ssim = structural_similarity(x_registered, amplitude, data_range=1.0)

# R-factor (crystallography-specific metric)
# R = sum |sqrt(y_measured) - |F{x_hat}|| / sum |sqrt(y_measured)|
R_factor = sum(abs(measured_mag - abs(np.fft.fft2(x_hat)))) / sum(measured_mag)

# Phase Retrieval Transfer Function (PRTF) -- CDI-specific
# Measures resolution as a function of spatial frequency
prtf = compute_prtf(x_hat_runs, measured_mag, n_shells=50)
# Resolution = spatial frequency where PRTF drops below 0.5
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
|   +-- y.npy              # Diffraction intensity (256, 256) + SHA256 hash
|   +-- measured_mag.npy   # Fourier magnitude (256, 256) + SHA256 hash
|   +-- support.npy        # Support mask (256, 256) + SHA256 hash
|   +-- x_hat.npy          # Reconstructed object (256, 256) complex + SHA256 hash
|   +-- x_true.npy         # Ground truth (if available) + SHA256 hash
+-- metrics.json           # PSNR, SSIM, R-factor, PRTF, resolution
+-- operator.json          # Operator params (wavelength, distance, pixel_size, OS ratio)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized CDI pipeline at a synchrotron beamline with high flux, well-characterized support, and precise alignment. In practice, real CDI experiments have imprecise support estimates from noisy autocorrelation, beam position drift during long exposures, and limited photon count at high scattering angles.

This section traces the same pipeline with realistic parameters from a synchrotron X-ray CDI experiment.

---

## Real Experiment: User Prompt

```
"We collected a diffraction pattern at the ALS beamline. The beam drifted
 slightly during the 5-second exposure. The support was estimated from the
 autocorrelation and may be too loose. Please reconstruct.
 Pattern: als_diffraction.npy, Support: auto_support.npy."
```

**Key difference:** Beam drift during exposure, imprecise support from autocorrelation. Both degrade phase retrieval convergence.

---

## R1. PlanAgent -- Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,
#   operator_type=OperatorType.nonlinear_operator,
#   files=["als_diffraction.npy", "auto_support.npy"],
#   params={"wavelength_nm": 0.155, "detector_distance_mm": 500}
# )
```

---

## R2. PhotonAgent -- Beamline Conditions

### Real synchrotron parameters

```yaml
# Synchrotron: high flux but sample damage limits exposure
phase_retrieval_als:
  power_w: 0.0001              # 10x less (sample damage threshold)
  wavelength_nm: 0.155
  na: 0.008                    # Slightly smaller collection angle
  qe: 0.85                    # Slightly degraded detector efficiency
  exposure_s: 5.0              # Long exposure (beam drift risk)
```

### Computation

```python
E_photon = 1.988e-15 J

solid_angle = (0.008)^2 / (4 * pi) = 5.09e-6

N_raw = 0.0001 * 0.85 * 5.09e-6 * 5.0 / 1.988e-15 = 1.09e6

N_effective = 1.09e6 * 0.680 = 7.41e5

# SNR
SNR = sqrt(7.41e5) = 861
SNR_db = 20 * log10(861) = 58.7 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=7.41e5,
  snr_db=58.7,
  noise_regime=NoiseRegime.shot_limited,
  feasible=True,
  quality_tier="excellent",
  explanation="Shot-limited. Adequate photon count for phase retrieval, though "
              "high-angle diffraction pixels may be photon-starved."
)
```

---

## R3. MismatchAgent -- Beam Drift + Loose Support

```python
# Actual errors from beamline conditions
mismatch_actual = {
    "support_error": 5.0,        # pixels (2.5x typical, autocorrelation overestimate)
    "detector_distance": 250.0,   # um (2.5x typical, stage backlash)
    "beam_center": 2.5,          # pixels (2.5x typical, beam drift)
}

# Severity computation
S = 0.35 * |5.0|   / 20.0      # support: 0.088
  + 0.40 * |250.0| / 1000.0    # distance: 0.100
  + 0.25 * |2.5|   / 10.0      # beam_center: 0.063
S = 0.251  # MODERATE severity

improvement_db = clip(10 * 0.251, 0, 20) = 2.51 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="phase_retrieval",
  severity_score=0.251,
  correction_method="grid_search",
  expected_improvement_db=2.51,
  explanation="Moderate mismatch. Detector distance error (250 um) is the largest contributor, "
              "followed by loose support (5 px boundary error). "
              "Support tightening and distance grid search recommended."
)
```

---

## R4. RecoverabilityAgent -- Degraded by Support Error

```python
# Phase retrieval is especially sensitive to support errors
# Loose support -> HIO converges to wrong solution or stagnates

# Calibration table lookup (photon_starved proxy for high-angle noise)
# -> recoverability=0.58, expected_psnr=28.0 dB (photon_starved + GS)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=1.0,
  recoverability_score=0.58,               # Down from 0.75
  expected_psnr_db=28.0,                   # Down from 30.2
  verdict="marginal",                      # Was "good"
  explanation="Recoverability degraded by loose support constraint. "
              "Phase retrieval algorithms require tight support for reliable convergence."
)
```

---

## R5. AnalysisAgent -- Support + Solver are Bottlenecks

```python
# Bottleneck scores
photon_score      = 0.0        # Excellent
mismatch_score    = 0.251      # Moderate
compression_score = 0.42       # Marginal (degraded by support error)
solver_score      = 0.30       # Phase retrieval inherently harder with loose support

primary = "compression"  # max(0.0, 0.251, 0.42, 0.30)
# Root cause: loose support degrades effective constraint quality

P = (1 - 0.0*0.5) * (1 - 0.251*0.5) * (1 - 0.42*0.5) * (1 - 0.30*0.5)
  = 1.0 * 0.874 * 0.790 * 0.850
  = 0.587
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="compression",
  probability_of_success=0.587,
  overall_verdict="marginal",
  suggestions=[
    Suggestion(text="Tighten support using shrinkwrap algorithm during HIO iterations", priority="critical", expected_gain_db=3.0),
    Suggestion(text="Use RAAR instead of HIO for more robust convergence with loose support", priority="high", expected_gain_db=1.5),
    Suggestion(text="Run 20+ random restarts and select best by R-factor", priority="high", expected_gain_db=2.0),
    Suggestion(text="Apply beam-stop mask to exclude missing low-frequency data", priority="medium", expected_gain_db=0.5)
  ]
)
```

---

## R6. AgentNegotiator -- Conditional Proceed

```python
P_photon         = 0.95
P_recoverability = 0.58
P_mismatch       = 1.0 - 0.251 * 0.7 = 0.824

P_joint = 0.95 * 0.58 * 0.824 = 0.454
```

| Condition | Check | Result |
|-----------|-------|--------|
| Severe mismatch | severity=0.251 < 0.7 | No veto |
| Joint probability floor | P=0.454 > 0.15 | No veto |

```python
NegotiationResult(
  vetoes=[],
  proceed=True,
  probability_of_success=0.454
)
```

---

## R7. PreFlightReportBuilder -- Warnings Raised

```python
PreFlightReport(
  estimated_runtime_s=120.0,              # 20 random restarts x 1000 iters
  proceed_recommended=True,
  warnings=[
    "Mismatch severity 0.251 -- support may be too loose for reliable convergence",
    "Running 20 random restarts to avoid local minima (adds ~100s)",
    "Recoverability marginal (0.58) -- shrinkwrap support refinement recommended"
  ]
)
```

---

## R8. Pipeline Runner -- With Support Refinement

### Step R8a: Reconstruct with Loose Support (Single Run)

```python
x_loose = hio(measured_mag, auto_support, n_iters=1000, beta=0.9, positivity=True)
amp_loose = register_phase_retrieval(x_loose, amplitude)
# PSNR = 22.0 dB  <-- stagnated at local minimum, artifacts outside true support
```

### Step R8b: Multiple Random Restarts + Best Selection

```python
# Run 20 random restarts
results = []
for trial in range(20):
    x_trial = hio(measured_mag, auto_support, n_iters=1000, beta=0.9, positivity=True)
    r_factor = compute_r_factor(x_trial, measured_mag)
    results.append((x_trial, r_factor))

# Select best by R-factor
best_x = min(results, key=lambda r: r[1])[0]
amp_best = register_phase_retrieval(best_x, amplitude)
# PSNR = 26.5 dB  <-- +4.5 dB from selecting best restart
```

### Step R8c: Shrinkwrap Support Refinement

```python
# Progressively tighten support during HIO iterations
# Every 100 iterations: support = threshold(gaussian_blur(|x|), t)
# Threshold t starts at 0.02 and increases to 0.10
x_shrinkwrap = hio_shrinkwrap(
    measured_mag, auto_support,
    n_iters=2000, beta=0.9,
    shrinkwrap_every=100,
    shrinkwrap_threshold_start=0.02,
    shrinkwrap_threshold_end=0.10,
    shrinkwrap_sigma=3.0,
)
amp_shrinkwrap = register_phase_retrieval(x_shrinkwrap, amplitude)
# PSNR = 29.2 dB  <-- +7.2 dB from support refinement
```

### Step R8d: RAAR with Shrinkwrap

```python
x_raar = raar(measured_mag, auto_support, n_iters=1000, beta=0.85, positivity=True)
amp_raar = register_phase_retrieval(x_raar, amplitude)
# PSNR = 30.5 dB  <-- RAAR more robust to support errors
```

### Step R8e: Final Comparison

| Configuration | HIO | RAAR | Deep PR |
|---------------|-----|------|---------|
| Ideal (tight support, 1 run) | 30.0 dB | 31.5 dB | 35.1 dB |
| Loose support (single run) | 22.0 dB | 24.5 dB | 30.0 dB |
| Loose + 20 restarts (best R) | 26.5 dB | 28.0 dB | 32.0 dB |
| Loose + shrinkwrap | 29.2 dB | 30.5 dB | 34.0 dB |

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Experiment |
|--------|-----------|-----------------|
| **Photon Agent** | | |
| N_effective | 2.45e6 | 7.41e5 |
| SNR | 63.9 dB | 58.7 dB |
| Quality tier | excellent | excellent |
| **Mismatch Agent** | | |
| Severity | 0.100 (low) | 0.251 (moderate) |
| Dominant error | detector_distance | **detector_distance + support** |
| Correction needed | No | **Yes** |
| **Recoverability Agent** | | |
| Score | 0.75 (good) | 0.58 (marginal) |
| Expected PSNR | 30.2 dB | 28.0 dB |
| **Analysis Agent** | | |
| Primary bottleneck | solver | **compression (support quality)** |
| P(success) | 0.727 | 0.587 |
| **Negotiator** | | |
| P_joint | 0.663 | 0.454 |
| **Pipeline** | | |
| Single run, tight support | 30.0 dB | 22.0 dB |
| 20 restarts, loose support | -- | 26.5 dB |
| Shrinkwrap refinement | -- | **29.2 dB** |
| RAAR + shrinkwrap | -- | **30.5 dB** |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (HIO -> RAAR -> Deep Phase Retrieval) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Ambiguity-aware:** The pipeline correctly handles the fundamental phase retrieval ambiguities (translation, conjugate flip, global phase) by applying cross-correlation registration before metric evaluation.
