# Holography Working Process

## End-to-End Pipeline for Digital Holographic Microscopy

This document traces a complete digital holography experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct the amplitude and phase of cells from an off-axis hologram.
 Hologram: hologram.tif, wavelength=633 nm, pixel_size=4.65 um,
 propagation_distance=50 um."
```

---

## 2. PlanAgent --- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "hologram.tif" detected
#   operator_type=OperatorType.nonlinear_operator,
#   files=["hologram.tif"],
#   params={"wavelength_nm": 633, "pixel_size_um": 4.65,
#           "propagation_distance_um": 50}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> holography entry
holography:
  keywords: [holography, off_axis, phase_retrieval, interference, DHM]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="holography",
#   confidence=0.96,
#   reasoning="Matched keywords: holography, off_axis, phase_retrieval"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the holography registry entry:

```python
system = plan_agent.build_imaging_system("holography")
# ImagingSystem(
#   modality_key="holography",
#   display_name="Digital Holographic Microscopy",
#   signal_dims={"x": [512, 512], "y": [512, 512]},
#   forward_model_type=ForwardModelType.nonlinear_operator,
#   elements=[...5 elements...],
#   default_solver="angular_spectrum"
# )
```

**Holography Element Chain (5 elements):**

```
HeNe Laser (633nm) ---> Beam Splitter ---> Microscope Obj (40x) ---> Off-Axis Reference ---> CCD Detector
  throughput=1.0         throughput=0.50    throughput=0.80          throughput=1.0          throughput=0.70
  noise: none            noise: none        noise: aberration        noise: alignment        noise: shot+read+quant
  power=5 mW             50/50 cube         NA=0.65, air             carrier=0.2 cyc/px      pixel=4.65 um
  coherence=0.3 m                                                    tilt=2.5 deg            QE=0.70, 12-bit
```

**Cumulative throughput:** `1.0 x 0.50 x 0.80 x 1.0 x 0.70 = 0.280`

**Forward model:** `I = |U_obj * exp(i*phi) + U_ref * exp(i*k*sin(theta)*r)|^2`

The recorded hologram is the squared modulus of the sum of object wave (bearing sample information) and a tilted reference wave. Off-axis geometry separates the object spectrum from the DC and twin-image terms in Fourier space.

---

## 3. PhotonAgent --- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  holography:
    model_id: "generic_detector"
    parameters:
      source_photons: 1.0e+07
      qe: 0.85
      exposure_s: 0.01
  ```

### Computation

```python
# 1. Source photon count
N_source = 1.0e7 photons/exposure

# 2. Apply cumulative throughput
N_effective = N_source * 0.280 = 2.80e6 photons total

# 3. Photons per pixel
# 512 x 512 detector = 262,144 pixels
N_per_pixel = N_effective / (512 * 512) = 10.7 photons/pixel

# 4. Note: off-axis holography has additional DC + cross terms
# Object beam carries ~25% of total intensity at detector
# Reference beam carries ~25%, DC carries ~50%
N_signal_pixel = N_per_pixel * 0.25 = 2.67 photons/pixel (object contribution)

# 5. Noise variances
shot_var   = N_per_pixel = 10.7              # Poisson (total intensity)
read_var   = 8.0^2 = 64.0                    # CCD read noise (8 e-)
dark_var   = 0.0                              # Short exposure, negligible
total_var  = shot_var + read_var = 74.7

# 6. SNR
SNR = N_signal_pixel / sqrt(total_var) = 2.67 / 8.64 = 0.309
SNR_db = 20 * log10(0.309) = -10.2 dB

# But: off-axis filtering extracts the +1 order, rejecting DC noise
# Effective SNR after Fourier filtering: 6-10 dB gain
SNR_filtered_db = -10.2 + 8.0 = -2.2 dB  # marginal per pixel

# Spatial averaging over PSF (NA=0.65, lambda=633nm, ~2 px Airy radius):
# 12 pixels contribute -> sqrt(12) = 3.5x gain
SNR_effective_db = -2.2 + 20*log10(3.5) = -2.2 + 10.9 = 8.7 dB

# NOTE: Low per-pixel SNR, but holography benefits from interferometric
# amplification (reference beam acts as local oscillator)
# With full-well capacity consideration (18000 e-):
# In practice, reference beam is set to fill ~50% of well depth
# N_ref ~ 9000 e- -> interferometric gain
N_ref = 9000
SNR_interferometric = 2 * sqrt(N_ref * N_signal_pixel)
                    = 2 * sqrt(9000 * 2.67) = 2 * 155 = 310
SNR_db_final = 20 * log10(310) = 49.8 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=10.7,
  snr_db=49.8,
  noise_regime=NoiseRegime.shot_limited,    # interferometric detection
  shot_noise_sigma=3.27,
  read_noise_sigma=8.0,
  total_noise_sigma=8.64,
  feasible=True,
  quality_tier="excellent",                  # SNR > 30 dB (interferometric)
  throughput_chain=[
    {"HeNe Laser": 1.0},
    {"Beam Splitter": 0.50},
    {"Microscope Objective": 0.80},
    {"Off-Axis Reference": 1.0},
    {"CCD Detector": 0.70}
  ],
  noise_model="poisson",
  explanation="Interferometric detection with strong reference beam provides "
              "excellent SNR despite low per-pixel photon count. Off-axis "
              "Fourier filtering rejects DC and twin-image noise."
)
```

---

## 4. MismatchAgent --- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"holography"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  holography:
    parameters:
      propagation_distance_error:
        range: [-500.0, 500.0]
        typical_error: 50.0
        unit: "um"
        description: "Numerical propagation distance error for refocusing"
      wavelength_error:
        range: [-5.0, 5.0]
        typical_error: 1.0
        unit: "nm"
        description: "Illumination wavelength uncertainty from source bandwidth"
      tilt_angle:
        range: [-2.0, 2.0]
        typical_error: 0.3
        unit: "degrees"
        description: "Reference beam tilt error in off-axis holography"
    severity_weights:
      propagation_distance_error: 0.45
      wavelength_error: 0.25
      tilt_angle: 0.30
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.45 * |50.0| / 1000.0    # propagation: 0.0225
  + 0.25 * |1.0| / 10.0       # wavelength:  0.025
  + 0.30 * |0.3| / 4.0        # tilt:        0.0225
S = 0.070  # Low severity (HeNe laser is very stable)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 0.70 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="holography",
  mismatch_family="grid_search",
  parameters={
    "propagation_distance_error": {"typical_error": 50.0, "range": [-500, 500], "weight": 0.45},
    "wavelength_error":           {"typical_error": 1.0, "range": [-5, 5], "weight": 0.25},
    "tilt_angle":                 {"typical_error": 0.3, "range": [-2, 2], "weight": 0.30}
  },
  severity_score=0.070,
  correction_method="grid_search",
  expected_improvement_db=0.70,
  explanation="Low mismatch severity. HeNe laser wavelength is extremely stable. "
              "Propagation distance is the main calibration parameter."
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
  holography:
    signal_prior_class: "tv"
    entries:
      - {cr: 1.0, noise: "shot_limited", solver: "angular_spectrum",
         recoverability: 0.93, expected_psnr_db: 42.3,
         provenance: {dataset_id: "holography_usaf_offaxis_2023", ...}}
      - {cr: 1.0, noise: "detector_limited", solver: "angular_spectrum",
         recoverability: 0.82, expected_psnr_db: 36.8, ...}
      - {cr: 1.0, noise: "shot_limited", solver: "phasenet",
         recoverability: 0.95, expected_psnr_db: 45.6, ...}
  ```

### Computation

```python
# 1. Compression ratio
# Holography: single 2D hologram -> single 2D complex field
# Measurement: 512 x 512 real intensity
# Signal: 512 x 512 complex (2 DOF per pixel)
# But off-axis filtering uses only ~1/4 of Fourier space
CR = (512 * 512 * 0.25) / (512 * 512) = 0.25 (usable bandwidth)
# For calibration: CR = 1.0 (holography is not compressed; it is well-determined)

# 2. Operator diversity
# Off-axis reference encodes phase into intensity fringes
# Full complex field recovery -> high diversity
diversity = 0.90  # Well-determined (not compressive)

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.526

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="angular_spectrum", cr=1.0
#    -> recoverability=0.93, expected_psnr=42.3 dB, confidence=1.0

# 5. Best solver selection
#    phasenet: 45.6 dB > angular_spectrum: 42.3 dB
#    -> recommended: "phasenet" (but angular_spectrum is more robust)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=1.0,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.tv,
  operator_diversity_score=0.90,
  condition_number_proxy=0.526,
  recoverability_score=0.93,
  recoverability_confidence=1.0,
  expected_psnr_db=42.3,
  expected_psnr_uncertainty_db=1.5,
  recommended_solver_family="angular_spectrum",
  verdict="excellent",               # score >= 0.85
  calibration_table_entry={...},
  explanation="Excellent recoverability. Off-axis holography is a well-determined "
              "problem. Angular spectrum propagation expected 42.3 dB on USAF target."
)
```

---

## 6. AnalysisAgent --- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(49.8 / 40, 1.0)   = 0.0     # Excellent SNR
mismatch_score    = 0.070                        = 0.070   # Very low
compression_score = 1 - 0.93                     = 0.07    # Excellent recoverability
solver_score      = 0.15                         = 0.15    # Angular spectrum baseline

# Primary bottleneck
primary = "solver"  # max(0.0, 0.070, 0.07, 0.15) = solver

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.070*0.5) * (1 - 0.07*0.5) * (1 - 0.15*0.5)
  = 1.0 * 0.965 * 0.965 * 0.925
  = 0.861
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="solver",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.070, compression=0.07, solver=0.15
  ),
  suggestions=[
    Suggestion(
      text="PhaseNet provides +3.3 dB over angular spectrum method",
      priority="medium",
      expected_gain_db=3.3
    ),
    Suggestion(
      text="Propagation distance autofocus can be applied as a preprocessing step",
      priority="low",
      expected_gain_db=0.5
    )
  ],
  overall_verdict="excellent",       # P >= 0.80
  probability_of_success=0.861,
  explanation="System is well-configured. Solver choice is the marginal bottleneck; "
              "PhaseNet can improve over classical angular spectrum method."
)
```

---

## 7. AgentNegotiator --- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="excellent" | No veto |
| Severe mismatch without correction | severity=0.070 < 0.7 | No veto |
| All marginal | All excellent | No veto |
| Joint probability floor | P=0.861 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.93    # recoverability_score
P_mismatch       = 1.0 - 0.070 * 0.7 = 0.951

P_joint = 0.95 * 0.93 * 0.951 = 0.840
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],               # No vetoes
  proceed=True,
  probability_of_success=0.840
)
```

---

## 8. PreFlightReportBuilder --- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 512 * 512 = 262,144
dim_factor   = total_pixels / (256 * 256) = 4.0
solver_complexity = 0.5  # Angular spectrum (single FFT + propagation)
cr_factor    = max(1.0, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 4.0 * 0.5 * 0.125 = 0.5 seconds
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="holography", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=0.5,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=[
    "hologram (2D intensity image, TIFF or .npy)",
    "wavelength_nm (float)",
    "pixel_size_um (float)",
    "propagation_distance_um (float)"
  ]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# Holography forward model:
#   I = |U_obj * exp(i*phi) + U_ref * exp(i*2*pi*f_c*r)|^2
#
# Expanding:
#   I = |U_obj|^2 + |U_ref|^2 + U_obj*conj(U_ref)*exp(-i*2pi*f_c*r)
#                              + conj(U_obj)*U_ref*exp(i*2pi*f_c*r)
#       \___ DC ___/   \___ twin image ___/     \___ +1 order ___/
#
# Off-axis filtering isolates the +1 order in Fourier space
#
# Parameters:
#   wavelength_nm: 633
#   pixel_size_um: 4.65
#   prop_dist_um:  50
#   carrier_freq:  0.2 cycles/pixel (from tilt_angle)
#
# Input:  U = (512, 512) complex object wave
# Output: I = (512, 512) real hologram intensity

class HolographyOperator(PhysicsOperator):
    def forward(self, U_obj):
        """I = |U_obj + U_ref * exp(i*k*sin(theta)*r)|^2"""
        # Propagate object field to detector plane
        U_det = angular_spectrum_propagate(
            U_obj, wavelength, pixel_size, prop_dist
        )
        # Add reference beam
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        U_ref = np.exp(1j * 2 * np.pi * carrier_freq * xx)
        # Record intensity
        total_field = U_det + U_ref
        return np.abs(total_field)**2

    def adjoint(self, I):
        """Pseudo-adjoint: Fourier filtering + back-propagation"""
        # 1. Fourier transform of hologram
        I_ft = np.fft.fftshift(np.fft.fft2(I))
        # 2. Isolate +1 order (bandpass around carrier frequency)
        mask = make_circular_mask(center=carrier_freq, radius=NA_bandwidth)
        U_filtered = I_ft * mask
        # 3. Demodulate (shift to baseband)
        U_baseband = np.fft.ifft2(np.fft.ifftshift(U_filtered))
        # 4. Back-propagate to sample plane
        U_sample = angular_spectrum_propagate(
            U_baseband, wavelength, pixel_size, -prop_dist
        )
        return U_sample

    def check_adjoint(self):
        """Nonlinear operator: gradient consistency check"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-4)
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided hologram.tif:
from PIL import Image
hologram = np.array(Image.open("hologram.tif")).astype(np.float32)
# (512, 512) intensity

# If simulating:
U_obj = amplitude * np.exp(1j * phase)           # complex object
U_det = angular_spectrum_propagate(U_obj, ...)    # propagate to detector
U_ref = np.exp(1j * 2*np.pi * fc * xx)           # reference beam
I = np.abs(U_det + U_ref)**2                     # hologram intensity
I += np.random.poisson(I)                        # Shot noise
```

### Step 9c: Reconstruction

**Algorithm 1: Angular Spectrum Propagation (classical)**

```python
from pwm_core.recon.holography_solver import angular_spectrum_propagate

# Step 1: Fourier transform the hologram
I_ft = np.fft.fftshift(np.fft.fft2(hologram))

# Step 2: Isolate +1 diffraction order (off-axis filtering)
# Carrier frequency at (0.2 cyc/px, 0) in Fourier space
mask = circular_bandpass(center_x=0.2*N, center_y=0, radius=0.15*N)
U_filtered = I_ft * mask

# Step 3: Demodulate (shift to baseband)
U_baseband = shift_spectrum_to_center(U_filtered, carrier_freq)

# Step 4: Back-propagate to sample plane
U_sample = angular_spectrum_propagate(
    np.fft.ifft2(np.fft.ifftshift(U_baseband)),
    wavelength=633e-9,           # m
    pixel_size=4.65e-6,          # m
    distance=-50e-6              # m (back-propagation)
)

amplitude_recon = np.abs(U_sample)
phase_recon = np.angle(U_sample)
# Expected PSNR (amplitude): ~42.3 dB (USAF target benchmark)
```

**Angular spectrum propagation kernel:**

```python
# H(fx, fy) = exp(i * 2*pi * d / lambda * sqrt(1 - (lambda*fx)^2 - (lambda*fy)^2))
#
# Valid when lambda * sqrt(fx^2 + fy^2) < 1 (propagating waves)
# Evanescent waves (above cutoff) are set to zero

def angular_spectrum_propagate(field, wavelength, pixel_size, distance):
    N = field.shape[0]
    fx = np.fft.fftfreq(N, d=pixel_size)
    FX, FY = np.meshgrid(fx, fx)

    # Transfer function
    arg = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
    H = np.exp(1j * 2 * np.pi * distance / wavelength * np.sqrt(np.maximum(arg, 0)))
    H[arg < 0] = 0  # evanescent suppression

    # Propagate
    F_field = np.fft.fft2(field)
    return np.fft.ifft2(F_field * H)
```

**Algorithm 2: PhaseNet (deep learning)**

```python
from pwm_core.recon.phasenet import phasenet_infer

amplitude_hat, phase_hat = phasenet_infer(
    hologram=hologram,               # (512, 512)
    wavelength_nm=633,
    pixel_size_um=4.65,
    model_path="weights/phasenet_2M.pth",
    device="cuda"
)
# Expected PSNR (amplitude): ~45.6 dB
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| Angular Spectrum | Traditional | 42.3 dB | No | `angular_spectrum_propagate(...)` |
| PhaseNet | Deep Learning | 45.6 dB | Yes | `phasenet_infer(hologram, ...)` |
| Angular Spectrum (noisy) | Traditional | 36.8 dB | No | `angular_spectrum_propagate(...)` |

### Step 9d: Metrics

```python
# Amplitude PSNR
amp_recon = np.abs(U_sample)
amp_true = np.abs(U_true)
psnr_amp = 10 * np.log10(amp_true.max()**2 / np.mean((amp_recon - amp_true)**2))

# Phase RMSE (radians) -- holography-specific (quantitative phase imaging)
phase_recon = np.angle(U_sample)
phase_true = np.angle(U_true)
# Remove 2*pi wrapping ambiguity
phase_diff = np.angle(np.exp(1j * (phase_recon - phase_true)))
phase_rmse = np.sqrt(np.mean(phase_diff**2))

# SSIM on amplitude
ssim_amp = structural_similarity(amp_recon, amp_true)

# Phase unwrapping quality (for thick samples)
# Counts residues in the reconstructed phase map
n_residues = count_phase_residues(phase_recon)
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
|   +-- hologram.npy       # Recorded hologram (512, 512) float + SHA256
|   +-- U_sample.npy       # Reconstructed complex field (512, 512) + SHA256
|   +-- amplitude.npy      # |U_sample| (512, 512) float + SHA256
|   +-- phase.npy          # angle(U_sample) (512, 512) float + SHA256
|   +-- U_true.npy         # Ground truth (if available) + SHA256
+-- metrics.json           # PSNR (amp), phase RMSE, SSIM, n_residues
+-- operator.json          # Operator params (wavelength, pixel_size, prop_dist)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (angular spectrum -> PhaseNet) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Holography-specific:** Interferometric SNR model, off-axis Fourier filtering, complex-valued reconstruction, quantitative phase imaging metrics, evanescent wave suppression.
