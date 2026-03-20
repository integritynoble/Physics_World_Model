# FLIM Working Process

## End-to-End Pipeline for Fluorescence Lifetime Imaging Microscopy

This document traces a complete FLIM experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct fluorescence lifetime maps from this TCSPC dataset.
 Data: tcspc_data.npy, IRF: irf.npy, 256 time bins, excitation 488 nm."
```

---

## 2. PlanAgent — Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "tcspc_data.npy" detected
#   operator_type=OperatorType.nonlinear_operator,
#   files=["tcspc_data.npy", "irf.npy"],
#   params={"n_time_bins": 256, "excitation_wavelength_nm": 488}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> flim entry
flim:
  keywords: [FLIM, fluorescence_lifetime, TCSPC, phasor, decay_fitting]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="flim",
#   confidence=0.95,
#   reasoning="Matched keywords: FLIM, TCSPC, fluorescence_lifetime"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the FLIM registry entry:

```python
system = plan_agent.build_imaging_system("flim")
# ImagingSystem(
#   modality_key="flim",
#   display_name="Fluorescence Lifetime Imaging",
#   signal_dims={"x": [256, 256, 2], "y": [256, 256, 256]},
#   forward_model_type=ForwardModelType.nonlinear_operator,
#   elements=[...5 elements...],
#   default_solver="phasor"
# )
```

**FLIM Element Chain (5 elements):**

```
Pulsed Laser -----> Dichroic Mirror -----> Objective Lens (60x/1.4 NA) -----> Emission Filter -----> TCSPC Detector (SPAD)
  throughput=1.0      throughput=0.95       throughput=0.75                     throughput=0.88         throughput=0.20
  noise: none         noise: none           noise: aberration                   noise: none             noise: shot+thermal
  70 ps pulses        LP edge 500 nm        oil immersion, PSF sigma=1px       BP 530/40 nm            50 ps resolution
  80 MHz rep rate                                                                                       256 time bins
```

**Cumulative throughput:** `0.95 x 0.75 x 0.88 x 0.20 = 0.1254`

**Forward model equation:**
```
y(t) = IRF(t) * [sum_i a_i * exp(-t / tau_i)] + background
```

where `*` denotes convolution, `a_i` are amplitudes, and `tau_i` are lifetime components.

---

## 3. PhotonAgent — SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  flim:
    model_id: "microscopy_fluorescence"
    parameters:
      power_w: 0.0001
      wavelength_nm: 488
      na: 1.4
      n_medium: 1.515
      qe: 0.45
      exposure_s: 60.0
  ```

### Computation

```python
# 1. Photon energy
E_photon = h * c / wavelength_nm
#        = 6.626e-34 * 3e8 / (488e-9)
#        = 4.07e-19 J

# 2. Collection solid angle (high-NA oil immersion)
solid_angle = (na / n_medium)^2 / (4 * pi)
#           = (1.4 / 1.515)^2 / (4 * pi)
#           = 0.8531^2 / 12.566
#           = 0.0579

# 3. Raw photon count
N_raw = power_w * qe * solid_angle * exposure_s / E_photon
#     = 0.0001 * 0.45 * 0.0579 * 60.0 / 4.07e-19
#     = 1.564e-4 / 4.07e-19
#     = 3.84e14 photons (over full acquisition)

# 4. Per-pixel photon count (256 x 256 = 65536 pixels)
N_per_pixel = N_raw / 65536 = 5.86e9 photons/pixel (total)

# 5. Apply cumulative throughput
N_effective = N_per_pixel * 0.1254 = 7.35e8 photons/pixel

# NOTE: In TCSPC, photons are distributed across 256 time bins
N_per_bin = N_effective / 256 = 2.87e6 photons/pixel/bin

# 6. Noise variances
shot_var     = N_per_bin                       # Poisson
dark_var     = dark_count_rate * exposure_s / n_pixels
#            = 25 * 60.0 / 65536 = 0.023      # Negligible
thermal_var  = 0.0                             # Cooled SPAD
total_var    = shot_var + dark_var

# 7. SNR (per time bin)
SNR_bin = N_per_bin / sqrt(total_var) = sqrt(N_per_bin) = 1694
SNR_db  = 20 * log10(1694) = 64.6 dB

# 8. Effective SNR for lifetime estimation (integrates across bins)
SNR_lifetime = sqrt(N_effective) = sqrt(7.35e8) = 27114
SNR_lifetime_db = 20 * log10(27114) = 88.7 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=7.35e8,
  snr_db=88.7,
  noise_regime=NoiseRegime.photon_starved,      # TCSPC is always photon-limited
  shot_noise_sigma=27114,
  read_noise_sigma=0.0,                          # No read noise in SPAD
  total_noise_sigma=27114,
  feasible=True,
  quality_tier="excellent",                      # SNR > 30 dB
  throughput_chain=[
    {"Pulsed Laser Source": 1.0},
    {"Dichroic Mirror": 0.95},
    {"Objective Lens (60x / 1.4 NA Oil)": 0.75},
    {"Emission Filter": 0.88},
    {"TCSPC Detector": 0.20}
  ],
  noise_model="poisson",
  explanation="Photon-limited regime typical of TCSPC. 60s acquisition yields "
              "~735M photons/pixel total. Sufficient for multi-exponential fitting."
)
```

---

## 4. MismatchAgent — Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"flim"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  flim:
    parameters:
      irf_shift:
        range: [-0.5, 0.5]
        typical_error: 0.05
        unit: "ns"
        description: "IRF temporal shift from electronics timing jitter"
      irf_width:
        range: [0.02, 0.5]
        typical_error: 0.03
        unit: "ns"
        description: "IRF FWHM error from detector transit time spread"
      background_offset:
        range: [0.0, 0.10]
        typical_error: 0.02
        unit: "normalized"
        description: "Additive background from dark counts and ambient light"
    severity_weights:
      irf_shift: 0.40
      irf_width: 0.35
      background_offset: 0.25
    correction_method: "grid_search"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.40 * |0.05| / 1.0      # irf_shift:        0.020
  + 0.35 * |0.03| / 0.48     # irf_width:         0.022
  + 0.25 * |0.02| / 0.10     # background_offset: 0.050
S = 0.092  # Low severity (well-characterized system)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 0.92 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="flim",
  mismatch_family="grid_search",
  parameters={
    "irf_shift":        {"typical_error": 0.05, "range": [-0.5, 0.5], "weight": 0.40},
    "irf_width":        {"typical_error": 0.03, "range": [0.02, 0.5], "weight": 0.35},
    "background_offset": {"typical_error": 0.02, "range": [0.0, 0.10], "weight": 0.25}
  },
  severity_score=0.092,
  correction_method="grid_search",
  expected_improvement_db=0.92,
  explanation="Low mismatch severity. IRF well-characterized under typical conditions."
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
  flim:
    signal_prior_class: "deep_prior"
    entries:
      - {cr: 0.008, noise: "photon_starved", solver: "phasor_flim",
         recoverability: 0.58, expected_psnr_db: 25.3,
         provenance: {dataset_id: "flimlab_hela_nadh_2023", ...}}
      - {cr: 0.008, noise: "photon_starved", solver: "deep_flim",
         recoverability: 0.75, expected_psnr_db: 30.1, ...}
      - {cr: 0.008, noise: "shot_limited", solver: "mle_flim",
         recoverability: 0.82, expected_psnr_db: 32.4, ...}
  ```

### Computation

```python
# 1. Compression ratio
#    x = (256, 256, 2)  -> 2 parameters (tau, amplitude) per pixel
#    y = (256, 256, 256) -> 256 time bins per pixel
CR = prod(x_shape) / prod(y_shape)
   = (256 * 256 * 2) / (256 * 256 * 256)
   = 131072 / 16777216
   = 0.0078  ~  0.008

# 2. Operator diversity (decay model richness)
#    FLIM uses a parametric model: each pixel has tau and amplitude
#    The forward operator (IRF convolution + exponential) is well-conditioned
#    per-pixel but globally sparse in parameter space
diversity = 0.6  # Moderate — parametric model constrains solutions

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.625

# 4. Calibration table lookup
#    Match: noise="photon_starved", solver="phasor_flim", cr=0.008
#    -> recoverability=0.58, expected_psnr=25.3 dB
#    But with 735M photons/pixel, regime is actually "shot_limited"
#    -> Match: noise="shot_limited", solver="mle_flim", cr=0.008
#    -> recoverability=0.82, expected_psnr=32.4 dB

# 5. Best solver selection
#    mle_flim: 32.4 dB > deep_flim: 30.1 dB > phasor: 25.3 dB
#    -> recommended: "mle_flim" (best quality)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.008,
  noise_regime=NoiseRegime.photon_starved,
  signal_prior_class=SignalPriorClass.deep_prior,
  operator_diversity_score=0.6,
  condition_number_proxy=0.625,
  recoverability_score=0.82,
  recoverability_confidence=1.0,
  expected_psnr_db=32.4,
  expected_psnr_uncertainty_db=1.0,
  recommended_solver_family="mle_flim",
  verdict="good",                        # score >= 0.70
  calibration_table_entry={...},
  explanation="Good recoverability. MLE fitting expected 32.4 dB on HeLa NADH benchmark. "
              "High photon count enables reliable multi-exponential fitting."
)
```

---

## 6. AnalysisAgent — Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(88.7 / 40, 1.0)   = 0.0     # Excellent SNR
mismatch_score    = 0.092                        = 0.092   # Low mismatch
compression_score = 1 - 0.82                     = 0.18    # Good recoverability
solver_score      = 0.15                         = 0.15    # Phasor is fast but less accurate

# Primary bottleneck
primary = "compression"  # max(0.0, 0.092, 0.18, 0.15) = compression

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.092*0.5) * (1 - 0.18*0.5) * (1 - 0.15*0.5)
  = 1.0 * 0.954 * 0.91 * 0.925
  = 0.803
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="compression",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.092, compression=0.18, solver=0.15
  ),
  suggestions=[
    Suggestion(
      text="Use MLE fitting instead of phasor for +7.1 dB improvement",
      priority="high",
      expected_gain_db=7.1
    ),
    Suggestion(
      text="Deep FLIM network offers balance of speed and quality (+4.8 dB over phasor)",
      priority="medium",
      expected_gain_db=4.8
    )
  ],
  overall_verdict="good",               # P >= 0.70
  probability_of_success=0.803,
  explanation="System is well-configured. Extreme compression ratio (0.008) is "
              "inherent to FLIM (2 parameters from 256 bins) but well-handled by MLE."
)
```

---

## 7. AgentNegotiator — Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="good" | No veto |
| Severe mismatch without correction | severity=0.092 < 0.7 | No veto |
| All marginal | All good/excellent | No veto |
| Joint probability floor | P=0.803 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.82    # recoverability_score
P_mismatch       = 1.0 - 0.092 * 0.7 = 0.936

P_joint = 0.95 * 0.82 * 0.936 = 0.729
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],              # No vetoes
  proceed=True,
  probability_of_success=0.729
)
```

---

## 8. PreFlightReportBuilder — Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 256 * 256 = 65536
n_time_bins  = 256
dim_factor   = total_pixels * n_time_bins / (256 * 256) = 256.0
solver_complexity = 1.5   # MLE fitting (iterative per-pixel)
cr_factor    = max(0.008, 1.0) / 8.0 = 0.125

runtime_s = 2.0 * 256.0 * 1.5 * 0.125 = 96.0 seconds
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="flim", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=96.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=[
    "TCSPC histogram data (H x W x T time bins)",
    "Instrument Response Function (1D array, T bins)",
    "Time axis calibration (ns per bin)"
  ]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# FLIM forward model: y(t) = IRF(t) * [sum_i a_i * exp(-t/tau_i)] + bg
#
# Parameters:
#   irf:          (T,) instrument response function     [loaded from irf.npy]
#   time_axis:    (T,) time values in nanoseconds
#   n_time_bins:  256
#   time_range:   12.5 ns (at 80 MHz rep rate)
#   time_res:     50 ps per bin
#
# Input:  x = (256, 256, 2) lifetime parameters [tau, amplitude] per pixel
# Output: y = (256, 256, 256) TCSPC histograms

class FLIMOperator(PhysicsOperator):
    def forward(self, x):
        """y(i,j,t) = IRF(t) * [a(i,j) * exp(-t/tau(i,j))]"""
        tau = x[:, :, 0]          # (H, W) lifetime map
        amp = x[:, :, 1]          # (H, W) amplitude map
        t = self.time_axis        # (T,)

        # Generate exponential decays
        decay = amp[:,:,None] * np.exp(-t[None,None,:] / tau[:,:,None])

        # Convolve each pixel with IRF
        y = np.zeros_like(decay)
        for i in range(H):
            for j in range(W):
                y[i,j] = fftconvolve(decay[i,j], self.irf, mode='full')[:T]
        return y

    def adjoint(self, y):
        """Approximate adjoint via matched filtering"""
        # Not a standard linear adjoint — FLIM is nonlinear
        # Use correlation with IRF as proxy
        x_hat = np.zeros((H, W, 2))
        for i in range(H):
            for j in range(W):
                corr = np.correlate(y[i,j], self.irf, mode='full')
                x_hat[i,j,0] = estimate_tau_from_corr(corr)
                x_hat[i,j,1] = np.sum(y[i,j])
        return x_hat
```

### Step 9b: Forward Simulation (or Load Measurement)

```python
# If user provided tcspc_data.npy:
y = np.load("tcspc_data.npy")     # (256, 256, 256)
irf = np.load("irf.npy")          # (256,)

# If simulating:
# Ground truth: 3 regions with different lifetimes
tau_true = np.ones((64, 64)) * 2.5      # Background: 2.5 ns
tau_true[region1] = 1.0                   # Short lifetime (bound NADH)
tau_true[region2] = 4.0                   # Long lifetime (free NADH)

# Generate synthetic TCSPC data
time_axis = np.linspace(0, 10, 64)        # 64 time bins
irf_sigma = 0.2                            # 200 ps IRF width
irf = np.exp(-0.5 * (time_axis / irf_sigma)**2)
decay = amp * np.exp(-t / tau)
y = convolve(decay, irf) + noise          # Poisson statistics
```

### Step 9c: Reconstruction with Phasor Analysis

```python
from pwm_core.recon.flim_solver import phasor_recon, mle_fit_recon

# Algorithm 1: Phasor Analysis (fast, model-free)
tau_phasor, amp_phasor = phasor_recon(
    decay_data=y,               # (H, W, T) TCSPC histograms
    time_axis=time_axis,        # (T,) ns
    harmonic=1
)
# Phasor coordinates: G = Re(FFT), S = Im(FFT) at angular freq omega
# tau = S / (omega * G) for single exponential
# Expected PSNR: ~25.0 dB on benchmark (tau map, max_val=10)

# Algorithm 2: MLE Fitting (accurate, iterative)
tau_mle, amp_mle = mle_fit_recon(
    decay_data=y,
    time_axis=time_axis,
    irf=irf,                    # Measured IRF for deconvolution
    n_iters=50                  # Levenberg-Marquardt iterations
)
# Model: I(t) = a * exp(-t/tau) convolved with IRF
# Per-pixel Gauss-Newton optimization
# Expected PSNR: ~32.4 dB on benchmark
```

**Alternative solvers:**

| Solver | Type | PSNR | GPU | Command |
|--------|------|------|-----|---------|
| Phasor | Traditional | 25.3 dB | No | `phasor_recon(y, time_axis)` |
| MLE Fit | Traditional | 32.4 dB | No | `mle_fit_recon(y, time_axis, irf=irf)` |
| Deep FLIM | Deep Learning | 30.1 dB | Yes | `deep_flim_recon(y, time_axis)` |

### Step 9d: Metrics

```python
# Lifetime PSNR (using max_val = 10 ns as dynamic range)
psnr_tau = 10 * log10(10.0^2 / mse(tau_hat, tau_true))
# ~25.0 dB (phasor), ~32.4 dB (MLE)

# Lifetime accuracy (mean absolute error in ns)
mae_tau = mean(|tau_hat - tau_true|)

# Phasor plot metrics
# G^2 + S^2 should lie on or inside the universal semicircle
# for single-exponential decays: G^2 + S^2 = G (semicircle)
semicircle_residual = mean(|(G^2 + S^2) - G|)

# Multi-component fitting quality (if applicable)
chi_squared = sum((y - model)^2 / model) / (n_bins - n_params)
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
|   +-- y.npy              # TCSPC data (256, 256, 256) + SHA256 hash
|   +-- tau_hat.npy         # Lifetime map (256, 256) + SHA256 hash
|   +-- amp_hat.npy         # Amplitude map (256, 256) + SHA256 hash
|   +-- tau_true.npy        # Ground truth lifetimes (if available) + SHA256 hash
+-- metrics.json           # PSNR(tau), MAE(tau), chi-squared per pixel
+-- operator.json          # Operator parameters (IRF hash, time_axis, n_bins)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

---

# Part II: Real Experiment Scenario

The previous sections (1-9) showed an idealized FLIM pipeline with high photon counts (735M photons/pixel, 88.7 dB SNR). In practice, real FLIM systems are severely photon-starved because TCSPC has a count-rate ceiling of ~10% of the laser repetition rate to avoid pile-up artifacts. Live-cell imaging further limits photon budgets to avoid phototoxicity.

---

## Real Experiment: User Prompt

```
"Live-cell FLIM of NADH autofluorescence. Short acquisition to minimize
 phototoxicity. Data: hela_tcspc.npy, IRF: measured_irf.npy, 256 bins.
 Need to distinguish bound vs free NADH (tau ~0.4 ns vs ~3.2 ns)."
```

**Key difference:** Live-cell imaging with low photon budget and two-component decay model.

---

## R1. PlanAgent — Intent Parsing

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,
#   operator_type=OperatorType.nonlinear_operator,
#   files=["hela_tcspc.npy", "measured_irf.npy"],
#   params={"n_time_bins": 256, "n_components": 2,
#           "target_lifetimes_ns": [0.4, 3.2]}
# )
```

---

## R2. PhotonAgent — Photon-Starved Live Cell

### Real acquisition parameters

```yaml
# Real live-cell: minimal laser power, short dwell
flim_live:
  power_w: 0.00001        # 10 uW (10x lower to avoid phototoxicity)
  wavelength_nm: 405      # 405 nm diode (NADH excitation)
  na: 1.4
  n_medium: 1.515
  qe: 0.20                # SPAD QE at 450 nm emission
  exposure_s: 10.0         # 10s total scan (256x256 pixels)
```

### Computation

```python
# Per-pixel photon count
N_raw = 0.00001 * 0.20 * 0.0579 * 10.0 / 4.91e-19 = 2.36e10
N_per_pixel = 2.36e10 / 65536 = 3.60e5
N_effective = 3.60e5 * 0.1254 = 4.51e4 photons/pixel

# TCSPC pile-up limit: ~10% of rep rate * dwell_time
# Dwell per pixel = 10.0 / 65536 = 0.153 ms
# Max count rate = 0.1 * 80e6 = 8e6 counts/s
# Max photons = 8e6 * 0.153e-3 = 1221 photons/pixel
# ACTUAL photon count limited by pile-up: ~100-500 photons/pixel

N_actual = 200  # Realistic for live-cell FLIM

# Per-bin count
N_per_bin = 200 / 256 = 0.78 photons/bin  # Severely sparse!

# SNR
SNR = sqrt(N_actual) = sqrt(200) = 14.1
SNR_db = 20 * log10(14.1) = 23.0 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=200,
  snr_db=23.0,
  noise_regime=NoiseRegime.photon_starved,    # <1 photon per bin!
  shot_noise_sigma=14.1,
  read_noise_sigma=0.0,
  total_noise_sigma=14.1,
  feasible=True,                               # Marginal but feasible
  quality_tier="moderate",                     # 20 < SNR < 30 dB
  noise_model="poisson",
  explanation="Severely photon-starved (~200 photons/pixel). Typical for "
              "live-cell NADH FLIM. Phasor analysis robust; MLE may be noisy."
)
```

---

## R3. MismatchAgent — IRF Drift in Live-Cell Setup

```python
# IRF may shift during long acquisitions from thermal drift
psi_true = {
    "irf_shift":        +0.15,   # 150 ps drift (3x typical)
    "irf_width":        +0.08,   # IRF broadened from detector aging
    "background_offset": 0.05,   # Ambient light leakage (incubator LEDs)
}

# Severity
S = 0.40 * |0.15| / 1.0     # irf_shift:        0.060
  + 0.35 * |0.08| / 0.48    # irf_width:         0.058
  + 0.25 * |0.05| / 0.10    # background_offset: 0.125
S = 0.243  # Moderate severity

improvement_db = clip(10 * 0.243, 0, 20) = 2.43 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  severity_score=0.243,
  correction_method="grid_search",
  expected_improvement_db=2.43,
  explanation="Moderate mismatch. Background offset from ambient light is the "
              "primary error source. IRF re-measurement recommended."
)
```

---

## R4. RecoverabilityAgent — Photon-Starved Regime

```python
# Calibration table: noise="photon_starved", solver="phasor_flim"
# -> recoverability=0.58, expected_psnr=25.3 dB
```

```python
RecoverabilityReport(
  recoverability_score=0.58,
  expected_psnr_db=25.3,
  verdict="marginal",
  explanation="Low photon count limits lifetime accuracy. Two-component fitting "
              "unreliable below ~500 photons/pixel."
)
```

---

## R5. AnalysisAgent — Photon Budget is the Bottleneck

```python
photon_score      = 1 - min(23.0 / 40, 1.0)   = 0.425
mismatch_score    = 0.243
compression_score = 1 - 0.58                     = 0.42
solver_score      = 0.15

primary = "photon"  # max(0.425, 0.243, 0.42, 0.15)

P = (1-0.425*0.5) * (1-0.243*0.5) * (1-0.42*0.5) * (1-0.15*0.5)
  = 0.788 * 0.879 * 0.79 * 0.925
  = 0.506
```

```python
SystemAnalysis(
  primary_bottleneck="photon",
  probability_of_success=0.506,
  overall_verdict="marginal",
  suggestions=[
    Suggestion(text="Increase acquisition time to 60s for 6x more photons", priority="high"),
    Suggestion(text="Use phasor analysis (robust to low photon counts)", priority="high"),
    Suggestion(text="Spatial binning 2x2 for 4x photon gain", priority="medium"),
  ]
)
```

---

## R6. AgentNegotiator — Conditional Proceed

```python
P_joint = 0.95 * 0.58 * (1 - 0.243*0.7) = 0.95 * 0.58 * 0.830 = 0.457

NegotiationResult(
  vetoes=[],
  proceed=True,         # P > 0.15
  probability_of_success=0.457
)
```

---

## R7. PreFlightReportBuilder

```python
PreFlightReport(
  estimated_runtime_s=45.0,
  proceed_recommended=True,
  warnings=[
    "Photon-starved regime (~200 photons/pixel) — lifetime estimates will be noisy",
    "Two-component fitting unreliable; phasor analysis recommended for initial assessment",
    "Background offset 5% — consider dark-frame subtraction"
  ]
)
```

---

## Side-by-Side Comparison: Ideal vs Real

| Metric | Ideal Lab | Real Live-Cell |
|--------|-----------|----------------|
| **Photon Agent** | | |
| N_effective | 7.35e8 | 200 |
| SNR | 88.7 dB | 23.0 dB |
| Quality tier | excellent | moderate |
| Noise regime | photon_starved | photon_starved |
| **Mismatch Agent** | | |
| Severity | 0.092 (low) | 0.243 (moderate) |
| Dominant error | none | background 5% |
| Expected gain | +0.92 dB | +2.43 dB |
| **Recoverability Agent** | | |
| Score | 0.82 (good) | 0.58 (marginal) |
| Expected PSNR | 32.4 dB | 25.3 dB |
| Verdict | good | marginal |
| **Analysis Agent** | | |
| Primary bottleneck | compression | **photon** |
| P(success) | 0.803 | 0.506 |
| **Negotiator** | | |
| P_joint | 0.729 | 0.457 |
| **PreFlight** | | |
| Runtime | 96s | 45s |
| Warnings | 0 | 3 |

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (phasor -> MLE -> deep FLIM) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **Adaptive:** FLIM-specific awareness of photon-starved regime and TCSPC pile-up limits.
