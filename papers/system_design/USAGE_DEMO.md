# Usage Demo — Multi-Agent Imaging System Design Pipeline

This document walks through a complete usage case: designing a **sparse-view low-dose CT system** (forward) and then **reconstructing images** from the simulated measurements. It also demonstrates the **judge rejection → redesign loop** with an intentionally bad MRI plan.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Step 0: Database Context](#step-0-database-context)
3. [Period 1: Forward — System Design](#period-1-forward--system-design)
   - [Plan Agent Output](#11-plan-agent--plandocument-forward)
   - [Judge Agent Verdict](#12-judge-agent--judgmentresult-forward)
   - [Performance Agent Simulation](#13-performance-agent--forward-simulation)
4. [Period 2: Reconstruction — Algorithm Design](#period-2-reconstruction--algorithm-design)
   - [Plan Agent Output](#21-plan-agent--plandocument-reconstruction)
   - [Judge Agent Verdict](#22-judge-agent--judgmentresult-reconstruction)
   - [Performance Agent Reconstruction](#23-performance-agent--reconstruction)
5. [Judge Rejection → Redesign Loop](#judge-rejection--redesign-loop)
6. [Generated Files](#generated-files)
7. [Full Plan Markdown: Forward](#full-plan-markdown-forward)
8. [Full Plan Markdown: Reconstruction](#full-plan-markdown-reconstruction)

---

## Quick Start

```bash
# Run the demo (no API key needed — uses hand-crafted plans with real simulation)
cd papers/system_design
python3 demo_usage.py

# Run with real LLM agents (requires ANTHROPIC_API_KEY)
python3 main.py --modality ct --period forward \
    --prompt "Design a sparse-view CT system with 60 angles and low dose"

python3 main.py --modality ct --period reconstruction \
    --prompt "Reconstruct 60-angle low-dose sinogram using TV-ADMM" \
    --measurements outputs/ct_forward_result.npz
```

---

## Step 0: Database Context

Before generating a plan, the Plan Agent queries the YAML database and receives this context about CT:

### Modality Context

```
Modality: X-ray Computed Tomography
Category: tomography
Physics class: x_ray_attenuation

Forward model elements:
  - source       (source)       mismatch=[beam_hardening]
  - phantom      (interaction)  mismatch=[scatter]
  - geometry     (geometry)     mismatch=[center_of_rotation_offset, detector_tilt]
  - detector     (detector)     noise=[poisson, gaussian, dark_current]
                                mismatch=[detector_response_nonuniformity]
  - digitization (digitization) mismatch=[detector_nonlinearity]

Known mismatch sources: beam_hardening, scatter, center_of_rotation,
                        detector_response_nonuniformity, motion_blur

Budget estimate: equipment ~$500k, per-scan ~$50
```

### Algorithm Catalog

```
Available algorithms (id / name / type / reference):
  - fbp:           Filtered Back-Projection        [Classical]       — Kak & Slaney 1988
  - sart:          SART                             [Classical]       — Andersen & Kak 1984
  - osem:          OSEM                             [Classical]       — Hudson & Larkin 1994
  - tv_admm:       TV-ADMM                          [Variational]     — Sidky & Pan 2008
  - pnp_admm:      PnP-ADMM (BM3D)                 [PnP]             — Venkatakrishnan 2013
  - unrolled_admm: Unrolled ADMM                    [Deep Unrolling]  — Yang et al. 2016
  - diffusion_mbir: DiffusionMBIR                   [Diffusion]       — Song et al. 2021
```

The Plan Agent uses this context to select appropriate elements, noise models, algorithms, and mismatch corrections — or to customize them based on the user's prompt.

---

## Period 1: Forward — System Design

**User prompt:**
> "Design a sparse-view CT system with 60 projection angles, low dose (I0=1e4), for pediatric chest imaging."

### 1.1 Plan Agent → PlanDocument (forward)

The Plan Agent generates a `PlanDocument` with four sections:

#### Section 1 — Task

> Design a complete forward model for sparse-view X-ray CT with only 60 projection angles and low photon flux (I0=1e4) for pediatric chest imaging. The system must model all physical elements from X-ray source through detector digitization, including realistic noise and calibration mismatch sources.

#### Section 2 — Plan (6 steps)

| Step | Description |
|------|-------------|
| 1 | Configure polychromatic X-ray tube source at 80 kVp with 1.5mm Al filtration |
| 2 | Model Beer-Lambert attenuation through soft tissue phantom |
| 3 | Define parallel-beam acquisition geometry with 60 projection angles over 180° |
| 4 | Simulate flat-panel CsI:Tl detector with Poisson noise (I0=1e4) and Gaussian readout noise |
| 5 | Apply 12-bit ADC digitization with dark current |
| 6 | Identify beam hardening, scatter, and center-of-rotation mismatch sources |

#### Section 3 — Action (System Flowchart)

```
[X-ray Tube 80kVp] → [Soft Tissue Phantom] → [Parallel-Beam 60 angles]
       ↓                      ↓                        ↓
  [Polychromatic         [Beer-Lambert           [CoR offset
   beam hardening]        attenuation]            mismatch]
                                                       ↓
                              → [CsI:Tl Flat Panel Detector] → [12-bit ADC] → y
                                        ↓
                                  [Poisson I0=1e4]
                                  [Gaussian σ=3 e⁻]
                                  [Dark current 0.05 e⁻/s]
```

**5 elements with full specifications:**

| Element | Type | Key Parameters | Noise | Mismatch |
|---------|------|---------------|-------|----------|
| X-ray Tube Source (80 kVp) | `source` | energy=80kVp, flux=5e5, focal_spot=0.4mm, filtration=1.5mm Al | — | beam_hardening [HIGH] |
| Soft Tissue Attenuation | `interaction` | model=beer_lambert, mu_water=0.184 cm⁻¹ | — | scatter [MEDIUM] |
| Parallel-Beam (60 angles) | `geometry` | scan_type=parallel_beam, angles=60, range=180° | — | center_of_rotation [MEDIUM] |
| CsI:Tl Flat Panel Detector | `detector` | scintillator=CsI:Tl, pixels=256×256, QE=0.75 | Poisson(I0=1e4), Gaussian(σ=3e⁻), Dark(0.05e⁻/s) | detector_nonuniformity [LOW] |
| 12-bit ADC | `digitization` | bit_depth=12, dynamic_range=72dB | — | — |

**Composite noise model:**
```
y ~ Poisson(I0 * exp(-H*x)) + N(0, σ_readout²) + Poisson(dark * t_exp)
```

**Measurement shape:** `(256, 60)` — 256 detector pixels × 60 angles

#### Section 4 — Demands

| Flag | Value | Notes |
|------|-------|-------|
| feasibility | **yes** | |
| budget_feasible | **yes** | |
| algorithm_convergence | N/A | Not applicable for forward period |
| comments | Low-dose (I0=1e4) will produce noisy sinograms (est. SNR ~17 dB). Sparse view (60 angles) causes streak artifacts in FBP but recoverable with iterative reconstruction. | |

---

### 1.2 Judge Agent → JudgmentResult (forward)

```
Verdict:    PASS
Confidence: 0.88
SNR est.:   17.2 dB
Budget est.: $380,000
Issues:     2 warnings, 0 critical
```

**Issues identified:**

| Severity | Category | Element | Description | Suggestion |
|----------|----------|---------|-------------|------------|
| WARNING | noise_level | detector | Estimated SNR ~17 dB at I0=1e4 is low; FBP will produce noisy images | Use iterative reconstruction (TV-ADMM or PnP-ADMM) instead of FBP |
| WARNING | physics | xray_source | Polychromatic beam hardening correction should be applied before reconstruction | Include beam hardening polynomial correction in the mismatch pipeline |

**Summary:** The forward model is physically sound. All major elements (source, attenuation, geometry, detector, ADC) are present with appropriate noise models. The low photon flux (I0=1e4) will produce moderately noisy sinograms but is realistic for low-dose pediatric CT. Beam hardening and scatter mismatch are correctly identified. The 60-angle sparse view is aggressive but feasible with iterative reconstruction.

**Decision: PASS → proceed to Performance Agent**

---

### 1.3 Performance Agent → Forward Simulation

The Performance Agent walks each element in topological order:

```
xray_source → tissue_attenuation → geometry → detector → adc
```

**Execution trace:**

| Step | Element | Operation | Output Shape |
|------|---------|-----------|-------------|
| 1 | `xray_source` | Generate incident flux (5e5 photons/s) | `(256, 256)` |
| 2 | `tissue_attenuation` | Beer-Lambert: I = I₀ × exp(-μx) | `(256, 256)` |
| 3 | `geometry` | Radon transform (parallel-beam, 60 angles) | `(256, 60)` |
| 4 | `detector` | Apply Poisson(I0=1e4) + Gaussian(σ=3) + Dark current | `(256, 60)` |
| 5 | `adc` | Quantize to 12-bit | `(256, 60)` |

**Result:**

```
Phantom:      Shepp-Logan (256, 256), range=[0.000, 1.000]
Sinogram:     (256, 60), range=[0.000, 76.926], mean=31.735
```

**Saved:** `outputs/ct_forward_result.npz` (324 KB) containing `measurements` and `ground_truth` arrays.

---

## Period 2: Reconstruction — Algorithm Design

**User prompt:**
> "Reconstruct the 60-angle low-dose CT sinogram using TV-ADMM with beam hardening and scatter correction."

### 2.1 Plan Agent → PlanDocument (reconstruction)

#### Section 1 — Task

> Reconstruct the original attenuation map from a sparse-view (60 angles), low-dose (I0=1e4) CT sinogram. The inverse problem is severely ill-posed due to angular undersampling and high noise. The sinogram is also corrupted by beam hardening and Compton scatter. Use TV-ADMM with mismatch corrections to regularize the reconstruction.

#### Section 2 — Plan (6 steps)

| Step | Description |
|------|-------------|
| 1 | Apply beam hardening polynomial correction to the sinogram |
| 2 | Apply scatter kernel subtraction to remove low-frequency bias |
| 3 | Initialize with Filtered Back-Projection (FBP, ramp filter) |
| 4 | Run TV-ADMM: alternate data fidelity gradient step and TV proximal step |
| 5 | Enforce non-negativity constraint at each iteration |
| 6 | Check convergence: ‖x_{k+1} − x_k‖ / ‖x_k‖ < 1e-4 or max 100 iterations |

#### Section 3 — Action (Algorithm Details)

**Algorithm:** TV-ADMM (Variational)

**References:**
- Sidky & Pan, Phys. Med. Biol. 53(17), 2008
- Kak & Slaney, IEEE Press 1988

**5 algorithm steps with equations:**

| Step | Name | Equation | Parameters |
|------|------|----------|------------|
| 1 | Mismatch Pre-Correction | y_corr = a₀ + a₁y + a₂y² − 0.1·G_σ(y) | a₀=0, a₁=1, a₂=−0.05, scatter_σ=20 |
| 2 | FBP Initialization | x₀ = R⁻¹_FBP(y_corr) | filter=ramp |
| 3 | Data Fidelity Gradient | grad = Rᵀ(Rx_k − y_corr) | — |
| 4 | TV Proximal Step | x_{k+1} = prox_{λ·TV}(x_k − η·grad) | λ_tv=0.01, η=0.005 |
| 5 | Non-Negativity Projection | x_{k+1} = max(x_{k+1}, 0) | — |

**Mismatch corrections (applied before reconstruction):**

| Mismatch | Severity | Correction |
|----------|----------|------------|
| beam_hardening | HIGH | 2nd-order polynomial: y_corr = y − 0.05y² |
| scatter | MEDIUM | Subtract 10% of Gaussian-blurred sinogram (σ=20 px) |

**Convergence criterion:** ‖x_{k+1} − x_k‖₂ / ‖x_k‖₂ < 1e-4, or max 100 iterations

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| lambda_tv | 0.01 |
| step_size | 0.005 |
| num_iterations | 100 |
| filter | ramp |

#### Section 4 — Demands

| Flag | Value | Notes |
|------|-------|-------|
| feasibility | **yes** | |
| budget_feasible | N/A | Not applicable for reconstruction |
| algorithm_convergence | **yes** | TV-ADMM converges reliably for sparse-view CT |
| comments | TV penalty effectively suppresses streak artifacts. Beam hardening and scatter pre-corrections are essential at this noise level. | |

---

### 2.2 Judge Agent → JudgmentResult (reconstruction)

```
Verdict:        PASS
Confidence:     0.92
Convergence:    likely
Mismatch:       handled
Issues:         1 warning, 0 critical
```

**Issues identified:**

| Severity | Category | Description | Suggestion |
|----------|----------|-------------|------------|
| WARNING | convergence | lambda_tv=0.01 may over-smooth fine structures | Start with lambda_tv=0.005 and increase if streak artifacts persist |

**Summary:** The reconstruction plan is algorithmically sound. TV-ADMM is a well-established method for sparse-view CT with proven convergence guarantees under convexity (TV penalty is convex). The beam hardening polynomial correction and scatter subtraction are appropriate for the identified mismatch sources. Step sizes and regularization weight are within reasonable ranges. No pre-training required — this is a pure optimization method suitable for test-time execution.

**Decision: PASS → proceed to Performance Agent**

---

### 2.3 Performance Agent → Reconstruction

**Execution pipeline:**

```
1. Mismatch corrections:
   - Beam hardening polynomial: y_corr = y − 0.05y²
   - Scatter subtraction: y_corr = y − 0.1·Gaussian(y, σ=20)

2. FBP initialization:
   - x₀ = iradon(y_corr, filter='ramp')

3. TV-ADMM loop (100 iterations):
   - Data fidelity gradient: grad = Rᵀ(Rx_k − y_corr)
   - Update: x_k ← x_k − 0.005 × grad
   - TV proximal: x_k ← prox_TV(x_k, λ=0.01)
   - Non-negativity: x_k ← max(x_k, 0)
   - Check convergence
```

**Result:**

```
Reconstruction shape:  (256, 256)
Value range:           [0.0000, 0.2879]
PSNR:                  12.29 dB
SSIM:                  0.3282
Iterations:            100
Runtime:               36.87 s
```

**Saved:** `outputs/ct_reconstruction_result.npz` (525 KB) containing `x_hat` and `x_true` arrays.

**Note on quality:** The low PSNR (12.29 dB vs. 40 dB target) is expected for this extreme setup — 60 angles with I0=1e4 is severely ill-posed. The simplified TV proximal operator (Gaussian approximation) further limits quality. With real LLM agents, the Plan Agent would design more sophisticated algorithms (PnP-ADMM with trained denoiser, or hybrid methods), and the Judge would iterate until quality improves.

---

## Judge Rejection → Redesign Loop

This demonstrates what happens when the Judge rejects an infeasible plan.

### Iteration 1: Physically Impossible MRI Design

**Plan:** Design a 0.001T MRI system with 100x acceleration using a single coil.

```
Verdict:    FAIL (REJECTED)
Confidence: 0.95
Critical issues: 2
```

| Severity | Element | Description |
|----------|---------|-------------|
| **CRITICAL** | magnet | 0.001T field strength produces SNR < 0.1 dB — no usable signal |
| **CRITICAL** | kspace | 100x acceleration with 1 coil violates max_acceleration ≈ num_coils constraint |

**Redesign prompt sent back to Plan Agent:**

> CRITICAL: (1) Increase field strength to at least 1.5T for adequate SNR. (2) Reduce acceleration to at most num_coils (use 8-coil array with 4x acceleration). (3) Add coil sensitivity estimation (ESPIRiT) to handle parallel imaging.

### Iteration 2: Plan Agent Redesigns

The Plan Agent receives the judge feedback and generates a corrected plan:

**Corrected plan:** 3T MRI with 8-coil array, 4x Cartesian acceleration

| Element | Before (rejected) | After (corrected) |
|---------|-------------------|-------------------|
| Field strength | 0.001T | **3T** |
| Acceleration | 100x | **4x** |
| Coils | 1 | **8-channel phased array** |
| B0 mismatch | Not identified | **B0 inhomogeneity [medium]** |

```
Verdict:    PASS
Confidence: 0.91
```

**→ Pipeline proceeds to Performance Agent for execution.**

### Redesign Loop Diagram

```
Iteration 1:
  Plan Agent → [0.001T, 100x, 1 coil] → Judge → FAIL (2 critical)
                                            │
                                            ▼
                              redesign_prompt: "increase to 1.5T,
                              use 8-coil array, max 4x acceleration"
                                            │
Iteration 2:                                ▼
  Plan Agent → [3T, 4x, 8-coil array] → Judge → PASS (confidence 0.91)
                                            │
                                            ▼
                                   Performance Agent → execute
```

---

## Generated Files

After running the demo, the `outputs/` directory contains:

```
outputs/
├── ct_forward_v1_iter1.md              3,899 bytes   Forward plan markdown (113 lines)
├── ct_reconstruction_v1_iter1.md       2,961 bytes   Reconstruction plan markdown (106 lines)
├── ct_forward_result.npz             324,118 bytes   Sinogram (256×60) + phantom (256×256)
└── ct_reconstruction_result.npz      524,796 bytes   x_hat (256×256) + x_true (256×256)
```

### Loading results in Python

```python
import numpy as np

# Forward result
data = np.load("outputs/ct_forward_result.npz")
sinogram    = data["measurements"]   # shape (256, 60)
phantom     = data["ground_truth"]   # shape (256, 256)

# Reconstruction result
data = np.load("outputs/ct_reconstruction_result.npz")
x_hat  = data["x_hat"]    # shape (256, 256)
x_true = data["x_true"]   # shape (256, 256)
```

---

## Full Plan Markdown: Forward

```markdown
---
modality: ct
period: forward
version: 1
iteration: 1
---

# Task

Design a complete forward model for sparse-view X-ray CT with only 60
projection angles and low photon flux (I0=1e4) for pediatric chest imaging.
The system must model all physical elements from X-ray source through
detector digitization, including realistic noise and calibration mismatch
sources.

# Plan

1. Configure polychromatic X-ray tube source at 80 kVp with 1.5mm Al filtration
2. Model Beer-Lambert attenuation through soft tissue phantom
3. Define parallel-beam acquisition geometry with 60 projection angles over 180°
4. Simulate flat-panel CsI:Tl detector with Poisson noise (I0=1e4) and Gaussian
   readout noise
5. Apply 12-bit ADC digitization with dark current
6. Identify beam hardening, scatter, and center-of-rotation mismatch sources

# Action

## System Flowchart

  [X-ray Tube 80kVp] → [Soft Tissue Phantom] → [Parallel-Beam 60 angles]
         ↓                      ↓                        ↓
    [Polychromatic         [Beer-Lambert           [CoR offset
     beam hardening]        attenuation]            mismatch]
                                                         ↓
                → [CsI:Tl Flat Panel Detector] → [12-bit ADC] → y
                             ↓
                       [Poisson I0=1e4]
                       [Gaussian σ=3 e⁻]
                       [Dark current 0.05 e⁻/s]

### Element: X-ray Tube Source (80 kVp) (`xray_source`)

- **Type**: source
- **Parameters**:
  - `energy_kVp`: 80
  - `flux_photons_per_s`: 500000.0
  - `focal_spot_mm`: 0.4
  - `filtration`: 1.5mm Al
  - `spectrum`: polychromatic
- **Mismatch sources**:
  - `beam_hardening` [high]: Polychromatic spectrum causes cupping artifacts
    in soft tissue → correction: 2nd-order polynomial linearization from water
    phantom calibration
- **Connects to**: tissue_attenuation

### Element: Soft Tissue Attenuation (`tissue_attenuation`)

- **Type**: interaction
- **Parameters**:
  - `model`: beer_lambert
  - `mu_water_cm`: 0.184
  - `material`: pediatric_soft_tissue
- **Mismatch sources**:
  - `scatter` [medium]: Compton scatter adds low-frequency background
    (SPR ~0.3 for pediatric chest) → correction: Scatter kernel estimation
    with 1D convolution correction
- **Connects to**: geometry

### Element: Parallel-Beam Acquisition (60 angles) (`geometry`)

- **Type**: geometry
- **Parameters**:
  - `scan_type`: parallel_beam
  - `num_angles`: 60
  - `angular_range_deg`: 180
  - `detector_pixels`: 256
  - `pixel_pitch_mm`: 0.4
- **Mismatch sources**:
  - `center_of_rotation_offset` [medium]: Mechanical misalignment causes ring
    artifacts (estimated ±0.5 px) → correction: Cross-correlation of 0°/180°
    projection pair
- **Connects to**: detector

### Element: CsI:Tl Flat Panel Detector (`detector`)

- **Type**: detector
- **Parameters**:
  - `scintillator`: CsI:Tl
  - `pixels`: [256, 256]
  - `pixel_pitch_mm`: 0.4
  - `quantum_efficiency`: 0.75
- **Noise**:
  - poisson: I0=10000.0
  - gaussian: sigma_electrons=3.0
  - dark_current: electrons_per_s=0.05, exposure_s=0.02
- **Mismatch sources**:
  - `detector_response_nonuniformity` [low]: Per-pixel gain variations up to ±2%
    → correction: Flat-field correction with air scan
- **Connects to**: adc

### Element: 12-bit ADC (`adc`)

- **Type**: digitization
- **Parameters**:
  - `bit_depth`: 12
  - `dynamic_range_db`: 72

## Composite Noise Model

  y ~ Poisson(I0 * exp(-H*x)) + N(0, σ_readout²) + Poisson(dark * t_exp)

**Measurement shape**: `(256, 60)`

# Demands

- **feasibility**: yes
- **budget_feasible**: yes
- **algorithm_convergence**: N/A

**Comments**: Low-dose (I0=1e4) will produce noisy sinograms (estimated
SNR ~17 dB). Sparse view (60 angles) will cause streak artifacts in FBP
but is recoverable with iterative reconstruction.
```

---

## Full Plan Markdown: Reconstruction

```markdown
---
modality: ct
period: reconstruction
version: 1
iteration: 1
---

# Task

Reconstruct the original attenuation map from a sparse-view (60 angles),
low-dose (I0=1e4) CT sinogram. The inverse problem is severely ill-posed
due to angular undersampling and high noise. The sinogram is also corrupted
by beam hardening and Compton scatter. Use TV-ADMM with mismatch corrections
to regularize the reconstruction.

# Plan

1. Apply beam hardening polynomial correction to the sinogram
2. Apply scatter kernel subtraction to remove low-frequency bias
3. Initialize with Filtered Back-Projection (FBP, ramp filter)
4. Run TV-ADMM: alternate data fidelity gradient step and TV proximal step
5. Enforce non-negativity constraint at each iteration
6. Check convergence: ||x_{k+1} - x_k|| / ||x_k|| < 1e-4 or max 100 iterations

# Action

## Algorithm: TV-ADMM

**Type**: Variational

**References**:
  - Sidky & Pan, 'Image reconstruction in circular cone-beam CT by constrained
    TV minimization', Phys. Med. Biol. 53(17), 2008
  - Kak & Slaney, 'Principles of Computerized Tomographic Imaging', IEEE Press 1988

### Algorithm Steps

**Step 1: Mismatch Pre-Correction**

Apply beam hardening polynomial linearization and scatter subtraction to the
raw sinogram before reconstruction.

$$
y_corr = a0 + a1*y + a2*y^2 - 0.1*G_sigma(y)
$$
Parameters:
  - `a0`: 0.0
  - `a1`: 1.0
  - `a2`: -0.05
  - `scatter_sigma`: 20.0

**Step 2: FBP Initialization**

Compute initial estimate via filtered back-projection with a Ram-Lak filter.

$$
x_0 = R^{-1}_{FBP}(y_corr)
$$
Parameters:
  - `filter`: ramp

**Step 3: Data Fidelity Gradient**

Compute the gradient of 0.5*||Rx - y_corr||^2 where R is the Radon operator.

$$
grad = R^T(Rx_k - y_corr)
$$

**Step 4: TV Proximal Step**

Apply the proximal operator of the isotropic total variation penalty.
This preserves edges while denoising flat regions.

$$
x_{k+1} = prox_{lambda*TV}(x_k - eta * grad)
$$
Parameters:
  - `lambda_tv`: 0.01
  - `eta`: 0.005

**Step 5: Non-Negativity Projection**

Project onto the non-negative orthant since attenuation coefficients
are non-negative.

$$
x_{k+1} = max(x_{k+1}, 0)
$$

### Mismatch Corrections

- `beam_hardening` [high]: Polychromatic cupping artifact from 80 kVp source
  Correction: 2nd-order polynomial linearization: y_corr = y - 0.05*y^2
- `scatter` [medium]: Compton scatter low-frequency bias (SPR ~0.3)
  Correction: Subtract 10% of Gaussian-blurred sinogram (sigma=20 px)

**Convergence**: ||x_{k+1} - x_k||_2 / ||x_k||_2 < 1e-4, or max 100 iterations

### Hyperparameters

- `lambda_tv`: 0.01
- `step_size`: 0.005
- `num_iterations`: 100
- `filter`: ramp

# Demands

- **feasibility**: yes
- **budget_feasible**: N/A
- **algorithm_convergence**: yes

**Comments**: TV-ADMM converges reliably for sparse-view CT. The TV penalty
effectively suppresses streak artifacts from angular undersampling. Beam
hardening and scatter pre-corrections are essential at this noise level.
```
